import copy
from dataclasses import dataclass, asdict, replace
import datetime
import functools
import gc
import json
import math
import multiprocessing as mp
import os
import pickle
import queue
import sys
import threading
import time
from typing import List, Dict, Optional, Union, Tuple, Callable, Sequence


import fsspec
import tqdm

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq


import torch
import torch.nn.functional as F
import rich

import dbx
from dbx import (
	Logger,
	Datablock,
	Databatch,
	DatabatchBuilder,
	datablock_method,
)

from .tiles import PancanTileBag, PancanTileBags


def tensors_to_device(tensors, device, *, detach: bool = False):
	_tensors = {k: v.to(device) for k, v in tensors.items()}
	if detach:
		_tensors = {k: v.detach() for k, v in _tensors.items()} 
	return _tensors


def cat_tensors(tensors):
	_tensors = {k: torch.cat(v) for k, v in tensors.items()}
	return _tensors


class FeatureBag(Datablock):
	VERSION = 1
	FILE = 'features.pt'

	@dataclass
	class CONFIG(Datablock.CONFIG):
		tilebag: PancanTileBag
		extractor: Callable

	def __len__(self):
		return len(self.labels)

	def store(self, features,):
		#TODO: check for consistency with self.config.tilebag
		self.__pre_build__()
		dbx.write_tensor(features, self.path(ensure_dirpath=True))
		self._write_journal_entry(event="store")
		self.__post_build__()
		return self

	def read(self):
		features = dbx.read_tensor(self.path())
		return features

	def features(self):
		return self.read()
		
	@functools.cached_property
	def labels(self):
		return self.config.tilebag.labels

	@property
	def label(self):
		return self.config.tilebag.label


class FeatureBags(Datablock):
	VERSION = 1
	FILES = {'bag_lens': 'bag_lens.pt'}
	@dataclass
	class CONFIG(Datablock.CONFIG):
		extractor: Callable
		tilebags: PancanTileBags
		lo: int = 0
		hi: Optional[int] = None

	def __init__(self, *args, devices: list[str] = 'cuda:0', gpu_batch_size: int = 16, **kwargs):
		super().__init__(*args, devices=devices, gpu_batch_size=gpu_batch_size, **kwargs)
	
	def __post_init__(self):
		if self.config.hi is None:
			self.config = replace(self.config, hi=len(self.config.tilebags))
		if isinstance(self.devices, str):
			self.devices = [self.devices]
		return self

	@functools.cached_property
	def bags(self):
		featurebags = [FeatureBag(root=self.root if not self._autoroot else None,
								      spec=dict(tilebag=dbx.quote(tilebag), extractor=self.spec.extractor,))
						for tilebag in self.config.tilebags.datablocks()[self.config.lo:self.config.hi]
		]
		return featurebags

	def __len__(self):
		return len(self.bags)

	def __build__(self):
		self.log.verbose(f"Building {len(self.bags)} feature bags")
		bag_lens = []
		remaining_bags = []
		for featurebag in self.bags:
			if featurebag.valid():
				self.log.verbose(f"Skipping existing feature bag {featurebag.hashpath()}")
				bag_lens.append(len(featurebag))
			else:
				remaining_bags.append(featurebag)
		if len(remaining_bags) > 0:
			result_queue = queue.Queue()
			stop_queue = queue.Queue()
			progress_bar = tqdm.tqdm(total=len(remaining_bags))
			feature_bag_lists = np.array_split(remaining_bags, len(self.devices))
			threads = [
				threading.Thread(target=self.__build_bags__, args=(feature_bag_list, device, result_queue, stop_queue, progress_bar))
				for feature_bag_list, device in zip(feature_bag_lists, self.devices)
			]
			for thread in threads:
				thread.start()
			while len(bag_lens) < len(self.bags):
				bag_lens.append(result_queue.get())
			for _ in range(len(self.devices)):
				stop_queue.put(None)
			for thread in threads:
				thread.join()
		dbx.write_tensor(torch.tensor(bag_lens), self.path('bag_lens', ensure_dirpath=True))
		return self

	def __build_bags__(self, featurebags: Sequence[FeatureBag], device: str, result_queue: queue.Queue, stop_queue: queue.Queue, progress_bar):
		self.log.debug(f"Building {len(featurebags)} feature bags on device: {device}")
		extractor = copy.deepcopy(self.config.extractor).to(device).eval()
		bag_lens = []
		for featurebag in featurebags:
			featurelen = self.__build_bag__(featurebag, extractor, device)
			bag_lens.append(featurelen)
			result_queue.put(featurelen)
			progress_bar.update(1)
		while True:
			item = stop_queue.get()
			if item is None:
				break
		return bag_lens

	def __build_bag__(self, featurebag: FeatureBag, extractor: Callable, device: str):
		self.log.verbose(f"Building new feature bag {featurebag.hashpath()} on device: {device}")
		tilebag = featurebag.config.tilebag
		feature_list = []
		sideband_list = []
		for k in range(math.ceil(len(tilebag.tiles)/self.gpu_batch_size)):
			m = k*self.gpu_batch_size
			n = min((k+1)*self.gpu_batch_size, len(tilebag.tiles))
			batch = tilebag.tiles[m:n].to(device)
			self.log.detailed(f"Evaluating batch {k}: {m}:{n} out of {len(tilebag.tiles)} on device: {device}")
			features_ = extractor(batch).to('cpu')
			del batch
			if hasattr(extractor, 'sideband'):
				sideband = tensors_to_device(extractor.sideband, 'cpu', detach=True)
				del sideband
			gc.collect()
			torch.cuda.empty_cache()
			self.log.detailed(f"done")
			if hasattr(extractor, 'sideband'):
				sideband_list.append(sideband)
			feature_list.append(features_)
		features = torch.cat(feature_list)
		if hasattr(extractor, 'sideband'):
			sideband = cat_tensors(sideband_list)
			featurebag.store(features, sideband)
		else:
			featurebag.store(features)
		lenfeatures = len(features)
		return lenfeatures

	def __read__(self, topic):
		bag_lens = dbx.read_tensor(self.path(topic))
		return bag_lens

	@functools.cached_property
	def bag_lens(self):
		return self.read('bag_lens')


class SidebandFeatureBag(FeatureBag):
	VERSION = 1

	@dataclass
	class CONFIG(Datablock.CONFIG):
		tilebag: PancanTileBag
		extractor: Callable

	def __post_init__(self):
		self.FILES = {
			'features': 'features.pt',
			'sideband': {layer: f"{layer}.pt" for layer in self.config.extractor.sideband}
		}
		return self

	def __len__(self):
		return len(self.labels)

	def store(self, features, sideband):
		#TODO: check for consistency with self.config.tilebag
		self.__pre_build__()
		dbx.write_tensor(features, self.path('features', ensure_dirpath=True))
		dbx.write_tensors(self.path('sideband', ensure_dirpath=True), **sideband)
		self._write_journal_entry(event="store")
		self.__post_build__()
		return self

	def read(self, topic):
		if topic == 'features':
			return dbx.read_tensor(self.path('features'))
		elif topic == 'sideband':
			return dbx.read_tensors(self.path('sideband'), *self.extractor.sideband.keys())

	def features(self):
		return self.read('features')
	
	def sideband(self):
		return self.read('sideband')

	
class SidebandFeatureBags(FeatureBags):
	@functools.cached_property
	def bags(self):
		featurebags = [SidebandFeatureBag(root=self.root if not self._autoroot else None,
								          spec=dict(tilebag=dbx.quote(tilebag), 
													extractor=self.spec.extractor,))
						for tilebag in self.config.tilebags.datablocks()[self.config.lo:self.config.hi]
		]
		return featurebags