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
import traceback
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


def cat_tensor_dicts(tensor_dicts):
	tensors = {k: [] for k in tensor_dicts[0].keys()}
	for tensor_dict in tensor_dicts:
		for k, v in tensor_dict.items():
			tensors[k].append(v)
	_tensors = {k: torch.cat(v) for k, v in tensors.items()}
	return _tensors


class FeatureBag(Datablock):
	VERSION = 3
	@dataclass
	class CONFIG(Datablock.CONFIG):
		tilebag: PancanTileBag
		extractor: Callable

	@property
	def has_sideband(self):
		return hasattr(self.config.extractor, 'sideband_layers')

	def __post_init__(self):
		self.FILES = {
			'features': 'features.pt',
			'sideband': None
		}
		if self.has_sideband:
			self.FILES['sideband'] = {
				layer: f"{layer}.pt" for layer in self.config.extractor.sideband_layers
		}
		return self

	def __len__(self):
		return len(self.labels)

	def store(self, features, sideband=None):
		assert sideband is not None or not self.has_sideband, "Expected sideband tensors to store"
		assert self.has_sideband or sideband is None, "Need a sideband extractor to store sideband tensors"
		#TODO: check for consistency with self.config.tilebag
		self.__pre_build__()
		dbx.write_tensor(features, self.path('features', ensure_dirpath=True))
		if self.has_sideband:
			dbx.write_tensors(self.path('sideband', ensure_dirpath=True), **sideband)
		self._write_journal_entry(event="store")
		self.__post_build__()
		return self

	def read(self, topic):
		if topic == 'features':
			return dbx.read_tensor(self.path(topic))
		elif topic == 'sideband':
			if self.has_sideband:
				sideband = dbx.read_tensors(self.path('sideband'), *self.config.extractor.sideband_layers)
				return sideband
			else:
				return None
		else:
			raise ValueError(f"Unknown topic: {topic}")

	def UNSAFE_clear(self):
		for bag in self.bags:
			bag.UNSAFE_clear()
		super().UNSAFE_clear()

	@functools.cached_property
	def features(self):
		return self.read('features')
	
	@functools.cached_property
	def sideband(self):
		return self.read('sideband') if self.has_sideband else None
		
	@functools.cached_property
	def labels(self):
		return self.config.tilebag.labels

	@property
	def label(self):
		return self.config.tilebag.label


class FeatureBags(Datablock):
	VERSION = 2
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
								      spec=dict(tilebag=dbx.quote(tilebag), extractor=self.spec['extractor'],))
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
			done_queue = queue.Queue()
			abort_event = threading.Event()
			progress_bar = tqdm.tqdm(total=len(remaining_bags))
			feature_bag_lists = np.array_split(remaining_bags, len(self.devices))
			threads = [
				threading.Thread(target=self.__build_bags__, args=(feature_bag_list, device, result_queue, done_queue, abort_event, progress_bar))
				for feature_bag_list, device in zip(feature_bag_lists, self.devices)
			]
			for thread in threads:
				thread.start()
			while len(bag_lens) < len(self.bags):
				success, payload = result_queue.get()
				if success:
					bag_len = payload
					bag_lens.append(bag_len)
					e = None
				else:
					e = payload
					break
			for _ in range(len(self.devices)):
				done_queue.put(None)
			for thread in threads:
				thread.join()
			if e is not None:
				raise e
		dbx.write_tensor(torch.tensor(bag_lens), self.path('bag_lens', ensure_dirpath=True))
		return self

	def __build_bags__(self, featurebags: Sequence[FeatureBag], device: str, result_queue: queue.Queue, done_queue: queue.Queue, abort_event: threading.Event, progress_bar):
		self.log.debug(f"Building {len(featurebags)} feature bags on device: {device}")
		extractor = copy.deepcopy(self.config.extractor).to(device).eval()
		for featurebag in featurebags:
			exception = None
			try:
				featurelen = self.__build_bag__(featurebag, extractor, device, abort_event)
			except Exception as e:
				self.log.info(f"ERROR building feature bag {featurebag.hashpath()}: {e}")
				tbstr = '\n'.join(traceback.format_tb(e.__traceback__))
				self.log.verbose(f"TRACEBACK:\n{tbstr}")
				featurelen = 0
				exception = e
			if exception is not None:
				result_queue.put((False, exception))
				break
			result_queue.put((True, featurelen))
			progress_bar.update(1)
		while True:
			item = done_queue.get()
			if item is None:
				break

	def __build_bag__(self, featurebag: FeatureBag, extractor: Callable, device: str, abort_event):
		self.log.verbose(f"Building new feature bag {featurebag.hashpath()} on device: {device}")
		tilebag = featurebag.config.tilebag
		feature_list = []
		sideband_list = []
		for k in range(math.ceil(len(tilebag.tiles)/self.gpu_batch_size)):
			if abort_event.is_set():
				return
			m = k*self.gpu_batch_size
			n = min((k+1)*self.gpu_batch_size, len(tilebag.tiles))
			batch = tilebag.tiles[m:n].to(device)
			self.log.verbose(f"Evaluating batch {k}: {m}:{n} out of {len(tilebag.tiles)} on device: {device}")
			features_ = extractor(batch).to('cpu')
			del batch
			if hasattr(extractor, 'sideband'):
				sideband = tensors_to_device(extractor.sideband, 'cpu', detach=True)
				sideband_list.append(sideband)
				del sideband
			gc.collect()
			torch.cuda.empty_cache()
			self.log.verbose(f"done")
			feature_list.append(features_)
		features = torch.cat(feature_list)
		if hasattr(extractor, 'sideband'):
			sideband = cat_tensor_dicts(sideband_list)
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
