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


class FeatureBag(Datablock):
	VERSION = 1
	FILE = "features.pt"
	@dataclass
	class CONFIG(Datablock.CONFIG):
		tilebag: PancanTileBag

	def __len__(self):
		return len(self.labels)

	def store(self, features):
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
								      spec=dict(tilebag=dbx.quote(tilebag)))
						for tilebag in self.config.tilebags.datablocks()[self.config.lo:self.config.hi]
		]
		return featurebags

	def __len__(self):
		return len(self.bags)

	def __build__(self):
		self.log.verbose(f"Building {len(self.bags)} feature bags")
		result_queue = queue.Queue()
		stop_queue = queue.Queue()
		progress_bar = tqdm.tqdm(total=len(self.bags))
		feature_bag_lists = np.array_split(self.bags, len(self.devices))
		threads = [
			threading.Thread(target=self.__build_bags__, args=(feature_bag_list, device, result_queue, stop_queue, progress_bar))
			for feature_bag_list, device in zip(feature_bag_lists, self.devices)
		]
		for thread in threads:
			thread.start()
		bag_lens = []
		while len(bag_lens) < len(self.bags):
			bag_lens.append(result_queue.get())
		for _ in range(len(self.devices)):
			stop_queue.put(None)
		for thread in threads:
			thread.join()
		dbx.write_tensor(torch.tensor(bag_lens), self.path('bag_lens', ensure_dirpath=True))
		return self

	def __build_bags__(self, featurebags: Sequence[FeatureBag], device: str, result_queue: queue.Queue, stop_queue: queue.Queue, progress_bar):
		bag_lens = []
		for featurebag in featurebags:
			featurelen = self.__build_bag__(featurebag, device)
			bag_lens.append(featurelen)
			result_queue.put(featurelen)
			progress_bar.update(1)
		while True:
			item = stop_queue.get()
			if item is None:
				break

	def __build_bag__(self, featurebag: FeatureBag, device: str):
		if featurebag.valid():
			self.log.verbose(f"Skipping existing feature bag {featurebag.hashpath()}")
			lenfeatures = len(featurebag)
		else:
			tilebag = featurebag.config.tilebag
			feature_list = []
			for k in range(math.ceil(len(tilebag.tiles)/self.gpu_batch_size)):
				m = k*self.gpu_batch_size
				n = min((k+1)*self.gpu_batch_size, len(tilebag.tiles))
				batch = tilebag.tiles[m:n].to(device)
				self.log.detailed(f"Evaluating batch {k}: {m}:{n} out of {len(tilebag.tiles)} on device: {device}...")
				features_ = self.config.extractor(batch).to('cpu')
				del batch
				gc.collect()
				torch.cuda.empty_cache()
				self.log.detailed(f"done")
				feature_list.append(features_)
			features = torch.cat(feature_list)
			#
			featurebag.store(features)
			lenfeatures = len(features)
		return lenfeatures

	def __read__(self, topic):
		bag_lens = dbx.read_tensor(self.path(topic))
		return bag_lens

	@functools.cached_property
	def bag_lens(self):
		return self.read('bag_lens')
