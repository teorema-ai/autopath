from dataclasses import dataclass, asdict, replace
import datetime
import functools
import gc
import json
import math
import multiprocessing as mp
import os
import pickle
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

from .tiles import PancanTileBag, PancanTileBatch


class FeatureBag(Datablock):
	VERSION = 1
	@dataclass
	class CONFIG(Datablock.CONFIG):
		tilebag: PancanTileBag

	def __post_init__(self):
		self.name = self.config.tilebag.name
		self.label = self.config.tilebag.label
		self.FILE = f"{self.name}-features.pt"

	def store(self, features):
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


class FeatureBags(Datablock):
	VERSION = 1
	FILES = {'bag_lens': 'bag_lens.pt'}
	@dataclass
	class CONFIG(Datablock.CONFIG):
		extractor: Callable
		tilebatch: PancanTileBatch
		lo: int = 0
		hi: Optional[int] = None

	def __init__(self, *args, device: str = 'cuda', gpu_batch_size: int = 16, **kwargs):
		super().__init__(*args, **kwargs)
		self.gpu_batch_size = gpu_batch_size
		self.device = device

	def __post_init__(self):
		if self.config.hi is None:
			self.config = replace(self.config, hi=len(self.config.tilebatch))
		return self

	@functools.cached_property
	def bags(self):
		featurebags = [FeatureBag(root=self.root if not self._autoroot else None,
								  spec=dict(tilebag=dbx.quote(tilebag)))
						for tilebag in self.config.tilebatch.datablocks()[self.config.lo:self.config.hi]
		]
		return featurebags

	def __len__(self):
		return len(self.bags)

	def __build__(self):
		"""Single-process, but, potentially, a multithreaded build."""
		bag_lens = []
		self.log.verbose("Building {len(bags)} feature bags")
		if self.verbose:
			featurebag_itor = tqdm.tqdm(self.bags)
		else:
			featurebag_itor = self.bags
		for featurebag in featurebag_itor:
			bag_len = self.__build_bag__(featurebag)
			bag_lens.append(bag_len)
		dbx.write_tensor(torch.tensor(bag_lens), self.path('bag_lens', ensure_dirpath=True))
		return self

	def __build_bag__(self, featurebag: FeatureBag):
		tilebag = featurebag.config.tilebag
		feature_list = []
		for k in range(math.ceil(len(tilebag.tiles)/self.gpu_batch_size)):
			m = k*self.gpu_batch_size
			n = min((k+1)*self.gpu_batch_size, len(tilebag.tiles))
			batch = tilebag.tiles[m:n].to(self.device)
			self.log.debug(f"Evaluating batch {k}: {m}:{n} out of {len(tilebag.tiles)} on device: {self.device}...")
			features_ = self.config.extractor(batch).to('cpu')
			del batch
			gc.collect()
			torch.cuda.empty_cache()
			self.log.debug(f"done")
			feature_list.append(features_)
		features = torch.cat(feature_list)
		#
		featurebag.store(features)
		return len(features)

	def __read__(self, topic):
		assert topic == "bag_lens", f"Unknown topic: {topic}"
		bag_lens = dbx.read_tensor(self.path(topic))
		return bag_lens

	@functools.cached_property
	def bag_lens(self):
		return self.read('bag_lens')


"""
#TODO: #REMOVE
class _FeatureBatch(Databatch):
	DATABLOCK = FeatureBag
	@dataclass
	class CONFIG(Databatch.CONFIG):
		extractor: Callable
		slidebatch: PancanSlideBatch
		max_bag_count: Optional[int] = None
		split: str = "test"

	def __init__(self, *args, gpu_batch_size: int = 16, **kwargs):
		
		super().__init__(*args, **kwargs)
		self.gpu_batch_size = gpu_batch_size

	def datablocks(self):
		slideshards = list(self.config.slidebatch.datablocks())
		if self.config.max_bag_count is not None:
			slideshards = slideshards[:self.config.max_bag_count]
		slideshardreprs = ["@"+repr(slideshard) for slideshard in slideshards]
		datablocks = [FeatureBag(
			root=self.root,
			spec=dict(extractor=self.spec.get('extractor'), slideshard=slideshardrepr, split=self.spec.get('split')),
			verbose=self.verbose,
			debug=self.debug,
			gpu_batch_size=self.gpu_batch_size,
		) for slideshardrepr in slideshardreprs]
		self.log.debug(f"{self.anchor()}: {datablocks=}")
		return datablocks

	def valid_bags(self):
		return [bag for bag in self.datablocks() if bag.valid()]

	def __read__(self):
		bagtensors = []
		baglabels = []
		for bag in self.datablocks():
			bagtensor, baglabel, _ = bag.read()
			bagtensors.append(bagtensor)
			baglabels.extend([baglabel]*bagtensor.shape[0])
		batchtensor = torch.cat(bagtensors)
		return batchtensor, baglabels

	#TODO: IMPL
	def dataloader(self, *, num_workers=None):
		...
		

class FeatureShard(Datablock):
	FILE = "shard.npz"
	@dataclass
	class CONFIG(Datablock.CONFIG):
		extractor: Callable
		tileset: PancanTileset
		shard_start: int
		shard_size: int

	def __init__(self, 
				*args, 
				device: str = 'cuda', 
				gpu_batch_size: int = 16,
				**kwargs):
		self.gpu_batch_size = gpu_batch_size
		self.device = device
		super().__init__(*args, **kwargs)

	def __post_init__(self):
		assert self.config.shard_size % self.gpu_batch_size == 0, "shard_size must be a multiple of gpu_batch_size"
		self.dataset = self.config.tileset.dataset
		self.batches_per_shard = self.config.shard_size//self.gpu_batch_size
		self._tensor = None
		self._labels = None
		return self

	def dataset_slice(self, lo, hi):
		tensors, labels = zip(*torch.utils.data.Subset(self.dataset, range(lo, hi)))
		return torch.stack(tensors), labels

	def __build__(self):
		self.log.verbose(f"Building a feature shard from tile dataset of size {len(self.dataset)} of size {self.config.shard_size} starting at {self.config.shard_start} using {self.gpu_batch_size=}")
		tensor_shard, label_shard = self.dataset_slice(self.config.shard_starg, self.config.shard_start + self.config.shard_size)
		if self.verbose:
			batch_itor = tqdm.tqdm(range(self.batches_per_shard))
		else:
			batch_itor = range(self.batches_per_shard)
		feature_shard_batches = []
		for batch_idx in batch_itor:
			batch = tensor_shard[batch_idx*self.gpu_batch_size:(batch_idx+1)*self.gpu_batch_size].to(self.device)
			feature_batch = self.config.extractor(batch).to('cpu')
			del batch
			gc.collect()
			torch.cuda.empty_cache()
			feature_shard_batches.append(feature_batch)
		feature_shard = torch.cat(feature_shard_batches)
		dbx.write_npz(self.path(ensure_dirpath=True), 
					  features=feature_shard.numpy(), 
					  labels=np.array(label_shard))
		del feature_shard
		del label_shard
		gc.collect()
		torch.cuda.empty_cache()
		return self

	def __read__(self):
		self._tensor, self._labels = dbx.read_npz(self.path(topic), 'features', 'labels')
		return self._tensor, self._labels

	@property
	def tensor(self):
		if self._tensor is None:
			self.__read__()
		return self._tensor

	@property
	def labels(self):
		if self._labels is None:
			self.__read__()
		return self._labels

	def __len__(self):
		return len(self.tensor)


class FeatureBatch(Databatch):
	DATABLOCK = FeatureShard
	FILES = {'lens': 'lens.npz'}
	@dataclass
	class CONFIG(Datablock.CONFIG):
		extractor: Callable
		tileset: PancanTileset
		shard_size: int

	def __init__(self, 
				*args, 
				device: str = 'cuda', 
				gpu_batch_size: int = 16,
				**kwargs):
		self.gpu_batch_size = gpu_batch_size
		self.device = device
		super().__init__(*args, **kwargs)

	def __post_init__(self):
		assert self.config.shard_size % self.gpu_batch_size == 0, "shard_size must be a multiple of gpu_batch_size"
		self.dataset = self.config.tileset.dataset
		self.num_shards = math.ceil(len(self.dataset)/self.config.shard_size)
		return self

	def datablocks(self):
		self.log.verbose(f"Building feature_shard datablocks from tile dataset of size {len(self.dataset)} with {self.num_shards} shards of size {self.config.shard_size} using {self.gpu_batch_size=}")
		if self.verbose:
			shard_itor = tqdm.tqdm(range(self.num_shards))
		else:
			shard_itor = range(self.num_shards)
		tagprefix = (f"{self.tag}/" if self.tag is not None else "") + f"{self.anchor()}/{self.hash}"
		datablocks = [
			FeatureShard(root=self.root,
					     spec=dict(extractor=self.spec.get('extractor'), 
									tileset=self.spec.get('tileset'), 
									shard_start=i*self.config.shard_size, 
									shard_size=self.config.shard_size),	
						 tag=f"{tagprefix}/{i}",					 
						 verbose=self.verbose,
					     debug=self.debug,
					     device=self.device,
					     gpu_batch_size=self.gpu_batch_size,
			)
			for i in shard_itor
		]
		return self

	def __build__(self):
		super().__build__()
		self.log.verbose(f"Computing shard lens")
		lens = [len(dbk) for dbk in self.datablocks()]
		dbx.write_npz(self.path(topic='lens', ensure_dirpath=True), lens=np.array(lens))
		return self

	def dataset(self, split, *, transform=None):
		dataset = LabeledBagDataset(
						  bags=self.datablocks(), 
						  bag_lens=self.read('lens'), 
						  transform=transform, 
						  debug=self.debug, 
						  verbose=self.verbose,
						  log=self.log,
		)
		return dataset
"""


				
			



		

	
	


