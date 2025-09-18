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

from .tiles import PancanTileShard, PancanTileShards


class FeatureShard(Datablock):
	VERSION = 1
	FILE = "features.pt"
	@dataclass
	class CONFIG(Datablock.CONFIG):
		tileshard: PancanTileShard

	def __len__(self):
		return len(self.labels)

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
		
	@functools.cached_property
	def labels(self):
		return self.config.tileshard.labels


class FeatureShards(Datablock):
	VERSION = 1
	FILES = {'shard_lens': 'shard_lens.pt'}
	@dataclass
	class CONFIG(Datablock.CONFIG):
		extractor: Callable
		tileshards: PancanTileShards
		lo: int = 0
		hi: Optional[int] = None

	def __init__(self, *args, device: str = 'cuda', gpu_batch_size: int = 16, **kwargs):
		super().__init__(*args, device=device, gpu_batch_size=gpu_batch_size, **kwargs)
	
	def __post_init__(self):
		if self.config.hi is None:
			self.config = replace(self.config, hi=len(self.config.tileshards))
		return self

	@functools.cached_property
	def shards(self):
		featureshards = [FeatureShard(root=self.root if not self._autoroot else None,
								      spec=dict(tileshard=dbx.quote(tileshard)))
						for tileshard in self.config.tileshards.datablocks()[self.config.lo:self.config.hi]
		]
		return featureshards

	def __len__(self):
		return len(self.shards)

	def __build__(self):
		"""Single-process, but, potentially, a multithreaded build."""
		self.log.verbose(f"Building {len(self.shards)} feature shards")
		if self.verbose:
			featureshard_itor = tqdm.tqdm(self.shards)
		else:
			featureshard_itor = self.shards
		shard_lens = []
		for featureshard in featureshard_itor:
			if featureshard.valid():
				self.log.verbose(f"Skipping existing feature shard {featureshard.hashpath()}")
				shard_len = len(featureshard)
			else:
				shard_len = self.__build_shard__(featureshard)
			shard_lens.append(shard_len)
		dbx.write_tensor(torch.tensor(shard_lens), self.path('shard_lens', ensure_dirpath=True))
		return self

	def __build_shard__(self, featureshard: FeatureShard):
		tileshard = featureshard.config.tileshard
		feature_list = []
		for k in range(math.ceil(len(tileshard.tiles)/self.gpu_batch_size)):
			m = k*self.gpu_batch_size
			n = min((k+1)*self.gpu_batch_size, len(tileshard.tiles))
			batch = tileshard.tiles[m:n].to(self.device)
			self.log.detailed(f"Evaluating batch {k}: {m}:{n} out of {len(tileshard.tiles)} on device: {self.device}...")
			features_ = self.config.extractor(batch).to('cpu')
			del batch
			gc.collect()
			torch.cuda.empty_cache()
			self.log.detailed(f"done")
			feature_list.append(features_)
		features = torch.cat(feature_list)
		#
		featureshard.store(features)
		return len(features)

	def __read__(self, topic):
		shard_lens = dbx.read_tensor(self.path(topic))
		return shard_lens

	@functools.cached_property
	def shard_lens(self):
		return self.read('shard_lens')
