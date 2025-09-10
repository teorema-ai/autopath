from dataclasses import dataclass, asdict
import datetime
import gc
import json
import math
import multiprocessing as mp
import os
import pickle
import sys
import threading
import time
from typing import List, Dict, Optional, Union, Tuple, Callable

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

from .images import PancanTileBag, PancanTileBags


class MultiprocessProgressTracker:
	"""Wrapper for a rich.progress tracker that can be shared across processes."""

	def __init__(self, tasks):
		ctx = mp.get_context('spawn')
		self.mp_values = {
			task.id: ctx.Value('i', task.completed)
			for task in tasks
		}

	def advance(self, id, amount):
		with self.mp_values[id].get_lock():
			self.mp_values[id].value += amount

	def __getitem__(self, id):
		return self.mp_values[id].value


class MultiprocessProgress:
	"""Wrapper for a rich.progress bar that can be shared across processes."""

	def __init__(self, pb):
		self.pb = pb
		self.tracker = MultiprocessProgressTracker(self.pb.tasks)
		self.should_stop = False

	def _update_progress(self):
		while not self.should_stop:
			for task in self.pb.tasks:
				self.pb.update(task.id, completed=self.tracker[task.id])
			time.sleep(0.1)

	def __enter__(self):
		self._thread = threading.Thread(target=self._update_progress)
		self._thread.start()
		return self

	def __exit__(self, *args):
		self.should_stop = True
		self._thread.join()


def datablock_method_multiprocessing(
		id,
		datablock_method_args_kwargs_list,
		progress_bar=None,
		progress_task=None,
):
		args, kwargs = datablock_method_args_kwargs_list[id]
		result = datablock_method(*args, **kwargs)
		if progress_bar is not None and progress_task is not None:
			progress_bar.advance(progress_task, 1)
		return result
	

class TorchMultiprocessingDatabatchBuilder(DatabatchBuilder):
	def __init__(self, *, num_gpus: int = 1):
		self.num_gpus = num_gpus
	
	def __call__(self, datablock_method_args_kwargs_list):
		N = len(datablock_method_args_kwargs_list)
		pb = rich.progress.Progress() 
		pb.add_task(
			"Speed: ",
			progress_type="speed",
			total=None
		)
		slide_task = pb.add_task(
			"Building {datablock_cls.__name__} ...",
			progress_type="slide_progress",
			total=N,
		)
		pb.start()
		with MultiprocessProgress(pb) as mp_pb:
			torch.multiprocessing.spawn(
				datablock_method_multiprocessing,
				args=(datablock_method_args_kwargs_list,
					  mp_pb.tracker,
					  slide_task,
				),       
				nprocs=self.num_gpus,
			)


class FeatureBag(Datablock):
	@dataclass
	class CONFIG(Datablock.CONFIG):
		extractor: Callable
		tilebag: PancanTileBag

	def __init__(self, *args, device: str = 'cuda', gpu_batch_size: int = 16, **kwargs):
		super().__init__(*args, **kwargs)
		self.gpu_batch_size = gpu_batch_size
		self.device = device

	def __post_init__(self):
		self.label = self.config.tilebag.label
		self.name = self.config.tilebag.name
		self.FILE = f"{self.name}-features.pt"

	@property
	def tiles(self):
		tiles = self.config.tilebag.read('tiles')
		return tiles

	def __build__(self):
		tiles = self.tiles
		feature_list = []
		for k in range(math.ceil(len(tiles)/self.gpu_batch_size)):
			m = k*self.gpu_batch_size
			n = min((k+1)*self.gpu_batch_size, len(tiles))
			batch = tiles[m:n].to(self.device)
			self.log.debug(f"Evaluating batch {k}: {m}:{n} out of {len(tiles)} on device: {self.device}...")
			features_ = self.config.extractor(batch).to('cpu')
			del batch
			gc.collect()
			torch.cuda.empty_cache()
			self.log.debug(f"done")
			feature_list.append(features_)
		features = torch.cat(feature_list)
		dbx.write_tensor(features, self.path(ensure_dirpath=True))
		return self

	def read(self):
		features = dbx.read_tensor(self.path())
		return features


class FeatureBags(Databatch):
	DATABLOCK = FeatureBag
	FILE = "breadcrumbs"
	@dataclass
	class CONFIG(Datablock.CONFIG):
		extractor: Callable
		tilebags: PancanTileBags

	def __init__(self, *args, device: str = 'cuda', gpu_batch_size: int = 16, **kwargs):
		super().__init__(*args, **kwargs)
		self.gpu_batch_size = gpu_batch_size
		self.device = device

	def datablocks(self):
		return [FeatureBag(root=self.root, 
						   device=self.device,
						   gpu_batch_size=self.gpu_batch_size,
						   spec=dict(extractor=self.spec.get('extractor'), tilebag=tilebag),
						   verbose=self.verbose,
						   debug=self.debug,
						   )
				for tilebag in self.config.tilebags.datablocks()
		]
	
	def __build__(self):
		self.leave_breadcrumbs()
		return self

	@property
	def bags(self):
		return self.datablocks()


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


				
			



		

	
	


