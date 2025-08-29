"""
	Examples:
		#BASH:
			dbx "autopath.pancan.bags.FeatureBag(batch_size=16, device='cuda', verbose=True, debug=True, cfg=dict(slideshard='@autopath.pancan.tiles.PancanSlideShard()', extractor='@autopath.gigaq.dinov2.models.BackboneEvaluator()', split='test')).build().read()"
		#PYTHON:
			import autopath.pancan.features;featurebag = autopath.pancan.features.FeatureBag(
				gpu_batch_size=16, verbose=True, debug=True, 
				cfg=dict(slideshard="@autopath.pancan.images.PancanSlideShard()", 
						 extractor="@autopath.gigaq.dinov2.backbone.BackboneEvaluator()",
						 split="test",
			)).build().read()
			#
			import autopath.pancan.features;featurebag = autopath.pancan.features.FeatureBatch(
				builder='@autopath.pancan.features.TorchMultiprocessingBatchBuilder(num_gpus=1)',
				gpu_batch_size=16, 
				verbose=True, 
				debug=True, 
				cfg=dict(slidebatch="@autopath.pancan.images.PancanSlideBatch()", 
						 split="test",
						 max_bag_count=1,
						 extractor="@autopath.gigaq.dinov2.backbone.BackboneEvaluator()",
			)).build().read()

"""
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

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq


import torch
import rich

import dbx
from dbx import (
	Logger,
	Datablock,
	Databatch,
	BatchBuilder,
	datablock_method,
)
from .images import PancanSlideShard, PancanSlideBatch

#from ..gigaq.dinov2.models import BackboneEvaluator

DBKSPACE = os.environ.get("DBKSPACE", "/mnt/labshare/PROJECTS/GIGAQ/dbx")
DBKREPO = os.environ.get("DBKREPO", f"{os.environ.get('HOME')}/autopath")


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


def datablock_multiprocessing_build(
		id,
		datablock_cls,
		datablock_method_build_kwargslist,
		progress_bar=None,
		progress_task=None,
):
		result = datablock_method(datablock_cls, 'build', **datablock_method_build_kwargslist[id])
		if progress_bar is not None and progress_task is not None:
			progress_bar.advance(progress_task, 1)
		return result
	

class TorchMultiprocessingBatchBuilder(BatchBuilder):
	def __init__(self, *, num_gpus: int = 1):
		self.num_gpus = num_gpus

	@property
	def tag(self):
		return "mp"
	
	def __call__(self, datablock_cls, datablock_method_build_kwargslist):
		n_kwargs = len(datablock_method_build_kwargslist)
		pb = rich.progress.Progress() 
		pb.add_task(
			"Speed: ",
			progress_type="speed",
			total=None
		)
		slide_task = pb.add_task(
			"Building {datablock_cls.__name__} ...",
			progress_type="slide_progress",
			total=n_kwargs,
		)
		pb.start()
		with MultiprocessProgress(pb) as mp_pb:
			torch.multiprocessing.spawn(
				datablock_multiprocessing_build,
				args=(datablock_cls,
					  datablock_method_build_kwargslist,
					  mp_pb.tracker,
					  slide_task,
				),       
				nprocs=self.num_gpus,
			)


class FeatureBag(Datablock):
	@dataclass
	class CONFIG(Datablock.CONFIG):
		extractor: Callable
		slideshard: PancanSlideShard
		split: str = "test"

	def __init__(self, *args, gpu_batch_size: int = 16, **kwargs):
		super().__init__(*args, **kwargs)
		self.gpu_batch_size = gpu_batch_size
		self.device = 'cuda' #TODO: inline?

	def __post_init__(self):
		self.tiles, self.origin, self.slide = self.config.slideshard.read(self.config.split)
		self.FILE = f"{self.slide}.pt"

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


class FeatureBatch(Databatch):
	DATABLOCK = FeatureBag
	@dataclass
	class CONFIG(Databatch.CONFIG):
		extractor: Callable
		slidebatch: PancanSlideBatch
		max_bag_count: Optional[int] = None
		split: str = "test"

	DEFAULT_BUILDER = TorchMultiprocessingBatchBuilder(num_gpus=1)

	def __init__(self, 
				root: str = None,
				verbose: bool = False,
				debug: bool = False,
				builder: BatchBuilder = None,
				gpu_batch_size: int = 16,
				*,
				cfg: Optional[Union[str,dict]] = None,
	):
		builder = builder or self.DEFAULT_BUILDER
		super().__init__(root, verbose, debug, builder, cfg=cfg)
		self.gpu_batch_size = gpu_batch_size

	def datablocks(self):
		slideshards = list(self.config.slidebatch.datablocks())
		if self.config.max_bag_count is not None:
			slideshards = slideshards[:self.config.max_bag_count]
		datablocks = [FeatureBag(
			cfg=dict(extractor=self.cfg.get('extractor'), slideshard=slideshard, split=self.cfg.get('split')),
			verbose=self.verbose,
			debug=self.debug,
			gpu_batch_size=self.gpu_batch_size,
		) for slideshard in slideshards]
		return datablocks

