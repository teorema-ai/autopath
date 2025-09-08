from dataclasses import dataclass
import functools
import itertools
import math
from math import floor
import os
from typing import Optional

import tqdm

import fsspec
import numpy as np

import torch
import torchvision
from torch.utils.data import Dataset

import ray

import slideflow as sf

import dbx
from dbx import Logger, Datablock, Databatch

from autopath.tools.dataset import LabeledBagDataset


logger = Logger()

CPTAC_ROOT = os.environ.get("CPTAC_ROOT", "/mnt/labshare/SLIDES/CPTAC_downloads")
CPTAC_SAMPLE_SHARD = os.path.join(CPTAC_ROOT, "HNSCC/tfrecords/256px_256um/C3L-02621-23.tfrecords")
CPTAC_RESOLUTION = os.environ.get("CPTAC_RESOLUTION", "256px_256um")

	
class TFRecordDataset(sf.io.TFRecordDataset):
		def __init__(self, tfrecords_path, index_path, transform):
			self.index = np.load(index_path)['arr_0']
			super().__init__(tfrecords_path, self.index, transform=transform)

		def __len__(self):
			return len(self.index)


class PancanTileBag(Datablock):
	@dataclass
	class CONFIG(Datablock.CONFIG):
		source: str

	def __post_init__(self):
		root, tail = self.config.source.split('/tfrecords/')
		self.label = root.split('/')[-1] #cancer
		self.resolution, records = tail.split('/')
		self.name, _  = os.path.splitext(records)
		tilesfile = os.path.basename(self.config.source)
		indexfile = tilesfile.split('.')[0] + '.index.npz'
		self.FILES = {'index': indexfile, 'tiles': tilesfile}
		self._dirpath = os.path.dirname(self.config.source)
		self._tensor = None

	def dirpath(self, topic, *, ensure: bool = False): 
		return self._dirpath

	def path(self, topic, *, ensure_dirpath: bool = False):
		return os.path.join(self._dirpath, self.FILES[topic])

	def UNSAFE_clear(self):
		raise ValueError(f"Read-Only datablock: {self}")

	def __read__(self, topic):
		if topic == 'index':
			result = np.load(self.path(topic))['arr_0']
		elif topic == 'tiles':
			result = self.tensor
		else:
			raise ValueError(f"Unknown topic: {topic}")
		return result

	def __len__(self):
		return len(self.read('index'))

	@property
	def size(self):
		return len(self)
	
	@property
	def dataset(self): 
		parser = sf.io.get_tfrecord_parser(
				self.path('tiles'),
				('image_raw',),
				to_numpy=True,
				decode_images=True
		)
		def transform(*args, **kwargs):
			return parser(*args, **kwargs)[0]
		dataset = TFRecordDataset(self.path('tiles'), self.path('index'), transform=transform)
		return dataset
	
	@property
	def tensor(self):
		if self._tensor is None:
			tensors = list(self.dataset)
			self._tensor = torch.stack(tensors).permute(0, 3, 1, 2)
		return self._tensor


class PancanTileBags(Databatch):
	DATABLOCK = PancanTileBag
	@dataclass
	class CONFIG:
		source: str
		resolution: str

	def __post_init__(self):
		def _is_tfrecords_dir(fs, d, resolution=CPTAC_RESOLUTION):
			is_tf_records_dir = (
				fs.isdir(d) and
				'tfrecords' in [os.path.basename(f) for f in fs.ls(d)] and
				fs.isdir(os.path.join(d, 'tfrecords', resolution))
			)
			return is_tf_records_dir
		def _tfrecords_paths(fs, d, resolution):
			dd = os.path.join(d, 'tfrecords', resolution)
			ff = [f for f in fs.ls(dd) if f.endswith('.tfrecords')]
			return ff
		fs, _ = fsspec.core.url_to_fs(self.root)
		slidepaths = list(itertools.chain.from_iterable(
			[
				_tfrecords_paths(fs, d, resolution=self.config.resolution) 
				for d in fs.ls(self.config.source) 
				if _is_tfrecords_dir(fs, d, resolution=self.config.resolution)
			]
		))
		self.bags = [
			PancanTileBag(
				self.root,
				spec=dict(source=slidepath,),
				verbose=self.verbose,
				debug=self.debug,
			)
			for slidepath in slidepaths
		]
		return self

	def datablocks(self):
		return self.bags


class PancanTileSets(Datablock): #from PancanSlideBatch
	FILES = {"train_bag_indices": "train_bag_indices.pt", 
			 "train_bag_lens":    "train_bag_lens.pt",
			 "test_bag_indices":  "test_bag_indices.pt",
			 "test_bag_lens":     "test_bag_lens.pt",
	}
	@dataclass
	class CONFIG:
		tilebags: PancanTileBags
		train_fraction: float = 0.8
		seed: int = 42

	def __post_init__(self):
		self.allbags = self.config.tilebags.datablocks()
		return self

	def __build__(self):
		self.log.verbose(f"Building tile datasets out of {len(self.allbags)} slides using train fraction {self.config.train_fraction}")
		N = len(self.allbags)
		K = int(math.ceil(N*self.config.train_fraction))
		np.random.seed(self.config.seed) #TODO: localize in a generator
		perm = np.random.permutation(N)
		train_bag_indices = torch.tensor(perm[:K])
		test_bag_indices = torch.tensor(perm[K:])
		self.log.verbose(f"Computing train bag lens")
		if self.verbose:
			train_bag_itor = tqdm.tqdm(train_bag_indices)
		else:
			train_bag_itor = train_bag_indices
		train_bag_lens = torch.tensor([len(self.allbags[i].dataset) for i in train_bag_itor])
		self.log.verbose(f"Computing test bag lens")
		if self.verbose:
			test_bag_itor = tqdm.tqdm(test_bag_indices)
		else:
			test_bag_itor = test_bag_indices
		test_bag_lens = torch.tensor([len(self.allbags[i].dataset) for i in test_bag_itor])
		dbx.write_tensor(train_bag_indices, self.path('train_bag_indices', ensure_dirpath=True),)
		dbx.write_tensor(train_bag_lens, self.path('train_bag_lens', ensure_dirpath=True),)
		dbx.write_tensor(test_bag_indices, self.path('test_bag_indices', ensure_dirpath=True),)
		dbx.write_tensor(test_bag_lens, self.path('test_bag_lens', ensure_dirpath=True),)
		return self
	
	def __read__(self, topic):
		tensor = dbx.read_tensor(self.path(topic))
		return tensor

	def bags(self, split):
		bag_indices = self.read(f"{split}_bag_indices")
		bags = [self.allbags[i] for i in bag_indices]
		return bags   

	def bag_lens(self, split):
		bag_lens = self.read(f"{split}_bag_lens")
		return bag_lens  	
			
	def dataset(self, split, *, transform=None):
		dataset = LabeledBagDataset(
						  bags=self.bags(split), 
						  bag_lens=self.bag_lens(split), 
						  transform=transform, 
						  debug=self.debug, 
						  verbose=self.verbose,
						  log=self.log,
		)
		return dataset


class PancanTileSet(Datablock):
	@dataclass
	class CONFIG(Datablock.CONFIG):
		datasets: PancanTileSets
		split: str = "train"
		transform: Optional[torchvision.transforms.Compose] = None

	@property
	def dataset(self):
		return self.config.datasets.dataset(self.config.split)

	def valid(self):
		return self.config.datasets.valid()

	def __read__(self):
		return self.dataset


"""
class PancanSlideShard(Datablock):
	FILES = {'train': None, 'test': None}
	@dataclass
	class CONFIG:
		source: str
		train_fraction: float = 0.8
		randomize: bool = False
		seed: Optional[int] = 42

	def __post_init__(self):
		self.origin, self.config.resolution, self.slide = self._parse_source(self.config.source)

	def __build__(self):
		index = os.path.splitext(self.config.source)[0]+'.index.npz'
		dataset = self._get_dataset(tfrecords_path=self.config.source, index_path=index)
		N = len(dataset)
		n = int(floor(N*self.config.train_fraction))
		if self.config.randomize:
			if self.config.seed is not None:
				torch.manual_seed(self.config.seed)
			train_indices = torch.tensor(np.random.choice(np.arange(N), size=(n,), replace=False))
			test_indices = torch.tensor([i for i in range(N) if i not in train_indices])
		else:
			train_indices = torch.tensor(np.arange(n))
			test_indices = torch.tensor(np.arange(n, N))
		self.log.verbose(f"computed {n}: train_indices len, and {N-n}:  test_indices len")
		train_subset = torch.stack(list(torch.utils.data.Subset(dataset, train_indices)))
		test_subset = torch.stack(list(torch.utils.data.Subset(dataset, test_indices)))

		self.log.verbose(f"extracted {len(train_indices)} train_indices and {len(test_indices)} test_indices")

		train_pt_path = self.path('train')
		train_index_path = self.path('train', index=True)
		torch.save(train_subset, train_pt_path)
		np.savez(train_index_path, train_indices)
		logger.verbose(f"Wrote train dataset to {train_pt_path=} and {train_index_path=}")

		test_pt_path = self.path('test')
		test_index_path = self.path('test', index=True)
		torch.save(test_subset, test_pt_path)
		np.savez(test_index_path, test_indices)
		logger.verbose(f"Wrote test dataset to {test_pt_path=} and {test_index_path=}")
		return self

	def index(self, topic):
		index = np.load(self.path(topic, index=True))['arr_0']
		return index

	@functools.lru_cache(maxsize=None)
	def size(self, topic):
		return len(self.index(topic))
	
	def read(self, topic):
		path = self.path(topic)
		images = torch.load(path)
		return torch.permute(images, (0, 3, 1, 2)), self.origin, self.slide
	
	def path(self, topic, *, index: bool = False, ensure: bool = True):
		path_ = super().path(topic)
		fs, _ = fsspec.core.url_to_fs(path_)
		if ensure:
			fs.makedirs(path_, exist_ok=True)
		path = os.path.join(path_, self.slide + ('.index.npz' if index else '.pt'))
		return path
	
	@staticmethod
	def _parse_source(source):
		root, tail = source.split('/tfrecords/')
		cancer = root.split('/')[-1]
		resolution, records = tail.split('/')
		slide, _  = os.path.splitext(records)
		return cancer, resolution, slide

	def _get_dataset(self, tfrecords_path: str, index_path: str) -> torch.Tensor:
		parser = sf.io.get_tfrecord_parser(
				tfrecords_path,
				('image_raw',),
				to_numpy=True,
				decode_images=True
		)
		def transform(*args, **kwargs):
			return parser(*args, **kwargs)[0]
		tensor_list = list(PancanSlideShard.TFRecordDataset(tfrecords_path, index_path, transform=transform))
		tensor = torch.stack(tensor_list)
		self.log.debug(f"Generated tensor of shape {tensor.shape}")
		return tensor


class PancanSlideBatch(Databatch):
	DATABLOCK = PancanSlideShard
	FILES = {"train": "train.npy", "test": "test.npy"}
	
	@dataclass
	class CONFIG:
		source: str
		resolution: str
		train_fraction: float = 0.8
		max_shards: Optional[int] = None
		randomize: bool = False
		seed: int = 42

	@functools.lru_cache(maxsize=None)
	def datablocks(self):
		tfrecords_paths = self._all_tfrecords_paths()
		datablocks = [
			PancanSlideShard(
				spec=dict(source=tfrecords_path,
						 train_fraction=self.config.train_fraction,
						 randomize=self.config.randomize,
						 seed=self.config.seed
				),
				verbose=self.verbose,
				debug=self.debug,
			)
			for tfrecords_path in tfrecords_paths
		]
		return datablocks

	def __build__(self):
		dbks = self.datablocks()
		#
		self.log.debug(f"__build__: launching ray tasks to build blocks")
		ray_refs = [
			ray.remote(dbx.datablock_build).remote(self.DATABLOCK, tag=self.tag, **dbk.kwargs())
			for dbk in dbks
		]
		self.log.debug(f"__build__: retrieving block build results from {len(ray_refs)} object references")
		with tqdm.tqdm(total=len(ray_refs), desc="Processing Ray tasks: building shards:") as pbar:
			while ray_refs:
				ready_refs, ray_refs = ray.wait(ray_refs)
				for ref in ready_refs:
					pbar.update(1)
		# We do not precompute lens as it is faster to recompute them every time than to retrieve them from file, it seems
		self.leave_breadcrumbs()
		return self

	@functools.lru_cache(maxsize=None)
	def read(self, split):
		self.log.verbose(f"Calculating shard lens for {split} split")
		dbks = self.datablocks()
		if self.verbose:
			dbks = tqdm.tqdm(dbks)
		return [dbk.size(split) for dbk in dbks]

	def lens(self, split):
		return self.read(split)

	def _is_tfrecords_dir(self, fs, d, resolution=CPTAC_RESOLUTION):
		is_tf_records_dir = (
			fs.isdir(d) and
			'tfrecords' in [os.path.basename(f) for f in fs.ls(d)] and
			fs.isdir(os.path.join(d, 'tfrecords', resolution))
		)
		return is_tf_records_dir

	def _tfrecords_paths(self, fs, d, resolution):
		dd = os.path.join(d, 'tfrecords', resolution)
		ff = [f for f in fs.ls(dd) if f.endswith('.tfrecords')]
		return ff

	def _all_tfrecords_paths(self):
		fs, _ = fsspec.core.url_to_fs(self.root)
		tfrecords_paths = list(itertools.chain.from_iterable(
			[self._tfrecords_paths(fs, d, resolution=self.config.resolution) for d in fs.ls(self.config.source) if self._is_tfrecords_dir(fs, d, resolution=self.config.resolution)]
		))
		if self.config.max_shards is not None:
			tfrecords_paths = tfrecords_paths[:self.config.max_shards]
		return tfrecords_paths


PANCAN_SLIDE_BATCH_CPTAC = PancanSlideBatch(spec=dict(source=CPTAC_ROOT, resolution=CPTAC_RESOLUTION))


class PancanTileSamples(Dataset):
	def __init__(self, databatch, *, split, transform=None, verbose: bool = False, debug: bool = False, log = None):
		self.databatch = databatch
		self.split = split
		self.transform = transform
		self.target_transform = None
		self.verbose = verbose
		self.debug = debug
		
		if log is None:
			self.log = Logger(verbose=self.verbose, debug=self.debug)
		else:
			self.log = log

		self._shards = self.databatch.build().datablocks()
		self._n_shards = len(self._shards)
		self._shard_lens = self.databatch.lens(self.split)
		self._shard_bounds = np.cumsum(self._shard_lens)
		self.log.debug(f"{self._n_shards=}, {self._shard_bounds=}")
		self._shard_idx = None
		self._shard = None
		self._shard_cancer = None
		self._shard_slide = None

	@functools.lru_cache(maxsize=3)
	def shard(self, shard_idx):
		if shard_idx != self._shard_idx:
			self._shard_idx = shard_idx
			self._shard, self._shard_cancer, self._shard_slide = self._shards[shard_idx].read(self.split)
		return self._shard, self._shard_cancer, self._shard_slide

	def __len__(self):
		return self._shard_bounds[-1]

	def __getitem__(self, index):
		shard_idx = np.searchsorted(self._shard_bounds, index, side='right')
		shard_lo = self._shard_bounds[shard_idx-1] if shard_idx > 0 else 0
		shard_hi = self._shard_bounds[shard_idx]
		shard_len = self._shard_lens[shard_idx]
		idx = index - shard_lo
		self.log.debug(f"__getitem__: {index=}, {shard_idx=}, {shard_lo=}, {shard_len=}, {shard_hi=}, {idx=}")
		tensor, target, _ = self.shard(shard_idx)
		sample = tensor[idx]
		if self.transform is not None:
			sample = self.transform(sample)
		return sample, target
"""

	
