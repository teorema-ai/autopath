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

from autopath.tools.dataset import BagDataset


logger = Logger()

	
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
			result = self.tiles
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

	@property
	def tiles(self):
		return self.tensor


class PancanTileBatch(Databatch):
	FILE = "bag_lens.npz"
	DATABLOCK = PancanTileBag
	@dataclass
	class CONFIG:
		source: str
		resolution: str

	def __post_init__(self):
		def _is_tfrecords_dir(fs, d, resolution):
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
		self.bagpaths = list(itertools.chain.from_iterable(
			[
				_tfrecords_paths(fs, d, resolution=self.config.resolution) 
				for d in fs.ls(self.config.source) 
				if _is_tfrecords_dir(fs, d, resolution=self.config.resolution)
			]
		))
		return self

	def __len__(self):
		return len(self.bags)

	@functools.cached_property
	def bags(self):
		return self.datablocks()

	@functools.cached_property
	def bag_lens(self):
		bag_lens = self.read()
		return bag_lens 

	@functools.lru_cache(maxsize=1)
	def datablocks(self):
		return [
			PancanTileBag(
				self.root,
				spec=dict(source=bagpath,),
				verbose=self.verbose,
				debug=self.debug,
			)
			for bagpath in self.bagpaths
		]

	def __build__(self):
		self.log.verbose(f"Computing bag lens")
		if self.verbose:
			bagsitor = tqdm.tqdm(self.datablocks())
		else:
			bagsitor = self.datablocks()
		bag_lens = [len(bag) for bag in bagsitor]
		dbx.write_npz(self.path(), bag_lens=bag_lens)
		return self

	def __read__(self):
		bag_lens = dbx.read_npz(self.path(), 'bag_lens')[0]
		return bag_lens


class PancanTileSplit(Datablock):
	FILES = {"train_bag_indices": "train_bag_indices.pt", 
			 "train_bag_lens":    "train_bag_lens.pt",
			 "test_bag_indices":  "test_bag_indices.pt",
			 "test_bag_lens":     "test_bag_lens.pt",
	}
	@dataclass
	class CONFIG:
		tilebatch: PancanTileBatch
		train_fraction: float = 0.8
		seed: int = 42

	def __build__(self):
		self.log.verbose(f"Building tile datasets out of {len(self.config.tilebatch.bags)} slides using train fraction {self.config.train_fraction}")
		N = len(self.config.tilebatch.bags)
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
		train_bag_lens = torch.tensor([len(self.config.tilebatch.bags[i].dataset) for i in train_bag_itor])
		self.log.verbose(f"Computing test bag lens")
		if self.verbose:
			test_bag_itor = tqdm.tqdm(test_bag_indices)
		else:
			test_bag_itor = test_bag_indices
		test_bag_lens = torch.tensor([len(self.config.tilebatch.bags[i].dataset) for i in test_bag_itor])
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
		bags = [self.config.tilebatch.bags[i] for i in bag_indices]
		return bags   

	def bag_lens(self, split):
		bag_lens = self.read(f"{split}_bag_lens")
		return bag_lens  	
			

class PancanTileFold(PancanTileBatch):
	@dataclass
	class CONFIG:
		tilesplit: PancanTileSplit
		fold: str

	def __post_init__(self):
		return self

	def datablocks(self):
		return self.config.tilesplit.bags(self.config.fold)
			

class PancanTileSet(Datablock):
	@dataclass
	class CONFIG(Datablock.CONFIG):
		tilebatch: PancanTileBatch
		transform: Optional[torchvision.transforms.Compose] = None

	@property
	def dataset(self):
		dataset = BagDataset(
						  bags=self.config.tilebatch.bags, 
						  bag_lens=self.config.tilebatch.bag_lens, 
						  transform=self.config.transform, 
						  debug=self.debug, 
						  verbose=self.verbose,
						  log=self.log,
		)
		return dataset

	def valid(self):
		return True

	def __read__(self):
		return self.dataset

