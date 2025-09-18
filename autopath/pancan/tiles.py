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

from autopath.tools.dataset import ShardDataset


logger = Logger()

	
class TFRecordDataset(sf.io.TFRecordDataset):
		def __init__(self, tfrecords_path, index_path, transform):
			self.index = np.load(index_path)['arr_0']
			super().__init__(tfrecords_path, self.index, transform=transform)

		def __len__(self):
			return len(self.index)


class PancanTileShard:
	def __len__(self):
		return len(self.tiles)

	@functools.cached_property
	def tiles(self):
		raise NotImplementedError

	@functools.cached_property
	def labels(self):
		raise NotImplementedError


class PancanTileBag(Datablock, PancanTileShard):
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
		self.FILES = {'index': indexfile, 'tiles': tilesfile, 'labels': None}
		self._dirpath = os.path.dirname(self.config.source)
		self._tensor = None

	def dirpath(self, topic, *, ensure: bool = False): 
		return self._dirpath

	def path(self, topic, *, ensure_dirpath: bool = False):
		return os.path.join(self._dirpath, self.FILES[topic]) if self.FILES[topic] is not None else None

	def UNSAFE_clear(self):
		raise ValueError(f"Read-Only datablock: {self}")

	def __read__(self, topic):
		if topic == 'index':
			result = np.load(self.path(topic))['arr_0']
		elif topic == 'tiles':
			result = self.tiles
		elif topic == 'labels':
			result = np.array([self.label]*len(self))
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
	
	@functools.cached_property
	def tensor(self):
		if self._tensor is None:
			tensors = list(self.dataset)
			self._tensor = torch.stack(tensors).permute(0, 3, 1, 2)
		return self._tensor

	@property
	def tiles(self):
		return self.tensor

	@functools.cached_property
	def labels(self):
		return self.read('labels')


class PancanTileShards:
	def __len__(self):
		return len(self.shards)

	@functools.cached_property
	def shards(self):
		raise NotImplementedError

	@functools.cached_property
	def shards_lens(self):
		raise NotImplementedError


class PancanTileBags(Databatch, PancanTileShards):
	DATABLOCK = PancanTileBag
	FILE = "bag_lens.npz"
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

	@property
	def bags(self):
		return self.datablocks()

	@property
	def bag_lens(self):
		return self.read()

	def __len__(self):
		return len(self.bags)
	
	def __build__(self):
		self.log.verbose(f"Computing bag lens")
		if self.verbose:
			bagsitor = tqdm.tqdm(self.datablocks())
		else:
			bagsitor = self.datablocks()
		bag_lens = [len(shard) for shard in bagsitor]
		dbx.write_npz(self.path(), bag_lens=bag_lens)
		return self

	def __read__(self):
		bag_lens = dbx.read_npz(self.path(), 'bag_lens')[0]
		return bag_lens

	@functools.cached_property
	def shards(self):
		return self.bags

	@functools.cached_property
	def shard_lens(self):
		return self.bag_lens


class PancanTileSplit(Datablock):
	FILES = {"train_shard_indices": "train_shard_indices.pt", 
			 "train_shard_lens":    "train_shard_lens.pt",
			 "test_shard_indices":  "test_shard_indices.pt",
			 "test_shard_lens":     "test_shard_lens.pt",
	}
	@dataclass
	class CONFIG:
		tileshards: PancanTileShards
		train_fraction: float = 0.8
		seed: int = 42

	def __build__(self):
		self.log.info(f"Building tile splits out of {len(self.config.tileshards.shards)} shards using train fraction {self.config.train_fraction}")
		N = len(self.config.tileshards.shards)
		K = int(math.ceil(N*self.config.train_fraction))
		np.random.seed(self.config.seed) #TODO: localize in a generator
		perm = np.random.permutation(N)
		train_shard_indices = torch.tensor(perm[:K])
		test_shard_indices = torch.tensor(perm[K:])
		self.log.verbose(f"Computing train shard lens")
		if self.verbose:
			train_shard_itor = tqdm.tqdm(train_shard_indices)
		else:
			train_shard_itor = train_shard_indices
		train_shard_lens = torch.tensor([len(self.config.tileshards.shards[i].dataset) for i in train_shard_itor])
		self.log.verbose(f"Computing test shard lens")
		if self.verbose:
			test_shard_itor = tqdm.tqdm(test_shard_indices)
		else:
			test_shard_itor = test_shard_indices
		test_shard_lens = torch.tensor([len(self.config.tileshards.shards[i].dataset) for i in test_shard_itor])
		dbx.write_tensor(train_shard_indices, self.path('train_shard_indices', ensure_dirpath=True),)
		dbx.write_tensor(train_shard_lens, self.path('train_shard_lens', ensure_dirpath=True),)
		dbx.write_tensor(test_shard_indices, self.path('test_shard_indices', ensure_dirpath=True),)
		dbx.write_tensor(test_shard_lens, self.path('test_shard_lens', ensure_dirpath=True),)
		return self
	
	def __read__(self, topic):
		tensor = dbx.read_tensor(self.path(topic))
		return tensor

	def shards(self, split):
		shard_indices = self.read(f"{split}_shard_indices")
		shards = [self.config.tileshards.shards[i] for i in shard_indices]
		return shards   

	def shard_lens(self, split):
		shard_lens = self.read(f"{split}_shard_lens")
		return shard_lens  	
			

class PancanTileFold(Databatch, PancanTileShards):
	@dataclass
	class CONFIG:
		tilesplit: PancanTileSplit
		fold: str

	def __post_init__(self):
		return self

	def datablocks(self):
		return self.config.tilesplit.shards(self.config.fold)

	@functools.cached_property
	def shards(self):
		return self.datablocks()

	@functools.cached_property
	def shard_lens(self):
		return self.config.tilesplit.shard_lens(self.config.fold)
			

class PancanTileSet(Datablock):
	@dataclass
	class CONFIG(Datablock.CONFIG):
		tileshards: PancanTileShards
		transform: Optional[torchvision.transforms.Compose] = None

	@property
	def dataset(self):
		dataset = ShardDataset(
						  shards=self.config.tileshards.shards, 
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


class PancanTileBatch(Datablock):
	VERSION=1
	@dataclass
	class CONFIG(Datablock.CONFIG):
		tileset: PancanTileSet
		start: int
		end: int

	def __post_init__(self):
		self.FILES = {
			"tiles": f"{self.config.start}-{self.config.end}-tiles.pt",
			"labels": f"{self.config.start}-{self.config.end}-labels.pt",
		}
		return self

	def store(self, tiles, labels):
		self.__pre_build__()
		dbx.write_tensor(tiles, self.path('tiles', ensure_dirpath=True))
		dbx.write_npz(self.path('labels', ensure_dirpath=True), labels=labels)
		self._write_journal_entry(event="store")
		self.__post_build__()
		return self

	def __read__(self, topic):
		if topic == 'tiles':
			result = dbx.read_tensor(self.path(topic))
		elif topic == 'labels':
			result = dbx.read_npz(self.path(topic), 'labels')
		else:
			raise ValueError(f"Unknown topic: {topic}")
		return result

	def __len__(self):
		return len(self.config.end - self.config.start)

	@functools.cached_property
	def tiles(self):
		return self.read('tiles')

	@functools.cached_property
	def labels(self):
		return self.read('labels')

	@property
	def start(self):
		return self.config.start

	@property
	def end(self):
		return self.config.end

	
class PancanTileBatches(Datablock):
	VERSION = 1
	FILES = {'batch_lens': 'batch_lens.pt'}
	@dataclass
	class CONFIG(Datablock.CONFIG):
		tileset: PancanTileSet
		batch_size: int = 512

	def __init__(self, *args, dataloader_num_workers: int = 0, **kwargs):
		super().__init__(*args, dataloader_num_workers=dataloader_num_workers, **kwargs)

	@functools.cached_property
	def batches(self):
		N = len(self.config.tileset.dataset)
		tilebatches = [PancanTileBatch(
								  root=self.root if not self._autoroot else None,
								  spec=dict(tileset=self.spec.get('tileset'), 
											start=batch_start, 
											end=min(batch_start+self.config.batch_size, N))
						)
						for batch_start in range(0, N, self.config.batch_size)
		]
		return tiledecks

	def __len__(self):
		return len(self.batches)

	def __build__(self):
		"""Single-process, but, potentially, a multithreaded build."""
		dataloader = torch.utils.data.DataLoader(
			self.config.tileset.dataset, 
			batch_size=self.config.batch_size,
			shuffle=True,
			num_workers=self.dataloader_num_workers,
			collate_fn=self.collate,
		)
		batch_lens = []
		self.log.info(f"Building {len(self.batches)} tile batches of size {self.config.batch_size} from dataset of size {len(self.config.tileset.dataset)}")
		if self.info:
			batch_itor = tqdm.tqdm(dataloader)
		else:
			batch_itor = dataloader
		lo = hi = 0
		for i, (sample, labels) in enumerate(batch_itor):
			lo = hi
			hi = lo + len(sample)
			batch_lens.append(hi-lo)
			assert len(sample) == len(labels), f"len(sample) != len(labels): {len(sample)} != {len(labels)}"
			assert lo == self.batchs[i].start, f"lo != self.batchs[i].start: {lo} != {self.batchs[i].start}"
			assert hi == self.batchs[i].end, f"hi != self.batchs[i].end: {hi} != {self.batchs[i].end}"
			self.batchs[i].store(sample, labels)
		dbx.write_tensor(torch.tensor(batch_lens), self.path('batch_lens', ensure_dirpath=True))
		return self

	def __read__(self, topic):
		reuturn dbx.read_tensor(self.path(topic))

	@functools.cached_property
	def batch_lens(self):
		return self.read('batch_lens')

	@property
	def shards(self):
		return self.batches

	@property
	def shard_lens(self):
		return self.batch_lens

	@staticmethod
	def collate(samples): 
		tensors_, labels_ = zip(*samples)
		tensors = torch.stack(tensors_)
		labels = np.array(labels_)
		return tensors, labels
	
