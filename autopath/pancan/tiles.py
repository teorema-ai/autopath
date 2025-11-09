from dataclasses import dataclass
import functools
import itertools
import os
from typing import Optional

import tqdm

import fsspec
import numpy as np

import torch
import torchvision


import slideflow as sf

import dbx
from dbx import Logger, Datablock

from autopath.databits import DataBag, DataClipDataset
from autopath.tiles import TileShard, TileBag, TileClip, TileSplit, TileFold


logger = Logger()

	
class TFRecordDataset(sf.io.TFRecordDataset):
		def __init__(self, tfrecords_path, index_path, transform):
			self.index = np.load(index_path)['arr_0']
			super().__init__(tfrecords_path, self.index, transform=transform)

		def __len__(self):
			return len(self.index)


class PancanTileShard(TileShard):
	...


class PancanTileBag(PancanTileShard, TileBag):
	@dataclass
	class CONFIG(Datablock.CONFIG):
		source: str

	def __post_init__(self):
		root, tail = self.config.source.split('/tfrecords/')
		self.label = root.split('/')[-1] #cancer
		self.resolution, records = tail.split('/')
		name, _  = os.path.splitext(records)
		DataBag.__init__(self, name)
		tilesfile = os.path.basename(self.config.source)
		indexfile = tilesfile.split('.')[0] + '.index.npz'
		self.TOPICFILES = {'index': indexfile, 'tiles': tilesfile, 'labels': None}
		self._dirpath = os.path.dirname(self.config.source)
		self._tensor = None

	def dirpath(self, topic, *, ensure: bool = False): 
		return self._dirpath

	def path(self, topic, *, ensure_dirpath: bool = False):
		return os.path.join(self._dirpath, self.TOPICFILES[topic]) if self.TOPICFILES[topic] is not None else None

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


class PancanTileClip(TileClip):
	TOPICFILE = "bag_lens.npz"
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

	@functools.cached_property
	def bags(self):
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
	def bag_lens(self):
		return self.read()

	def __len__(self):
		return len(self.bags)
	
	def __build__(self):
		self.log.verbose(f"Computing bag lens")
		if self.verbose:
			bagsitor = tqdm.tqdm(self.bags)
		else:
			bagsitor = self.bags
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


class PancanTileSplit(TileSplit):
	@dataclass
	class CONFIG:
		tileclip: PancanTileClip
		train_fraction: float = 0.8
		seed: int = 42

	def bags(self, split):
		return self.shards(split) 
	
	def bag_lens(self, split):
		return self.shard_lens(split)


class PancanTileFold(TileFold):
	@dataclass
	class CONFIG:
		tilesplit: PancanTileSplit
		fold: str


def pancan_tileset(tileclip: PancanTileClip,
				   transform: Optional[torchvision.transforms.Compose] = None,
				   *,
				   debug: bool = False,
               	   verbose: bool = False,
                   log = None,
):
		tileclip = dbx.eval_term(tileclip)
		transform = dbx.eval_term(transform)
		return DataClipDataset(spec=dict(clip=tileclip, transform=transform,),
								  debug=debug, 
								  verbose=verbose,
								  log=log,
		)


'''
#DEPRECATE?
class PancanTileBatch(Datablock):
	VERSION=1
	@dataclass
	class CONFIG(Datablock.CONFIG):
		tileset: PancanTileSet
		start: int
		end: int

	def __post_init__(self):
		self.TOPICFILES = {
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


#DEPRECATE?
class PancanTileBatches(Datablock):
	VERSION = 1
	TOPICFILES = {'batch_lens': 'batch_lens.pt'}
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
		return tilebatches

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
		return dbx.read_tensor(self.path(topic))

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
'''	
