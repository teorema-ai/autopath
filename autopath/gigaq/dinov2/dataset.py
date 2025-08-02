from dataclasses import dataclass
import functools
import itertools
from math import floor
import os
from typing import Optional

import fsspec
import numpy as np

import torch
import torchvision
from torch.utils.data import IterableDataset, ChainDataset

import slideflow as sf

from dbx import Logger, Datablock, Databatch

from dinov2.data import collate_data_and_cast, MaskingGenerator
from .augmentations import DataAugmentationDINO


logger = Logger()


class TFRecordDataset(sf.io.TFRecordDataset):
        def __init__(self, tfrecords_path, index_path, transform):
            self.index = np.load(index_path)['arr_0']
            super().__init__(tfrecords_path, self.index, transform=transform)

        def __len__(self):
            return len(self.index)


class GigapathShard(Datablock):
    TOPICS = {'train': None, 'test': None}
    ROOT = "/mnt/labshare/PROJECTS/GIGAPATH_UQ/dbx"

    @dataclass
    class CONFIG:
        source: str = "/mnt/labshare/SLIDES/CPTAC_downloads/HNSCC/tfrecords/256px_256um/C3L-02621-23.tfrecords"
        train_fraction: float = 0.8
        randomize: bool = False
        seed: Optional[int] = 42

    def __post_init__(self):
        self.cancer, self.resolution, self.slide = self._parse_source(self.source)

    def __build__(self):
        index = os.path.splitext(self.source)[0]+'.index.npz'
        dataset = self._get_dataset(tfrecords_path=self.source, index_path=index)
        N = len(dataset)
        n = int(floor(N*self.train_fraction))
        if self.randomize:
            if self.seed is not None:
                torch.manual_seed(self.seed)
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
        return torch.permute(images, (0, 3, 1, 2))
    
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
        tensor_list = list(GigapathShard.TFRecordDataset(tfrecords_path, index_path, transform=transform))
        tensor = torch.stack(tensor_list)
        self.log.debug(f"Generated tensor of shape {tensor.shape}")
        return tensor


class GigapathDatabatch(Databatch):
    TOPICS = {'train': 'train', 'test': 'test'}
    ROOT = "/mnt/labshare/PROJECTS/GIGAPATH_UQ/dbx"

    @dataclass
    class CONFIG:
        source: str = "/mnt/labshare/SLIDES/CPTAC_downloads"
        resolution: str ="256px_256um" # 256px_256um
        train_fraction: float = 0.8
        max_shards: Optional[int] = None
        randomize: bool = False
        seed: int = 42

    def datablocks(self):
        tfrecords_paths = self._all_tfrecords_paths()
        datablocks = [
            GigapathShard(
                cfg=dict(source=tfrecords_path,
                         train_fraction=self.train_fraction,
                         randomize=self.randomize,
                         seed=self.seed
                ),
                verbose=self.verbose,
                debug=self.debug,
            )
            for tfrecords_path in tfrecords_paths
        ]
        return datablocks

    def __build__(self, *args, **kwargs):
        super().__build__(*args, **kwargs)
        self.leave_breadcrumbs()
        return self

    def read(self, topic):
        class TensorSliceIterableDataset(IterableDataset):
            def __init__(self, tensor: torch.Tensor):
                super(TensorSliceIterableDataset).__init__()
                self.tensor = tensor

            def __iter__(self):
                for i in range(self.tensor.shape[0]):
                    yield self.tensor[i], ()
            
        for i, dbk in enumerate(self.datablocks()):
            self.log.debug(f"using shard {i}")
            yield TensorSliceIterableDataset(dbk.read(topic))

    def _is_tfrecords_dir(self, fs, d, resolution="256px_256um"):
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
            [self._tfrecords_paths(fs, d, resolution=self.resolution) for d in fs.ls(self.source) if self._is_tfrecords_dir(fs, d, resolution=self.resolution)]
        ))
        if self.max_shards is not None:
            tfrecords_paths = tfrecords_paths[:self.max_shards]
        return tfrecords_paths


class GigapathDataset(IterableDataset):
    def __init__(self, databatch, *, split, transform=None):
        self.databatch = databatch
        self.split = split
        self.transform = transform
        self.target_transform = None

    def __iter__(self):
        shards = self.databatch.build().read(self.split.lower())
        chain_dataset = ChainDataset(shards)
        for sample, target in chain_dataset:
            if self.transform is not None:
                sample = self.transform(sample)
            yield sample, target

    @functools.lru_cache(maxsize=1)
    def __len__(self):
        return sum(dbk.size(self.split) for dbk in self.databatch.datablocks())

    
def make_dataset(
    *,
    path: str = "/mnt/labshare/SLIDES/CPTAC_downloads",
    resolution: str = "256px_256um", # 256px_256um,
    global_crops_scale=[0.32, 1.0],
    local_crops_scale=[0.05, 0.32],
    local_crops_number=8,
    global_crops_size=224,
    local_crops_size=96,
    split: str = "train",
    train_fraction: float = 0.8,
    randomize: bool = False,
    seed: int = 42,  
    verbose: bool = True,
    debug: bool = False,
):     
    databatch = GigapathDatabatch(
        cfg=dict(
            source=path,
            resolution=resolution,
            train_fraction=train_fraction,
            randomize=randomize,
            seed=seed,
        ),
        verbose=verbose,
        debug=debug,
    )
    transform = DataAugmentationDINO(
        global_crops_scale,
        local_crops_scale,
        local_crops_number,
        global_crops_size=global_crops_size,
        local_crops_size=local_crops_size,
    )
    dataset = GigapathDataset(databatch, split=split, transform=transform)
    return dataset