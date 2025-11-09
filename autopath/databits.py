from dataclasses import dataclass
import functools
import math
from typing import Optional

import tqdm

import numpy as np
import torch
import torchvision


import dbx
from dbx import Datablock


class DataShard(Datablock):
    TOPICFILES = {'index': '', 'tensor': '', 'labels': None}
    @dataclass
    class CONFIG(Datablock.CONFIG):
        ...	

    def __read__(self, topic):
        if topic == 'index':
            result = np.load(self.path(topic))['arr_0']
        elif topic == 'tensor':
            result = self.tiles
        elif topic == 'labels':
            result = self.labels
        else:
            raise ValueError(f"Unknown topic: {topic}")
        return result
    
    def __len__(self):
        return len(self.tensor)
    
    def size(self):
        return len(self)

    @functools.cached_property
    def tensor(self):
        raise NotImplementedError

    @functools.cached_property
    def labels(self):
        raise NotImplementedError
    

class DataBag(DataShard):
    def __init__(self, name):
        self.name = name
    
    @functools.cached_property
    def labels(self):
        return [self.name]*len(self)

    
class DataClip(Datablock):
    TOPICFILE = "shard_lens.npz"
    def __len__(self):
        return len(self.shards)
    
    def __build__(self):
        self.log.verbose(f"Computing shard lens")
        if self.verbose:
            shardsitor = tqdm.tqdm(self.shards)
        else:
            shardsitor = self.shards
        shard_lens = [len(shard) for shard in shardsitor]
        dbx.write_npz(self.path(), shard_lens=shard_lens)
        return self
    
    def __read__(self):
        shard_lens = dbx.read_npz(self.path(), 'shard_lens')[0]
        return shard_lens

    @functools.cached_property
    def shards(self):
        raise NotImplementedError

    @functools.cached_property
    def shards_lens(self):
        return self.read()
    

class DataSplit(Datablock):
    TOPICFILES = {"train_shard_indices": "train_shard_indices.pt", 
                  "train_shard_lens":    "train_shard_lens.pt",
                  "test_shard_indices":  "test_shard_indices.pt",
                  "test_shard_lens":     "test_shard_lens.pt",
    }
    @dataclass
    class CONFIG:
        clip: DataClip
        train_fraction: float = 0.8
        seed: int = 42

    def __build__(self):
        self.log.info(f"Building splits out of {len(self.cfg.clip.shards)} shards using train fraction {self.cfg.train_fraction}")
        N = len(self.cfg.clip.shards)
        K = int(math.ceil(N*self.cfg.train_fraction))
        np.random.seed(self.cfg.seed) #TODO: localize in a generator
        perm = np.random.permutation(N)
        train_shard_indices = torch.tensor(perm[:K])
        test_shard_indices = torch.tensor(perm[K:])
        self.log.verbose(f"Computing train shard lens")
        if self.verbose:
            train_shard_itor = tqdm.tqdm(train_shard_indices)
        else:
            train_shard_itor = train_shard_indices
        train_shard_lens = torch.tensor([len(self.cfg.clip.shards[i].dataset) for i in train_shard_itor])
        self.log.verbose(f"Computing test shard lens")
        if self.verbose:
            test_shard_itor = tqdm.tqdm(test_shard_indices)
        else:
            test_shard_itor = test_shard_indices
        test_shard_lens = torch.tensor([len(self.cfg.clip.shards[i].dataset) for i in test_shard_itor])
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
        shards = [self.cfg.clip.shards[i] for i in shard_indices]
        return shards 
    
    def shard_lens(self, split):
        shard_lens = self.read(f"{split}_shard_lens")
        return shard_lens  	


class DataFold(DataClip):
    @dataclass
    class CONFIG:
        split: DataSplit
        fold: str

    def __post_init__(self):
        return self
    
    def valid(self):
        return self.cfg.split.valid()

    @functools.cached_property
    def shards(self):
        return self.cfg.split.shards(self.cfg.fold)

    @functools.cached_property
    def shard_lens(self):
        return self.cfg.split.shard_lens(self.cfg.fold)
    
            
class DataClipDataset(Datablock, torch.utils.data.Dataset):
    @dataclass
    class CONFIG:
        clip: DataClip
        shard_lens: list[int] = None
        transform: Optional[torchvision.transforms.Compose] = None

    def __post_init__(self):
        self._n_shards = len(self.cfg.clip.shards)
        self.log.debug(f"Building dataset out of {self._n_shards} shards")
        if self._shard_lens is None:
            if self.verbose:
                shardsitor = tqdm.tqdm(self.cfg.clip.shards)
                self.log.verbose(f"Computing _shard_lens")
            else:
                shardsitor = self.cfg.clip.shards
            self._shard_lens = [len(shard) for shard in shardsitor]
        self.log.debug(f"Computing _shard_bounds")
        self._shard_bounds = np.cumsum(self._shard_lens)
        self.log.debug(f"{self._n_shards=}, {self._shard_bounds=}")
        self._shard_idx = None
        self._shard = None
        self._shard_label = None
        self._shard_slide = None

    @functools.lru_cache(maxsize=3)
    def shard(self, shard_idx):
        if shard_idx != self._shard_idx:
            self._shard_idx = shard_idx
            self._shard = self.cfg.clip.shards[self._shard_idx]
        return self._shard

    def __len__(self):
        return self._shard_bounds[-1]

    def __getitem__(self, index):
        shard_idx = np.searchsorted(self._shard_bounds, index, side='right')
        shard_lo = self._shard_bounds[shard_idx-1] if shard_idx > 0 else 0
        shard_hi = self._shard_bounds[shard_idx]
        shard_len = self._shard_lens[shard_idx]
        idx = index - shard_lo
        self.log.detailed(f"{index=}, {shard_idx=}, {shard_lo=}, {shard_len=}, {shard_hi=}, {idx=}")
        tensor = self.shard(shard_idx).tensor
        sample = tensor[idx]
        if self.transform is not None:
            sample = self.transform(sample)
        labels = self.shard(shard_idx).labels
        label = labels[idx]
        return sample, label
