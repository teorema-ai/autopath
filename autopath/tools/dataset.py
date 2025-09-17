import functools

import tqdm

import numpy as np
import torch

from dbx import Logger


class BagDataset(torch.utils.data.Dataset):
    def __init__(self, bags, bag_lens=None, transform=None, *, verbose: bool = False, debug: bool = False, log = None):
        self.bags = bags
        self._bag_lens = bag_lens
        self.transform = transform
        self.target_transform = None
        self.verbose = verbose
        self.debug = debug
        
        if log is None:
            self.log = Logger(verbose=self.verbose, debug=self.debug)
        else:
            self.log = log

        self._n_bags = len(self.bags)
        self.log.debug(f"Building dataset out of {self._n_bags} bags")
        if self._bag_lens is None:
            if self.verbose:
                bagsitor = tqdm.tqdm(self.bags)
                self.log.verbose(f"Computing _bag_lens")
            else:
                bagsitor = self.bags
            self._bag_lens = [len(bag) for bag in bagsitor]
        self.log.debug(f"Computing _bag_bounds")
        self._bag_bounds = np.cumsum(self._bag_lens)
        self.log.debug(f"{self._n_bags=}, {self._bag_bounds=}")
        self._bag_idx = None
        self._bag = None
        self._bag_label = None
        self._bag_slide = None

    @functools.lru_cache(maxsize=3)
    def bag(self, bag_idx):
        if bag_idx != self._bag_idx:
            self._bag_idx = bag_idx
            self._bag = self.bags[self._bag_idx]
        return self._bag

    def __len__(self):
        return self._bag_bounds[-1]

    def __getitem__(self, index):
        bag_idx = np.searchsorted(self._bag_bounds, index, side='right')
        bag_lo = self._bag_bounds[bag_idx-1] if bag_idx > 0 else 0
        bag_hi = self._bag_bounds[bag_idx]
        bag_len = self._bag_lens[bag_idx]
        idx = index - bag_lo
        self.log.debug(f"__getitem__: {index=}, {bag_idx=}, {bag_lo=}, {bag_len=}, {bag_hi=}, {idx=}")
        tensor = self.bag(bag_idx).tensor
        sample = tensor[idx]
        if self.transform is not None:
            sample = self.transform(sample)
        label = self.bag(bag_idx).label
        return sample, label


class ShardDataset(torch.utils.data.Dataset):
    def __init__(self, shards, shard_lens=None, transform=None, *, verbose: bool = False, debug: bool = False, log = None):
        self.shards = shards
        self._shard_lens = shard_lens
        self.transform = transform
        self.target_transform = None
        self.verbose = verbose
        self.debug = debug
        
        if log is None:
            self.log = Logger(verbose=self.verbose, debug=self.debug)
        else:
            self.log = log

        self._n_shards = len(self.shards)
        self.log.debug(f"Building dataset out of {self._n_shards} shards")
        if self._shard_lens is None:
            if self.verbose:
                shardsitor = tqdm.tqdm(self.shards)
                self.log.verbose(f"Computing _shard_lens")
            else:
                shardsitor = self.shards
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
            self._shard = self.shards[self._shard_idx]
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
