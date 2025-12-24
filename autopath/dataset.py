from dataclasses import dataclass
import functools
from typing import Optional


import tqdm

import numpy as np
import torch
import torchvision

from dbx import Datablock

from autopath.databits import Clip


class ClipDataset(Datablock, torch.utils.data.Dataset):
    @dataclass
    class CONFIG:
        clip: Clip
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
