import copy
from dataclasses import dataclass
import functools
import gc
import math
from typing import Callable

import tqdm

import numpy as np


import torch

import dbx
from dbx import Datablock

from autopath.databits import Bag, Clip, ClipDataset
from .tiles import TileBag


def tensors_to_device(tensors, device, *, detach: bool = False):
    _tensors = {k: v.to(device) for k, v in tensors.items()}
    if detach:
        _tensors = {k: v.detach() for k, v in _tensors.items()} 
    return _tensors


def cat_tensor_dicts(tensor_dicts):
    tensors = {k: [] for k in tensor_dicts[0].keys()}
    for tensor_dict in tensor_dicts:
        for k, v in tensor_dict.items():
            tensors[k].append(v)
    _tensors = {k: torch.cat(v) for k, v in tensors.items()}
    return _tensors

class FeatureBag(Bag):
    VERSION = 1
    @dataclass
    class CONFIG(Datablock.CONFIG):
        tilebag: TileBag
        extractor: Callable

    def __init__(self, *args, gpu_batch_size: int = 16, **kwargs):
        Datablock.__init__(self, *args, gpu_batch_size=gpu_batch_size, **kwargs)
        Bag.__init__(self.cfg.tilebag.name)


    def __post_init__(self):
        self.TOPICFILES = {
            'features': 'features.npy',
        }
        if self.has_sideband:
            for layer in self.cfg.extractor.sideband_layers:
                self.TOPICFILES[f'sideband_{layer}' ] = \
                    f'sideband_{layer}.npy'
        self._len = None
        return self

    def __len__(self):
        if self._len is None:
            self._len = len(self.labels)
        return self._len
    
    @property
    def has_sideband(self):
        return hasattr(self.cfg.extractor, 'sideband_layers')

    def __build__(self, extractor):
        tilebag = self.cfg.tilebag
        feature_list = []
        sideband_list = []
        n_tiles = len(tilebag.tiles)
        for k in range(math.ceil(len(tilebag.tiles)/self.gpu_batch_size)):
            m = k*self.gpu_batch_size
            n = min((k+1)*self.gpu_batch_size, n_tiles)
            batch = tilebag.tiles[m:n].to(self.device)
            self.log.verbose(f"Evaluating batch {k}: {m}:{n} out of {n_tiles} on device: {self.device}")
            feature = extractor(batch)
            feature_list.append(feature.to('cpu'))
            self.log.debug(f"Evaluated batch to a feature of shape {feature.shape} on device: {self.device}")
            del batch
            del feature
            gc.collect()
            torch.cuda.empty_cache()
            if self.has_sideband:
                _sideband = tensors_to_device(extractor.sideband, 'cpu', detach=True)
                extractor.clear_sideband()
                assert set(_sideband.keys()) == set(extractor.sideband_layers), f"_sideband keys must match sideband_layers: {_sideband.keys()} != {extractor.sideband_layers}"
                _sideband_shapes = {k: v.shape for k, v in _sideband.items()}
                self.log.debug(f"Captured _sideband with shapes {_sideband_shapes} on device: {self.device}")
                sideband_list.append(_sideband)
                del _sideband
                gc.collect
                torch.cuda.empty_cache()
            gc.collect()
            torch.cuda.empty_cache()
        self.log.debug(f"Concatenating {len(feature_list)} device batch features on device: {self.device}")
        features = torch.cat(feature_list)
        assert len(features) == n_tiles, f"Number of features does not match the number of tiles: {len(features)} != {n_tiles}"
        del feature_list
        gc.collect()
        torch.cuda.empty_cache()
        self.log.debug(f"Storing features of shape {features.shape} on device: {self.device}")
        dbx.write_tensor(features, self.path('features', ensure_dirpath=True))
        del features
        gc.collect()
        torch.cuda.empty_cache()
        if hasattr(extractor, 'sideband'):
            self.log.debug(f"Concatenating {len(sideband_list)} device batch sidebands on device: {self.device}")
            sideband = cat_tensor_dicts(sideband_list)
            del sideband_list
            gc.collect()
            torch.cuda.empty_cache()
            if self.debug:
                sideband_shapes = {k: v.shape for k, v in sideband.items()}
                self.log.debug(f"Storing sidebands of shapes {sideband_shapes} on device: {self.device}")
            for lyr, sbd in sideband.items():
                dbx.write_tensor(sbd, self.path(f'sideband_{lyr}', ensure_dirpath=True))
            del sbd
            del sideband
            gc.collect()
            torch.cuda.empty_cache()
        self._len = n_tiles
        return self

    def read(self, topic):
        return dbx.read_tensor(self.path(topic))

    @functools.cached_property
    def features(self):
        return self.read('features')
    
    def sideband(self, layer):
        return self.read(f'sideband_{layer}')
    
    def layer(self, layer):
        return self.sideband(layer) if layer is not None else self.features
        
    @functools.cached_property
    def tensor(self):
        return self.features
    
    @functools.cached_property
    def labels(self):
        return list(zip(self.cfg.tilebag.labels, self.cfg.tilebag.tiles))


class FeatureBagClip(Clip):
    VERSION = 1
    TOPICFILES = {"bag_lens": "bag_lens.npy"}
    @dataclass
    class CONFIG:
        tilebagclip: Clip
        extractor: Callable

    def __init__(self, *args, devices: list[str] = ["cuda"], gpu_batch_size: int = 16, skip_unreadable: bool = True, **kwargs):
        super().__init__(*args, devices=devices, gpu_batch_size=gpu_batch_size, skip_unreadable=skip_unreadable, **kwargs)
        self.log.debug(f"devices={self.devices}, gpu_batch_size={self.gpu_batch_size}, skip_unreadable={self.skip_unreadable}")

    def __build__(self):
        bags = self.bags
        self.log.debug(f"Formed {len(bags)} FeatureBags.  Looking for missing bags.")
        missing_bags = [bag for bag in bags if not bag.valid()]
        self.log.debug(f"Found {len(missing_bags)} missing bags")
        self.log.debug(f"Building all missing features bags using devices {self.devices} and gpu_batch_size {self.gpu_batch_size}")
        built_bags = dbx.TorchMultithreadingDatablocksBuilder(devices=self.devices, log=self.log).build_blocks(missing_bags, self.cfg.extractor)
        self.log.verbose(f"Built all missing features shards: {len(built_bags)}")
        self.log.verbose(f"Building bag_lens: BEGIN")
        if self.verbose:
            bags = tqdm.tqdm(bags)
        bag_lens = torch.tensor([len(bag) for bag in bags])
        self.log.verbose(f"Building bag_lens: END")
        dbx.write_tensor(bag_lens, self.path("bag_lens", ensure_dirpath=True))
        return self
    
    def __read__(self, topic):
        if topic == "bag_lens":
            result = dbx.read_tensor(self.path("bag_lens"))
        else:
            raise ValueError(f"Unknown {topic=}")
        return result
    
    def features(self):
        self.log.debug(f"Reading features from {len(self.n_bags)} feature bags")
        feature_list = []
        for featurebag in self.bags:
            try:
                feature_list.append(featurebag.features)
            except Exception as e:
                if self.skip_unreadable:
                    continue
                else:
                    raise(e)
        features = torch.stack(feature_list)
        self.log.debug(f"Combined features: shape: {features.shape}")
        return features
    
    def tiles(self):
        self.log.debug(f"Reading tiles from {len(self.n_bags)} feature bags")
        tile_list = []
        for featurebag in self.bags:
            try:
                _tilebag_tiles = featurebag.cfg.tilebag.tiles
                tilebag_tiles = (
                    self.cfg.extractor.transform(_tilebag_tiles) 
                    if self.cfg.extractor.transform is not None 
                    else _tilebag_tiles
                )
                del _tilebag_tiles
                tile_list.append(tilebag_tiles)
            except Exception as e:
                if self.skip_unreadable:
                    continue
                else:
                    raise(e)
        tiles = torch.stack(tile_list)
        self.log.debug(f"Combined tiles: shape: {tiles.shape}")
        return tiles
    
    def sideband(self, layer):
        sideband_list = []
        for featurebag in self.bags:
            try:
                sideband_list.append(featurebag.sideband(layer))
            except Exception as e:
                if self.skip_unreadable:
                    continue
                else:
                    raise(e)
        sideband = torch.cat(sideband_list)
        return sideband
    
    def layer(self, layer=None):
        return self.sideband(layer) if layer is not None else self.features()
    
    @functools.cached_property
    def bags(self):
        return [
            FeatureBag(spec=dict(tilebag=dbx.quote(tilebag), extractor=self.spec['extractor'],), gpu_batch_size=self.gpu_batch_size)
            for tilebag in self.cfg.tilebagclip.shards
        ]
    
    @property
    def shards(self):
        return self.bags
    
    @property
    def n_bags(self):
        return len(self.bags)
    
    @property
    def n_shards(self):
        return self.n_bags
    
    @property
    def bag_lens(self):
        return self.read("bag_lens")
    
    @property
    def shard_lens(self):
        return self.bag_lens
    
    def labels(self):
        self.log.debug(f"Reading labels from {len(self.n_bags)} feature bags")
        label_list = []
        for featurebag in self.bags:
            try:
                label_list.append(featurebag.labels)  
            except Exception as e:
                if self.skip_unreadable:
                    continue
                else:
                    raise(e)
        labels = np.concatenate(label_list)
        self.log.debug(f"Combined labels: shape: {labels.shape}")
        return labels
     

def featurebagset(featurebagclip: FeatureBagClip, transform=None):
    return ClipDataset(spec=dict(clip=featurebagclip, transform=transform))
