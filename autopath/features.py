import copy
from dataclasses import dataclass
import functools
import gc
import math
import queue
import threading
from typing import Optional, Callable, Sequence


import tqdm

import numpy as np


import torch

import dbx
from dbx import (
    Datablock,
    Databag,
)

from autopath.databits import DataShard, DataClip, DataClipDataset
from .tiles import TileShard, TileClip


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

"""
#TODO: #REMOVE
class FeatureBag(Datablock, Bag):
    VERSION = 3
    @dataclass
    class CONFIG(Datablock.CONFIG):
        tilebag: TileBag
        extractor: Callable

    @property
    def has_sideband(self):
        return hasattr(self.config.extractor, 'sideband_layers')

    def __post_init__(self):
        self.TOPICFILES = {
            'features': 'features.npy',
            'sideband': None
        }
        if self.has_sideband:
            self.TOPICFILES['sideband'] = 'sideband.npz'
        return self

    def __len__(self):
        return len(self.labels)

    def store(self, features, sideband=None):
        assert sideband is not None or not self.has_sideband, "Expected sideband tensors to store"
        assert self.has_sideband or sideband is None, "Need a sideband extractor to store sideband tensors"
        #TODO: check for consistency with self.config.tilebag
        self.__pre_build__()
        dbx.write_tensor(features, self.path('features', ensure_dirpath=True))
        if self.has_sideband:
            dbx.write_tensors(self.path('sideband', ensure_dirpath=True), **sideband)
        self._write_journal_entry(event="store")
        self.__post_build__()
        return self

    def read(self, topic):
        if topic == 'features':
            return dbx.read_tensor(self.path(topic))
        elif topic == 'sideband':
            if self.has_sideband:
                sideband = dbx.read_tensors(self.path('sideband'), *self.config.extractor.sideband_layers)
                return sideband
            else:
                return None
        else:
            raise ValueError(f"Unknown topic: {topic}")

    def UNSAFE_clear(self):
        for bag in self.bags:
            bag.UNSAFE_clear()
        super().UNSAFE_clear()

    @functools.cached_property
    def tensor(self):
        return self.features

    @functools.cached_property
    def features(self):
        return self.read('features')
    
    @functools.cached_property
    def sideband(self):
        return self.read('sideband') if self.has_sideband else None
        
    @functools.cached_property
    def labels(self):
        return self.cfg.tilebag.labels

    @property
    def name(self):
        return self.cfg.tilebag.name


class FeatureBags(Datablock):
    VERSION = 2
    TOPICFILES = {'bag_lens': 'bag_lens.pt'}
    @dataclass
    class CONFIG(Datablock.CONFIG):
        extractor: Callable
        tilebags: TileBags
        lo: int = 0
        hi: Optional[int] = None

    def __init__(self, *args, devices: list[str] = 'cuda:0', gpu_batch_size: int = 16, **kwargs):
        super().__init__(*args, devices=[devices] if isinstance(devices, str) else devices, gpu_batch_size=gpu_batch_size, **kwargs)

    @property
    def lo(self):
        return self.config.lo

    @property
    def hi(self):
        return self.config.hi if self.config.hi is not None else len(self.config.tilebags)

    @property
    def has_sideband(self):
        return hasattr(self.config.extractor, 'sideband_layers')

    @functools.cached_property
    def bags(self):
        featurebags = [FeatureBag(root=self.root if not self._autoroot else None,
                                      spec=dict(tilebag=dbx.quote(tilebag), extractor=self.spec['extractor'],))
                        for tilebag in self.config.tilebags.bags[self.lo:self.hi]
        ]
        return featurebags
    
    @property
    def features(self):
        return torch.cat([bag.features for bag in self.bags])
    
    @property
    def labels(self):
        return np.concatenate([bag.labels for bag in self.bags])

    @property
    def tensor(self):
        return self.features
    
    def sideband(self, layer):
        sidebands = []
        self.log.info(f"Collecting sideband {layer} from {len(self.bags)} feature bags")
        skipped = 0
        successful = 0
        for bag in self.bags:
            try:
                _sideband = bag.sideband
                sidebands.append(bag.sideband[layer])
                del _sideband
                successful += 1
            except Exception:
                self.log.info(f"Skipping corrupted bag {bag}")
                skipped += 1
        self.log.info(f"Collected {successful}/{len(self.bags)} sidebands, skipped {skipped}")
        sideband = torch.cat(sidebands)
        return sideband

    def __len__(self):
        return len(self.bags)
    
    def size(self):
        return sum([len(bag) for bag in self.bags])

    def __build__(self):
        #TODO: #REFACTOR through dbx.TorchXXXDatashardBatchBuilder
        self.log.verbose(f"Building {len(self.bags)} feature bags")
        bag_lens = []
        remaining_bags = []
        for featurebag in self.bags:
            if featurebag.valid():
                self.log.verbose(f"Skipping existing feature bag {featurebag.hashpath()}")
                bag_lens.append(len(featurebag))
            else:
                remaining_bags.append(featurebag)
        self.log.verbose(f"Building {len(remaining_bags)} remaining feature bags on devices: {self.devices}")
        if len(remaining_bags) > 0:
            result_queue = queue.Queue()
            done_queue = queue.Queue()
            abort_event = threading.Event()
            progress_bar = tqdm.tqdm(total=len(remaining_bags))
            feature_bag_lists = np.array_split(remaining_bags, len(self.devices))
            threads = [
                threading.Thread(target=self.__build_bags__, args=(feature_bag_list, device, result_queue, done_queue, abort_event, progress_bar))
                for feature_bag_list, device in zip(feature_bag_lists, self.devices)
            ]
            for thread in threads:
                thread.start()
            while len(bag_lens) < len(self.bags):
                success, payload = result_queue.get()
                if success:
                    bag_len = payload
                    bag_lens.append(bag_len)
                    e = None
                else:
                    e = payload
                    self.log.info(f"Received error from one of the threads on result_queue. Abandoning result_queue polling.")
                    break
            self.log.debug(f"Production loop done, feeding done_queue")
            for _ in range(len(self.devices)):
                done_queue.put(None)
            self.log.debug(f"Joining threads")
            for thread in threads:
                thread.join()
            if e is not None:
                self.log.debug("Raising exception")
                raise e
            self.log.debug("Threads successfully joined")
        dbx.write_tensor(torch.tensor(bag_lens), self.path('bag_lens', ensure_dirpath=True))
        return self

    def __build_bags__(self, featurebags: Sequence[FeatureBag], device: str, result_queue: queue.Queue, done_queue: queue.Queue, abort_event: threading.Event, progress_bar):
        self.log.debug(f"Building {len(featurebags)} feature bags on device: {device}")
        extractor = copy.deepcopy(self.config.extractor).to(device).eval()
        for featurebag in featurebags:
            exception = None
            try:
                featurelen = self.__build_bag__(featurebag, extractor, device, abort_event)
            except Exception as e:
                featurelen = 0
                exception = e
                self.log.info(f"ERROR building feature bag {featurebag.hashpath()} on device: {device}")
            if exception is not None:
                result_queue.put((False, exception))
                break
            result_queue.put((True, featurelen))
            progress_bar.update(1)
        if exception is None:
            self.log.debug(f"Done building {len(featurebags)} feature bags on device: {device}")
        else:
            self.log.debug(f"Abandoning building {len(featurebags)} feature bags on device: {device} due to an exception")
        self.log.debug(f"Waiting on the done_queue on device: {device}")
        while True:
            item = done_queue.get()
            if item is None:
                self.log.debug(f"Done message received on the done_queue on device: {device}")
                break

    def __build_bag__(self, featurebag: FeatureBag, extractor: Callable, device: str, abort_event):
        self.log.verbose(f"Building new feature bag {featurebag.hashpath()} on device: {device}")
        tilebag = featurebag.config.tilebag
        feature_list = []
        sideband_list = []
        for k in range(math.ceil(len(tilebag.tiles)/self.gpu_batch_size)):
            if abort_event.is_set():
                return
            m = k*self.gpu_batch_size
            n = min((k+1)*self.gpu_batch_size, len(tilebag.tiles))
            batch = tilebag.tiles[m:n].to(device)
            self.log.verbose(f"Evaluating batch {k}: {m}:{n} out of {len(tilebag.tiles)} on device: {device}")
            _feature = extractor(batch).to('cpu')
            self.log.debug(f"Evaluated batch to a _feature of shape {_feature.shape} on device: {device}")
            del batch
            if self.has_sideband:
                _sideband = tensors_to_device(extractor.sideband, 'cpu', detach=True)
                assert set(_sideband.keys()) == set(extractor.sideband_layers), f"_sideband keys must match sideband_layers: {_sideband.keys()} != {extractor.sideband_layers}"
                _sideband_shapes = {k: v.shape for k, v in _sideband.items()}
                self.log.debug(f"Captured _sideband with shapes {_sideband_shapes} on device: {device}")
                sideband_list.append(_sideband)
            gc.collect()
            torch.cuda.empty_cache()
            self.log.verbose(f"done")
            feature_list.append(_feature)
        self.log.debug(f"Concatenating {len(feature_list)} device batch features on device: {device}")
        features = torch.cat(feature_list)
        if hasattr(extractor, 'sideband'):
            self.log.debug(f"Concatenating {len(sideband_list)} device batch sidebands on device: {device}")
            sideband = cat_tensor_dicts(sideband_list)
            if self.debug:
                sideband_shapes = {k: v.shape for k, v in sideband.items()}
                self.log.debug(f"Storing features of shape {features.shape} and sidebands of shapes {sideband_shapes} on device: {device}")
            featurebag.store(features, sideband)
        else:
            featurebag.store(features)
        lenfeatures = len(features)
        return lenfeatures

    def __read__(self, topic):
        bag_lens = dbx.read_tensor(self.path(topic))
        return bag_lens

    @functools.cached_property
    def bag_lens(self):
        return self.read('bag_lens')
"""

class FeatureShard(DataShard):
    VERSION = 1
    @dataclass
    class CONFIG(Datablock.CONFIG):
        tileshard: TileShard
        extractor: Callable

    def __init__(self, *args, gpu_batch_size: int = 16, **kwargs):
        super().__init__(*args, gpu_batch_size=gpu_batch_size, **kwargs)

    def __post_init__(self):
        self.TOPICFILES = {
            'features': 'features.npy',
        }
        if self.has_sideband:
            for layer in self.cfg.extractor.sideband_layers:
                self.TOPICFILES[f'sideband_{layer}' ] = \
                    f'sideband_{layer}.npy'
        return self

    def __len__(self):
        return len(self.labels)
    
    @property
    def has_sideband(self):
        return hasattr(self.cfg.extractor, 'sideband_layers')

    def __build__(self, extractor):
        tileshard = self.cfg.tileshard
        feature_list = []
        sideband_list = []
        for k in range(math.ceil(len(tileshard.tiles)/self.gpu_batch_size)):
            m = k*self.gpu_batch_size
            n = min((k+1)*self.gpu_batch_size, len(tileshard.tiles))
            batch = tileshard.tiles[m:n].to(self.device)
            self.log.verbose(f"Evaluating batch {k}: {m}:{n} out of {len(tileshard.tiles)} on device: {self.device}")
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
            self.log.verbose(f"done")
        self.log.debug(f"Concatenating {len(feature_list)} device batch features on device: {self.device}")
        features = torch.cat(feature_list)
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
        return zip(self.cfg.tileshard.labels, self.cfg.tileshard.tiles)


class FeatureClip(DataClip):
    VERSION = 1

    @dataclass
    class CONFIG:
        tileclip: TileClip
        extractor: Callable

    def __init__(self, *args, devices: list[str] = ["cuda"], gpu_batch_size: int = 16, skip_unreadable: bool = True, **kwargs):
        super().__init__(*args, devices=devices, gpu_batch_size=gpu_batch_size, skip_unreadable=skip_unreadable, **kwargs)
        self.log.debug(f"devices={self.devices}, gpu_batch_size={self.gpu_batch_size}, skip_unreadable={self.skip_unreadable}")

    def __build__(self):
        shards = self.shards
        self.log.debug(f"Formed {len(shards)} FeatureShards.  Looking for missing shards")
        missing_shards = [shard for shard in shards if not shard.valid()]
        self.log.debug(f"Found {len(missing_shards)} missing shards")
        self.log.debug(f"Building all missing features shards using devices {self.devices} and gpu_batch_size {self.gpu_batch_size}")
        built_shards = dbx.TorchMultithreadingDatashardBatchBuilder(devices=self.devices, log=self.log).build_shards(missing_shards, self.cfg.extractor)
        self.log.verbose(f"Built all missing features shards: {len(built_shards)}")
        self.leave_breadcrumbs()
        return self
    
    def features(self):
        self.log.debug(f"Reading features from {len(self.n_shards)} feature shards")
        feature_list = []
        for featureshard in self.shards:
            try:
                feature_list.append(featureshard.features)
            except Exception as e:
                if self.skip_unreadable:
                    continue
                else:
                    raise(e)
        features = torch.stack(feature_list)
        self.log.debug(f"Combined features: shape: {features.shape}")
        return features
    
    def tiles(self):
        self.log.debug(f"Reading tiles from {len(self.n_shards)} feature shards")
        tile_list = []
        for featureshard in self.shards:
            try:
                _tileshard_tiles = featureshard.cfg.tileshard.tiles
                tileshard_tiles = (
                    self.cfg.extractor.transform(_tileshard_tiles) 
                    if self.cfg.extractor.transform is not None 
                    else _tileshard_tiles
                )
                del _tileshard_tiles
                tile_list.append(tileshard_tiles)
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
        for featureshard in self.shards:
            try:
                sideband_list.append(featureshard.sideband(layer))
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
    def shards(self):
        return [
            FeatureShard(spec=dict(tileshard=dbx.quote(tileshard), extractor=self.spec['extractor'],), gpu_batch_size=self.gpu_batch_size)
            for tileshard in self.cfg.tileshards.shards
        ]
    
    @property
    def n_shards(self):
        return len(self.shards)
    
    def labels(self):
        self.log.debug(f"Reading labels from {len(self.n_shards)} feature shards")
        label_list = []
        for featureshard in self.shards:
            try:
                label_list.append(featureshard.labels)  
            except Exception as e:
                if self.skip_unreadable:
                    continue
                else:
                    raise(e)
        labels = np.concatenate(label_list)
        self.log.debug(f"Combined labels: shape: {labels.shape}")
        return labels
     

def featureset(featureclip: FeatureClip, 
               *,
               debug: bool = False,
               verbose: bool = False,
               log = None,
):
    featureclip = dbx.eval_term(featureclip)
    return DataClipDataset(spec=dict(clip=featureclip.shards,),
                           debug=debug, 
                           verbose=verbose,
                           log=log,
    )
