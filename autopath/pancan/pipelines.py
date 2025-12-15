import os
from typing import Optional

#TODO: REMOVE?
# import torch.multiprocessing as mp
# mp.set_start_method('spawn', force=True)

import dbx

from autopath.databits import ClipDataset
from autopath.pancan.tiles import PancanTileBag, PancanTileBagClip, PancanTileBagSplit, PancanTileBagFold, pancan_tilebagset

PANCAN_CPTAC = os.environ.get("PANCAN_CPTAC", "/mnt/labshare/SLIDES/CPTAC_downloads")
PANCAN_CPTAC_SAMPLE = os.path.join(PANCAN_CPTAC, "HNSCC/tfrecords/256px_256um/C3L-02621-23.tfrecords")
PANCAN_CPTAC_RESOLUTION = os.environ.get("PANCAN_CPTAC_RESOLUTION", "256px_256um")


# git commit -am "gigaq: PancanTileBag: READ"; git commit -am "gigaq: PancanTileBag: READ"; dbx "autopath.pancan.pipelines.pancan_tile_bag('CPTAC_SAMPLE').read('tiles')"
# git commit -am "gigaq: PancanTileBag: READ"; dbx "autopath.pancan.pipelines.pancan_tile_bag('CPTAC_SAMPLE').read('labels')"
def pancan_tile_bag(name=None) -> PancanTileBag:
    if name is None:
        return PancanTileBag
    elif name == "CPTAC_SAMPLE":     
        return PancanTileBag(spec=dict(source=PANCAN_CPTAC_SAMPLE))
    else:
        raise ValueError(f"Unknown tile_bag: {name}")

# git commit -am "gigaq: PancanTileBagClip: BUILD"; dbx.print "autopath.pancan.pipelines.pancan_tile_bag_clip('CPTAC').build()"
def pancan_tile_bag_clip(name=None) -> PancanTileBagClip:
    if name is None:
        return PancanTileBagClip
    elif name == "CPTAC":     
        return PancanTileBagClip(spec=dict(
                                source=PANCAN_CPTAC,
                                resolution=PANCAN_CPTAC_RESOLUTION,
        ))
    else:
        raise ValueError(f"Unknown tile clip: {name}")

# git commit -am 'gigaq: PancanTileBagSplit: BUILD'; dbx.print 'autopath.pancan.pipelines.pancan_tile_bag_split("CPTAC_8020").build_tree()'
# git commit -am 'gigaq: PancanTileBagSplit: BUILD'; dbx.print 'autopath.pancan.pipelines.pancan_tile_bag_split("CPTAC_9802").build_tree()'
def pancan_tile_bag_split(name=None, train_fraction: Optional[float] = None) -> PancanTileBagSplit:
    if name is None:
        return PancanTileBagSplit
    elif name == "CPTAC":
        assert train_fraction is not None, "train_fraction must be specified"
        return PancanTileBagSplit(spec=dict(clip=dbx.quote(pancan_tile_bag_clip, 'CPTAC'), train_fraction=train_fraction))   
    elif name == "CPTAC_8020":
        assert train_fraction is None or train_fraction == 0.8, "train_fraction must be 0.8"   
        return PancanTileBagSplit(spec=dict(clip=dbx.quote(pancan_tile_bag_clip, 'CPTAC'), train_fraction=0.8))
    elif name == "CPTAC_9802": 
        assert train_fraction is None or train_fraction == 0.98, "train_fraction must be 0.98"  
        return PancanTileBagSplit(spec=dict(clip=dbx.quote(pancan_tile_bag_clip, 'CPTAC'), train_fraction=0.98)
        )
    else:
        raise ValueError(f"Unknown tile_split: {name}")

# dbx "autopath.pancan.pipelines.pancan_tile_bag_fold('CPTAC_8020_TRAIN').build()"
# dbx "autopath.pancan.pipelines.pancan_tile_bag_fold('CPTAC_8020_TEST').build()"
# dbx "autopath.pancan.pipelines.pancan_tile_bag_fold('CPTAC_9802_TRAIN').build()"
# dbx "autopath.pancan.pipelines.pancan_tile_bag_fold('CPTAC_9802_TEST').build()"
def pancan_tile_bag_fold(name=None) -> PancanTileBagFold:
    if name is None:
        return PancanTileBagFold
    elif name == "CPTAC_8020_TEST":   
        return PancanTileBagFold(spec=dict(split=dbx.quote(pancan_tile_bag_split, 'CPTAC_8020'), fold='test'))
    elif name == "CPTAC_8020_TRAIN":   
        return PancanTileBagFold(spec=dict(split=dbx.quote(pancan_tile_bag_split, 'CPTAC_8020'), fold='train'))
    elif name == "CPTAC_9802_TEST":   
        return PancanTileBagFold(spec=dict(split=dbx.quote(pancan_tile_bag_split, 'CPTAC_9802'), fold='test'))
    elif name == "CPTAC_9802_TRAIN":   
        return PancanTileBagFold(spec=dict(split=dbx.quote(pancan_tile_bag_split, 'CPTAC_9802'), fold='train'))
    else:
        raise ValueError(f"Unknown tile_fold: {name}")

# dbx "autopath.pancan.pipelines.pancan_tile_bag_set('CPTAC_8020_TRAIN').build()"
# dbx "autopath.pancan.pipelines.pancan_tile_bag_set('CPTAC_8020_TEST').build()"
# dbx "autopath.pancan.pipelines.pancan_tile_bag_set('CPTAC_9802_TRAIN').build()"
# dbx "autopath.pancan.pipelines.pancan_tile_bag_set('CPTAC_9802_TEST').build()"
def pancan_tile_bag_set(name=None) -> ClipDataset:
    if name is None:
        return ClipDataset
    elif name == "CPTAC_8020_TEST":
        return pancan_tilebagset(clip=dbx.quote(pancan_tile_bag_fold, 'CPTAC_8020_TEST'),)
    elif name == "CPTAC_8020_TRAIN":
        return pancan_tilebagset(clip=dbx.quote(pancan_tile_bag_fold, 'CPTAC_8020_TRAIN'),)
    elif name == "CPTAC_9802_TEST":
        return pancan_tilebagset(clip=dbx.quote(pancan_tile_bag_fold, 'CPTAC_9802_TEST'),)
    elif name == "CPTAC_9802_TRAIN":
        return pancan_tilebagset(clip=dbx.quote(pancan_tile_bag_fold, 'CPTAC_9802_TRAIN'),)
    else:
        raise ValueError(f"Unknown tile_bag_set: {name}")

