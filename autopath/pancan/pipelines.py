import os
from typing import Optional

#TODO: REMOVE?
# import torch.multiprocessing as mp
# mp.set_start_method('spawn', force=True)

import dbx

from autopath.databits import ClipDataset
from autopath.pancan.tiles import PancanTileBag, PancanTileClip, PancanTileSplit, PancanTileFold, pancan_tileset

PANCAN_CPTAC = os.environ.get("PANCAN_CPTAC", "/mnt/labshare/SLIDES/CPTAC_downloads")
PANCAN_CPTAC_SAMPLE = os.path.join(PANCAN_CPTAC, "HNSCC/tfrecords/256px_256um/C3L-02621-23.tfrecords")
PANCAN_CPTAC_RESOLUTION = os.environ.get("PANCAN_CPTAC_RESOLUTION", "256px_256um")


# git commit -am "gigaq: PancanTileBag: READ"; dbx "autopath.pancan.pipelines.pancan_tile_bag('CPTAC_SAMPLE').read('tiles')"
# git commit -am "gigaq: PancanTileBag: READ"; dbx "autopath.pancan.pipelines.pancan_tile_bag('CPTAC_SAMPLE').read('labels')"
def pancan_tile_bag(name) -> PancanTileBag:
    if name == "CPTAC_SAMPLE":     
        return PancanTileBag(spec=dict(source=PANCAN_CPTAC_SAMPLE))
    else:
        raise ValueError(f"Unknown tile_bag: {name}")

# dbx "autopath.pancan.pipelines.pancan_tile_clip('CPTAC').build()"
def pancan_tile_clip(name) -> PancanTileClip:
    if name == "CPTAC":     
        return PancanTileClip(spec=dict(
                                source=PANCAN_CPTAC,
                                resolution=PANCAN_CPTAC_RESOLUTION,
        ))
    else:
        raise ValueError(f"Unknown tile clip: {name}")

# git commit -am 'gigaq: PancanTileSplit: BUILD'; dbx 'autopath.pancan.pipelines.pancan_tile_split("CPTAC_8020").build_tree()'
# git commit -am 'gigaq: PancanTileSplit: BUILD'; dbx 'autopath.pancan.pipelines.pancan_tile_split("CPTAC_9802").build_tree()'
def pancan_tile_split(name, train_fraction: Optional[float] = None) -> PancanTileSplit:
    if name == "CPTAC":
        assert train_fraction is not None, "train_fraction must be specified"
        return PancanTileSplit(spec=dict(clip=dbx.quote(pancan_tile_clip, 'CPTAC'), train_fraction=train_fraction))   
    elif name == "CPTAC_8020":
        assert train_fraction is None or train_fraction == 0.8, "train_fraction must be 0.8"   
        return PancanTileSplit(spec=dict(clip=dbx.quote(pancan_tile_clip, 'CPTAC'), train_fraction=0.8))
    elif name == "CPTAC_9802": 
        assert train_fraction is None or train_fraction == 0.98, "train_fraction must be 0.98"  
        return PancanTileSplit(spec=dict(clip=dbx.quote(pancan_tile_clip, 'CPTAC'), train_fraction=0.98)
        )
    else:
        raise ValueError(f"Unknown tile_split: {name}")

# dbx "autopath.pancan.pipelines.pancan_tile_fold('CPTAC_8020_TRAIN').build()"
# dbx "autopath.pancan.pipelines.pancan_tile_fold('CPTAC_8020_TEST').build()"
# dbx "autopath.pancan.pipelines.pancan_tile_fold('CPTAC_9802_TRAIN').build()"
# dbx "autopath.pancan.pipelines.pancan_tile_fold('CPTAC_9802_TEST').build()"
def pancan_tile_fold(name) -> PancanTileFold:
    if name == "CPTAC_8020_TEST":   
        return PancanTileFold(spec=dict(split=dbx.quote(pancan_tile_split, 'CPTAC_8020'), fold='test'))
    elif name == "CPTAC_8020_TRAIN":   
        return PancanTileFold(spec=dict(split=dbx.quote(pancan_tile_split, 'CPTAC_8020'), fold='train'))
    elif name == "CPTAC_9802_TEST":   
        return PancanTileFold(spec=dict(split=dbx.quote(pancan_tile_split, 'CPTAC_9802'), fold='test'))
    elif name == "CPTAC_9802_TRAIN":   
        return PancanTileFold(spec=dict(split=dbx.quote(pancan_tile_split, 'CPTAC_9802'), fold='train'))
    else:
        raise ValueError(f"Unknown tile_fold: {name}")

# dbx "autopath.pancan.pipelines.pancan_tile_set('CPTAC_8020_TRAIN').build()"
# dbx "autopath.pancan.pipelines.pancan_tile_set('CPTAC_8020_TEST').build()"
# dbx "autopath.pancan.pipelines.pancan_tile_set('CPTAC_9802_TRAIN').build()"
# dbx "autopath.pancan.pipelines.pancan_tile_set('CPTAC_9802_TEST').build()"
def pancan_tile_set(name) -> ClipDataset:
    if name == "CPTAC_8020_TEST":
        return pancan_tileset(clip=dbx.quote(pancan_tile_fold, 'CPTAC_8020_TEST'),)
    elif name == "CPTAC_8020_TRAIN":
        return pancan_tileset(clip=dbx.quote(pancan_tile_fold, 'CPTAC_8020_TRAIN'),)
    elif name == "CPTAC_9802_TEST":
        return pancan_tileset(clip=dbx.quote(pancan_tile_fold, 'CPTAC_9802_TEST'),)
    elif name == "CPTAC_9802_TRAIN":
        return pancan_tileset(clip=dbx.quote(pancan_tile_fold, 'CPTAC_9802_TRAIN'),)
    else:
        raise ValueError(f"Unknown tile_set: {name}")

