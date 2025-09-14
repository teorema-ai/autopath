import os

from .tiles import PancanTileBag, PancanTileBatch, PancanTileSplit, PancanTileFold, PancanTileSet


PANCAN_CPTAC = os.environ.get("PANCAN_CPTAC", "/mnt/labshare/SLIDES/CPTAC_downloads")
PANCAN_CPTAC_SAMPLE = os.path.join(PANCAN_CPTAC, "HNSCC/tfrecords/256px_256um/C3L-02621-23.tfrecords")
PANCAN_CPTAC_RESOLUTION = os.environ.get("PANCAN_CPTAC_RESOLUTION", "256px_256um")


def pancan_tile_bag(name) -> PancanTileBag:
    if name == "CPTAC_SAMPLE":     
        return PancanTileBag(spec=dict(source=CPTAC_SAMPLE))
    else:
        raise ValueError(f"Unknown tile_bag: {name}")


def pancan_tile_batch(name) -> PancanTileBatch:
    if name == "PANCAN_CPTAC":     
        return PancanTileBatch(spec=dict(
                                source=PANCAN_CPTAC,
                                resolution=PANCAN_CPTAC_RESOLUTION,
        ))
    else:
        raise ValueError(f"Unknown tile_batch: {name}")


def pancan_tile_split(name) -> PancanTileSplit:
    if name == "CPTAC_8020":   
        return PancanTileSplit(spec=dict(tilebatch=f"@autopath.pancan.pipelines.pancan_tile_batch('PANCAN_CPTAC')",
                                         train_fraction=0.8)
        )
    elif name == "CPTAC_9802":   
        return PancanTileSplit(spec=dict(tilebatch=f"@autopath.pancan.pipelines.pancan_tile_batch('PANCAN_CPTAC')",
                                         train_fraction=0.98)
        )
    else:
        raise ValueError(f"Unknown tile_split: {name}")


def pancan_tile_fold(name) -> PancanTileFold:
    if name == "CPTAC_8020_TEST":   
        return PancanTileFold(spec=dict(tilesplit=f"@autopath.pancan.pipelines.pancan_tile_split('CPTAC_8020')", fold='test'))
    elif name == "CPTAC_8020_TRAIN":   
        return PancanTileFold(spec=dict(tilesplit=f"@autopath.pancan.pipelines.pancan_tile_split('CPTAC_8020')", fold='train'))
    elif name == "CPTAC_9802_TEST":   
        return PancanTileFold(spec=dict(tilesplit=f"@autopath.pancan.pipelines.pancan_tile_split('CPTAC_9802')", fold='test'))
    elif name == "CPTAC_9802_TRAIN":   
        return PancanTileFold(spec=dict(tilesplit=f"@autopath.pancan.pipelines.pancan_tile_split('CPTAC_9802')", fold='train'))
    else:
        raise ValueError(f"Unknown tile_fold: {name}")


def pancan_tile_set(name) -> PancanTileSet:
    if name == "CPTAC_8020_TEST":
        return PancanTileSet(spec=dict(tilebatch=f"@autopath.pancan.pipelines.pancan_tile_fold('CPTAC_8020_TEST')",))
    elif name == "CPTAC_8020_TRAIN":
        return PancanTileSet(spec=dict(tilebatch=f"@autopath.pancan.pipelines.pancan_tile_fold('CPTAC_8020_TRAIN')",))
    elif name == "CPTAC_9802_TEST":
        return PancanTileSet(spec=dict(tilebatch=f"@autopath.pancan.pipelines.pancan_tile_fold('CPTAC_9802_TEST')",))
    elif name == "CPTAC_9802_TRAIN":
        return PancanTileSet(spec=dict(tilebatch=f"@autopath.pancan.pipelines.pancan_tile_fold('CPTAC_9802_TRAIN')",))
    else:
        raise ValueError(f"Unknown tile_set: {name}")
