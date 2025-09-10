import os

from .tiles import PancanTileBag, PancanTileBatch, PancanTileSplit, PancanTileFold, PancanTileSet


PANCAN_DATASPACE = os.environ.get("PANCAN_DATASPACE", "/mnt/labshare/PERSONAL/dmitry/dbx") 
PANCAN_CPTAC = os.environ.get("PANCAN_CPTAC", "/mnt/labshare/SLIDES/CPTAC_downloads")
PANCAN_CPTAC_SAMPLE = os.path.join(PANCAN_CPTAC, "HNSCC/tfrecords/256px_256um/C3L-02621-23.tfrecords")
PANCAN_CPTAC_RESOLUTION = os.environ.get("PANCAN_CPTAC_RESOLUTION", "256px_256um")


def pancan_tile_bag(name, *, root=PANCAN_DATASPACE) -> PancanTileBag:
        if name == "PANCAN_CPTAC_SAMPLE":     
            return PancanTileBag(root, 
                                 spec=dict(source=PANCAN_CPTAC_SAMPLE), 
            )
        else:
            raise ValueError(f"Unknown slide tilebag: {name}")


def pancan_tile_batch(name, *, root=PANCAN_DATASPACE) -> PancanTileBatch:
        if name == "PANCAN_CPTAC":     
            return PancanTileBatch(root, 
                                   spec=dict(
                                    source=PANCAN_CPTAC,
                                    resolution=PANCAN_CPTAC_RESOLUTION,
                                  ), 
            )
        else:
            raise ValueError(f"Unknown slide tile_bag: {name}")


def pancan_tile_split(name, *, root=PANCAN_DATASPACE) -> PancanTileSplit:
        if name == "PANCAN_CPTAC_8020":   
            return PancanTileSplit(root, 
                                  spec=dict(tilebatch=f"@autopath.pancan.pipelines.pancan_tile_batch('PANCAN_CPTAC')",
                                            train_fraction=0.8,
                                  ),
            )
        elif name == "PANCAN_CPTAC_9802":   
            return PancanTileSplit(root, 
                                  spec=dict(tilebatch=f"@autopath.pancan.pipelines.pancan_tile_batch('PANCAN_CPTAC')",
                                            train_fraction=0.98,
                                  ),
            )
        else:
            raise ValueError(f"Unknown tile_split: {name}")


def pancan_tile_fold(name, *, root=PANCAN_DATASPACE) -> PancanTileFold:
        if name == "PANCAN_CPTAC_8020_TEST":   
            return PancanTileFold(root, 
                                  spec=dict(tilesplit=f"@autopath.pancan.pipelines.pancan_tile_split('PANCAN_CPTAC_8020')", fold='test',),
            )
        elif name == "PANCAN_CPTAC_8020_TRAIN":   
            return PancanTileFold(root, 
                                  spec=dict(tilesplit=f"@autopath.pancan.pipelines.pancan_tile_split('PANCAN_CPTAC_8020')", fold='train',),
            )
        elif name == "PANCAN_CPTAC_9802_TEST":   
            return PancanTileFold(root, 
                                  spec=dict(tilesplit=f"@autopath.pancan.pipelines.pancan_tile_split('PANCAN_CPTAC_9802')", fold='test',),
            )
        elif name == "PANCAN_CPTAC_9802_TRAIN":   
            return PancanTileFold(root, 
                                  spec=dict(tilesplit=f"@autopath.pancan.pipelines.pancan_tile_split('PANCAN_CPTAC_9802')", fold='train',),
            )
        else:
            raise ValueError(f"Unknown tile_fold: {name}")


def pancan_tile_set(name, *, root=PANCAN_DATASPACE,) -> PancanTileSet:
    if name == "PANCAN_CPTAC_8020_TEST":
            return PancanTileSet(root,
                                 spec=dict(tilebatch=f"@autopath.pancan.pipelines.pancan_tile_fold('PANCAN_CPTAC_8020_TEST')",),
            )
    elif name == "PANCAN_CPTAC_8020_TRAIN":
            return PancanTileSet(root,
                                spec=dict(tilebatch=f"@autopath.pancan.pipelines.pancan_tile_fold('PANCAN_CPTAC_8020_TRAIN')",),
                                        
            )
    elif name == "PANCAN_CPTAC_9802_TEST":
            return PancanTileSet(root,
                                 spec=dict(tilebatch=f"@autopath.pancan.pipelines.pancan_tile_fold('PANCAN_CPTAC_9802_TEST')",),
            )
    elif name == "PANCAN_CPTAC_9802_TRAIN":
            return PancanTileSet(root,
                                 spec=dict(tilebatch=f"@autopath.pancan.pipelines.pancan_tile_fold('PANCAN_CPTAC_9802_TRAIN')",),
            )
    else:
            raise ValueError(f"Unknown tile_set: {name}")
