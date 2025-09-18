import os
from typing import Optional

#TODO: REMOVE?
# import torch.multiprocessing as mp
# mp.set_start_method('spawn', force=True)

from .tiles import PancanTileBag, PancanTileBags, PancanTileSplit, PancanTileFold, PancanTileSet, PancanTileDecks


PANCAN_CPTAC = os.environ.get("PANCAN_CPTAC", "/mnt/labshare/SLIDES/CPTAC_downloads")
PANCAN_CPTAC_SAMPLE = os.path.join(PANCAN_CPTAC, "HNSCC/tfrecords/256px_256um/C3L-02621-23.tfrecords")
PANCAN_CPTAC_RESOLUTION = os.environ.get("PANCAN_CPTAC_RESOLUTION", "256px_256um")


# dbx "autopath.pancan.pipelines.pancan_tile_bag('CPTAC_SAMPLE').read('tiles')"
# dbx "autopath.pancan.pipelines.pancan_tile_bag('CPTAC_SAMPLE').read('labels')"
def pancan_tile_bag(name) -> PancanTileBag:
    if name == "CPTAC_SAMPLE":     
        return PancanTileBag(spec=dict(source=PANCAN_CPTAC_SAMPLE))
    else:
        raise ValueError(f"Unknown tile_bag: {name}")

# dbx "autopath.pancan.pipelines.pancan_tile_bags('CPTAC').build()"
def pancan_tile_bags(name) -> PancanTileBags:
    if name == "CPTAC":     
        return PancanTileBags(spec=dict(
                                source=PANCAN_CPTAC,
                                resolution=PANCAN_CPTAC_RESOLUTION,
        ))
    else:
        raise ValueError(f"Unknown tile_batch: {name}")

# dbx "autopath.pancan.pipelines.pancan_tile_split('CPTAC_8020').build()"
# dbx "autopath.pancan.pipelines.pancan_tile_split('CPTAC_9802').build()"
def pancan_tile_split(name, train_fraction: Optional[float] = None) -> PancanTileSplit:
    if name == "CPTAC":
        assert train_fraction is not None, "train_fraction must be specified"
        return PancanTileSplit(spec=dict(tileshards=f"@autopath.pancan.pipelines.pancan_tile_bags('CPTAC')",
                                         train_fraction=train_fraction)
        )   
    elif name == "CPTAC_8020":
        assert train_fraction is None or train_fraction == 0.8, "train_fraction must be 0.8"   
        return PancanTileSplit(spec=dict(tileshards=f"@autopath.pancan.pipelines.pancan_tile_bags('CPTAC')",
                                         train_fraction=0.8)
        )
    elif name == "CPTAC_9802": 
        assert train_fraction is None or train_fraction == 0.98, "train_fraction must be 0.98"  
        return PancanTileSplit(spec=dict(tileshards=f"@autopath.pancan.pipelines.pancan_tile_bags('CPTAC')",
                                         train_fraction=0.98)
        )
    else:
        raise ValueError(f"Unknown tile_split: {name}")

# dbx "autopath.pancan.pipelines.pancan_tile_fold('CPTAC_8020_TRAIN').build()"
# dbx "autopath.pancan.pipelines.pancan_tile_fold('CPTAC_8020_TEST').build()"
# dbx "autopath.pancan.pipelines.pancan_tile_fold('CPTAC_9802_TRAIN').build()"
# dbx "autopath.pancan.pipelines.pancan_tile_fold('CPTAC_9802_TEST').build()"
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

# dbx "autopath.pancan.pipelines.pancan_tile_set('CPTAC_8020_TRAIN').build()"
# dbx "autopath.pancan.pipelines.pancan_tile_set('CPTAC_8020_TEST').build()"
# dbx "autopath.pancan.pipelines.pancan_tile_set('CPTAC_9802_TRAIN').build()"
# dbx "autopath.pancan.pipelines.pancan_tile_set('CPTAC_9802_TEST').build()"
def pancan_tile_set(name) -> PancanTileSet:
    if name == "CPTAC_8020_TEST":
        return PancanTileSet(spec=dict(tileshards=f"@autopath.pancan.pipelines.pancan_tile_fold('CPTAC_8020_TEST')",))
    elif name == "CPTAC_8020_TRAIN":
        return PancanTileSet(spec=dict(tileshards=f"@autopath.pancan.pipelines.pancan_tile_fold('CPTAC_8020_TRAIN')",))
    elif name == "CPTAC_9802_TEST":
        return PancanTileSet(spec=dict(tileshards=f"@autopath.pancan.pipelines.pancan_tile_fold('CPTAC_9802_TEST')",))
    elif name == "CPTAC_9802_TRAIN":
        return PancanTileSet(spec=dict(tileshards=f"@autopath.pancan.pipelines.pancan_tile_fold('CPTAC_9802_TRAIN')",))
    else:
        raise ValueError(f"Unknown tile_set: {name}")

# dbx "autopath.pancan.pipelines.pancan_tile_decks('CPTAC_9802_TEST_8', num_workers=8).build()"
# dbx "autopath.pancan.pipelines.pancan_tile_decks('CPTAC_9802_TEST_512', num_workers=8).build()"
# dbx "autopath.pancan.pipelines.pancan_tile_decks('CPTAC_8020_TRAIN_512').build()"
# dbx "autopath.pancan.pipelines.pancan_tile_decks('CPTAC_8020_TEST', shard_size=128, num_workers=2).build()"
def pancan_tile_decks(name, shard_size: Optional[int] = None, num_workers: int = 0) -> PancanTileDecks:
    if name == "CPTAC_8020_TEST":
        assert shard_size is not None, "shard_size must be specified"
        return PancanTileDeck(spec=dict(
            tileset=f"@autopath.pancan.pipelines.pancan_tile_set('CPTAC_8020_TEST')",
            shard_size=shard_size,),
            dataloader_num_workers=num_workers,
        )
    elif name == "CPTAC_8020_TRAIN":
        assert shard_size is not None, "shard_size must be specified"
        return PancanTileDeck(spec=dict(
            tileset=f"@autopath.pancan.pipelines.pancan_tile_set('CPTAC_8020_TRAIN')",
            shard_size=shard_size,),
            dataloader_num_workers=num_workers,
        )
    elif name == "CPTAC_8020_TEST_512":
        assert shard_size is None or shard_size == 512, "shard_size must be 512"
        return PancanTileDeck(spec=dict(
            tileset=f"@autopath.pancan.pipelines.pancan_tile_set('CPTAC_8020_TEST')",
            shard_size=512,),
            dataloader_num_workers=num_workers,
        )
    elif name == "CPTAC_8020_TRAIN_512":
        assert shard_size is None or shard_size == 512, "shard_size must be 512"
        return PancanTileDeck(spec=dict(
            tileset=f"@autopath.pancan.pipelines.pancan_tile_set('CPTAC_8020_TRAIN')",
            shard_size=512,),
            dataloader_num_workers=num_workers,
        )
    elif name == "CPTAC_9802_TEST_512":
        assert shard_size is None or shard_size == 512, "shard_size must be 512"
        return PancanTileDeck(spec=dict(
            tileset=f"@autopath.pancan.pipelines.pancan_tile_set('CPTAC_9802_TEST')",
            shard_size=512,),
            dataloader_num_workers=num_workers,
        )
    elif name == "CPTAC_9802_TRAIN_512":
        assert shard_size is None or shard_size == 512, "shard_size must be 512"
        return PancanTileDeck(spec=dict(
            tileset=f"@autopath.pancan.pipelines.pancan_tile_set('CPTAC_9802_TRAIN')",
            shard_size=512,),
            dataloader_num_workers=num_workers,
        )
    elif name == "CPTAC_9802_TEST_8":
        assert shard_size is None or shard_size == 8, "shard_size must be 8"
        return PancanTileDeck(spec=dict(
                                 tileset=f"@autopath.pancan.pipelines.pancan_tile_set('CPTAC_9802_TEST')",
                                 shard_size=8,),
            dataloader_num_workers=num_workers,
        )
    elif name == "CPTAC_9802_TRAIN_8":
        assert shard_size is None or shard_size == 8, "shard_size must be 8"
        return PancanTileDeck(spec=dict(
            tileset=f"@autopath.pancan.pipelines.pancan_tile_set('CPTAC_9802_TRAIN')",
            shard_size=8,),
            dataloader_num_workers=num_workers,
        )
    else:
        raise ValueError(f"Unknown tile deck: {name}")
