from dinov2.data import collate_data_and_cast, MaskingGenerator
from .augmentations import DataAugmentationDINO
from ...pancan.tiles import (
    CPTAC_ROOT,
    CPTAC_RESOLUTION,
    PancanSlideShard, 
    PancanSlideBatch, 
    PancanTileset
)

def pancan_slidebatch(
    *,
    path: str = CPTAC_ROOT,
    resolution: str = CPTAC_RESOLUTION, # 256px_256um,
    train_fraction: float = 0.8,
    randomize: bool = False,
    seed: int = 42,  
    verbose: bool = True,
    debug: bool = False,
):     
    databatch = PancanSlideBatch(
        cfg=dict(
            source=path,
            resolution=resolution,
            train_fraction=train_fraction,
            randomize=randomize,
            seed=seed,
        ),
        verbose=verbose,
        debug=debug,
    )
    return databatch


def pancan_tileset(
    *,
    path: str = CPTAC_ROOT,
    resolution: str = CPTAC_RESOLUTION,
    global_crops_scale=[0.32, 1.0],
    local_crops_scale=[0.05, 0.32],
    local_crops_number=8,
    global_crops_size=224,
    local_crops_size=96,
    split: str = "train",
    train_fraction: float = 0.8,
    randomize: bool = False,
    seed: int = 42,  
    verbose: bool = True,
    debug: bool = False,
):  
    split = split.lower()
    slidebatch = pancan_slidebatch(
        path=path, 
        resolution=resolution, 
        train_fraction=train_fraction, 
        randomize=randomize, 
        seed=seed, 
        verbose=verbose, 
        debug=debug,
    )
    transform = DataAugmentationDINO(
        global_crops_scale,
        local_crops_scale,
        local_crops_number,
        global_crops_size=global_crops_size,
        local_crops_size=local_crops_size,
    )
    dataset = PancanTileset(slidebatch, split=split, transform=transform, verbose=verbose, debug=debug)
    return dataset