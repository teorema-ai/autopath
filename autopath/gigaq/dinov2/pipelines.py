from functools import partial
from typing import Any, TypeVar, Optional, Callable, List


import torch
from torchvision import transforms

from dinov2.data import collate_data_and_cast, MaskingGenerator, SamplerType, make_data_loader

from .still import ARCH, GigaqStill
from . import backbone

from .augmentations import DataAugmentationDINO, dino_tile_transform

from ...pancan.tiles import PancanTileSet
from ...pancan.pipelines import pancan_tile_fold


def dino_augmentations(name="DINO_DEFAULT"):
    if name == "DINO_DEFAULT":
        return DataAugmentationDINO(
            global_crops_scale=[0.32, 1.0],
            local_crops_scale=[0.05, 0.32],
            local_crops_number=8,
            global_crops_size=224,
            local_crops_size=96,
        )
    else:
        raise ValueError(f"Unknown augmentation: {name}")


def dino_pancan_tile_dataset(name):
    if name == "CPTAC_8020_TRAIN":
        return PancanTileSet(spec=dict(tilebatch=f"@autopath.pancan.pipelines.pancan_tile_fold(name='CPTAC_8020_TRAIN')", 
                                       transform=f"@autopath.gigaq.dinov2.pipelines.dino_augmentations('DINO_DEFAULT')")).dataset
    elif name == "CPTAC_8020_TEST":
        return PancanTileSet(spec=dict(tilebatch=f"@autopath.pancan.pipelines.pancan_tile_fold(name='CPTAC_8020_TEST')", 
                                       transform=f"@autopath.gigaq.dinov2.pipelines.dino_augmentations('DINO_DEFAULT')")).dataset
    elif name == "CPTAC_9802_TRAIN":
        return PancanTileSet(spec=dict(tilebatch=f"@autopath.pancan.pipelines.pancan_tile_fold(name='CPTAC_9802_TRAIN')", 
                                       transform=f"@autopath.gigaq.dinov2.pipelines.dino_augmentations('DINO_DEFAULT')")).dataset
    elif name == "CPTAC_9802_TEST":
        return PancanTileSet(spec=dict(tilebatch=f"@autopath.pancan.pipelines.pancan_tile_fold(name='CPTAC_9802_TEST')", 
                                       transform=f"@autopath.gigaq.dinov2.pipelines.dino_augmentations('DINO_DEFAULT')")).dataset
    else:
        raise ValueError(f"Unknown dataset: {name}")


def dino_pancan_tile_dataloader(name, *, batch_size: int = 4, num_workers: int = 1) -> torch.utils.data.DataLoader:
    if name == "CPTAC_8020_TRAIN":
        dataset = dino_pancan_tile_dataset("CPTAC_8020_TRAIN")
    elif name == "CPTAC_8020_TEST":
        dataset = dino_pancan_tile_dataset("CPTAC_8020_TEST")
    elif name == "CPTAC_9802_TRAIN":
        dataset = dino_pancan_tile_dataset("CPTAC_9802_TRAIN")
    elif name == "CPTAC_9802_TEST":
        dataset = dino_pancan_tile_dataset("CPTAC_9802_TEST")
    else:
        raise ValueError(f"Unknown dataloader: {name}")
    global_crops_size: int = 224
    student_patch_size: int = 16
    ibot_mask_ratio_min_max: list[float] = [0.1, 0.5]
    ibot_mask_sample_probability: float = 0.5
    start_iter: int = 0
    shuffle: bool = True
    sampler_type: Optional[SamplerType] = SamplerType.SHARDED_INFINITE
    sampler_advance: int = 0
    collate_fn: Optional[Callable] = None
    drop_last: bool = True
    inputs_dtype: str = torch.half

    img_size = global_crops_size
    patch_size = student_patch_size
    n_tokens = (img_size // patch_size) ** 2
    mask_generator = MaskingGenerator(
        input_size=(img_size // patch_size, img_size // patch_size),
        max_num_patches=0.5 * img_size // patch_size * img_size // patch_size,
    )
    collate_fn = partial(
        collate_data_and_cast,
        mask_ratio_tuple=ibot_mask_ratio_min_max,
        mask_probability=ibot_mask_sample_probability,
        n_tokens=n_tokens,
        mask_generator=mask_generator,
        dtype=inputs_dtype,
    )
    data_loader = make_data_loader(
            dataset=dataset,
            batch_size=batch_size,
            num_workers=num_workers,
            shuffle=shuffle,
            seed=start_iter,  # TODO: Fix this -- cfg.train.seed
            sampler_type=sampler_type,
            sampler_advance=sampler_advance,  # TODO(qas): fix this -- start_iter * cfg.train.batch_size_per_gpu,
            drop_last=drop_last,
            collate_fn=collate_fn,
        )
    return data_loader


def gigapath_tile_backbone(name):
    if name == "GIGAPATH_BASELINE_BACKBONE":
        return backbone.gigapath_tile_backbone()
    else:
        raise ValueError(f"Unknown backbone: {name}")


def gigapath_tile_backbone_evaluator(name):
    if name == "GIGAPATH_BASELINE_BACKBONE":
        return backbone.BackboneEvaluator(gigapath_tile_backbone(name), transform=dino_augmentations())
    else:
        raise ValueError(f"Unknown backbone evaluator: {name}")


def gigaq_arch(name):
    if name == 'BASELINE':
        return ARCH()
    else:
        raise ValueError(f"Unknown gigaq arch: {name}")


def gigaq_still(name):
    if name == 'BASELINE_CPTAC_9802_TRAIN':
        return GigaqStill(spec=dict(dataloader="@autopath.gigaq.dinov2.pipelines.dino_pancan_tile_dataloader('CPTAC_9802_TRAIN')",
                               arch="@autopath.gigaq.dinov2.pipelines.gigaq_arch('BASELINE')",))
    if name == 'BASELINE_CPTAC_8020_TRAIN':
        return GigaqStill(spec=dict(dataloader="@autopath.gigaq.dinov2.pipelines.dino_pancan_tile_dataloader('CPTAC_8020_TRAIN')",
                               arch="@autopath.gigaq.dinov2.pipelines.gigaq_arch('BASELINE')",))
    
    else:
        raise ValueError(f"Unknown gigaq: {name}")



