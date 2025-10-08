import os
from typing import Optional, List

import torch.multiprocessing as mp

from ..pancan.features import FeatureBag, FeatureBags
from ..pancan.probe import (
    FeatureBagsProbe, 
    FeatureBagsPairwiseDistancesProbe,
    FeatureBagsUniquePairwiseDistancesProbe,
    FeatureBags2NNDimProbe,
)

from .dinov2.backbone import BackboneEvaluator, SidebandBackboneEvaluator


mp.set_start_method("spawn", force=True)


def gigapath_backbone_evaluator(name, *, device: str = 'cuda', sideband: bool = False, capture_blocks: Optional[List[int]] = None):
    if name == "GIGAPATH_BASELINE_BACKBONE_EVALUATOR":
        if sideband:
            return SidebandBackboneEvaluator(device=device, capture_blocks=capture_blocks)
        else:
            return BackboneEvaluator(device=device)
    else:
        raise ValueError(f"Unknown backbone evaluator: {name}")


# dbx "autopath.gigaq.pipelines.gigapath_feature_bag('GIGAPATH_BASELINE_CPTAC_SAMPLE').build()"
def gigapath_feature_bag(name, *, device = 'cuda', gpu_batch_size=1024,) -> FeatureBag:
    if name == "GIGAPATH_BASELINE_CPTAC_SAMPLE":
        return FeatureBag(spec=dict(tilebag="@autopath.pancan.pipelines.pancan_tile_bag('CPTAC_SAMPLE')",))
    else:
        raise ValueError(f"Unknown feature bag: {name}")


# dbx "autopath.gigaq.pipelines.gigapath_feature_bags('GIGAPATH_BASELINE_CPTAC_8020_TEST').build()"
# dbx "autopath.gigaq.pipelines.gigapath_feature_bags('GIGAPATH_BASELINE_CPTAC_9802_TEST', sideband=True, num_gpus=3).build()"
def gigapath_feature_bags(name, *, sideband: bool = False, capture_blocks: Optional[List[int]] = None, num_gpus: int = 1, gpu_batch_size=1024) -> FeatureBags:
    def get_extractor(name, sideband: bool = False, capture_blocks: Optional[List[int]] = None):
        if sideband: 
            return f"@autopath.gigaq.pipelines.gigapath_backbone_evaluator('GIGAPATH_BASELINE_BACKBONE_EVALUATOR', sideband=True, capture_blocks={repr(capture_blocks)})"
        else:
            return f"@autopath.gigaq.pipelines.gigapath_backbone_evaluator('GIGAPATH_BASELINE_BACKBONE_EVALUATOR')"

    if name == "GIGAPATH_BASELINE_SIDEBAND_CPTAC_9802_TEST":
        extractor=get_extractor('GIGAPATH_BASELINE_BACKBONE_EVALUATOR', sideband=sideband, capture_blocks=capture_blocks)
        tilebags="@autopath.pancan.pipelines.pancan_tile_fold('CPTAC_9802_TEST')"
    elif name == "GIGAPATH_BASELINE_CPTAC_9802_TEST":
        extractor=get_extractor('GIGAPATH_BASELINE_BACKBONE_EVALUATOR', sideband=sideband, capture_blocks=capture_blocks)
        tilebags="@autopath.pancan.pipelines.pancan_tile_fold('CPTAC_9802_TEST')"
    elif name == "GIGAPATH_BASELINE_CPTAC_8020_TEST":
        extractor=get_extractor('GIGAPATH_BASELINE_BACKBONE_EVALUATOR', sideband=sideband, capture_blocks=capture_blocks)
        tilebags="@autopath.pancan.pipelines.pancan_tile_fold('CPTAC_8020_TEST')"
    else:
        raise ValueError(f"Unknown feature bags: {name}")
    devices = [f"cuda:{i}" for i in range(num_gpus)]
    return FeatureBags(devices=devices, gpu_batch_size=gpu_batch_size, spec=dict(extractor=extractor, tilebags=tilebags))


# dbx "autopath.gigaq.pipelines.gigapath_feature_bags_probe('GIGAPATH_BASELINE_CPTAC_8020_TEST', n_bins=2).build()"
def gigapath_feature_bags_probe(name, n_bins: int = 2) -> FeatureBagsProbe:
    if name == "GIGAPATH_BASELINE_CPTAC_8020_TEST":
        return FeatureBagsProbe(
                    spec=dict(featurebags="@autopath.gigaq.pipelines.gigapath_feature_bags('GIGAPATH_BASELINE_CPTAC_8020_TEST')",
                              n_bins=n_bins,
        ))
    else:
        raise ValueError(f"Unknown feature bags probe: {name}")

# dbx "autopath.gigaq.pipelines.gigapath_feature_bags_pairwise_distances_probe('GIGAPATH_BASELINE_CPTAC_8020_TEST', sideband_layer='', row_batch_size=4000, n_gpus=4).build()"
def gigapath_feature_bags_pairwise_distances_probe(name, sideband_layer: Optional[str] = None, capture_blocks: Optional[List[int]] = None, row_batch_size: int = 1000, n_gpus: int = 1) -> FeatureBagsPairwiseDistancesProbe:
    sideband = sideband_layer is not None
    devices = [f"cuda:{i}" for i in range(n_gpus)]
    if name == "GIGAPATH_BASELINE_CPTAC_8020_TEST":
        return FeatureBagsPairwiseDistancesProbe(
                    spec=dict(featurebags=f"@autopath.gigaq.pipelines.gigapath_feature_bags('GIGAPATH_BASELINE_CPTAC_8020_TEST', sideband={sideband}, capture_blocks={capture_blocks})",
                              sideband_layer=sideband_layer,
                              row_batch_size=row_batch_size,
                    ),
                    devices=devices,  
        )
    else:
        raise ValueError(f"Unknown feature bags pairwise distances probe: {name}")


# dbx "autopath.gigaq.pipelines.gigapath_feature_bags_unique_pairwise_distances_probe('GIGAPATH_BASELINE_CPTAC_8020_TEST', sideband_layer='', row_batch_size=4000, subsample_fraction=0.05, n_gpus=3).build()"
def gigapath_feature_bags_unique_pairwise_distances_probe(name, sideband_layer: Optional[str] = None, capture_blocks: Optional[List[int]] = None, row_batch_size: int = 1000, subsample_fraction: float = 1.0, n_gpus: int = 1) -> FeatureBagsUniquePairwiseDistancesProbe:
    devices = [f"cuda:{i}" for i in range(n_gpus)]
    if name == "GIGAPATH_BASELINE_CPTAC_8020_TEST":
        return FeatureBagsUniquePairwiseDistancesProbe(
                    spec=dict(featurebags_pairwise_distances_probe=f"@autopath.gigaq.pipelines.gigapath_feature_bags_pairwise_distances_probe('GIGAPATH_BASELINE_CPTAC_8020_TEST', sideband_layer='{sideband_layer}', capture_blocks={capture_blocks}, row_batch_size={row_batch_size}, subsample_fraction={subsample_fraction})",),
                    devices=devices,
        )
    else:
        raise ValueError(f"Unknown feature bags unique pairwise distances probe: {name}")
    

# dbx "autopath.gigaq.pipelines.gigapath_feature_bags_2nn_dim_probe('GIGAPATH_BASELINE_CPTAC_8020_TEST', sideband_layer='', row_batch_size=4000, n_workers=3).build()"
def gigapath_feature_bags_2nn_dim_probe(name, sideband_layer: Optional[str] = None, capture_blocks: Optional[List[int]] = None, row_batch_size: int = 1000, n_workers: int = 1) -> FeatureBags2NNDimProbe:
    if name == "GIGAPATH_BASELINE_CPTAC_8020_TEST":
        return FeatureBags2NNDimProbe(
                    spec=dict(featurebags_pairwise_distances_probe=f"@autopath.gigaq.pipelines.gigapath_feature_bags_pairwise_distances_probe('GIGAPATH_BASELINE_CPTAC_8020_TEST', sideband_layer='{sideband_layer}', capture_blocks={capture_blocks}, row_batch_size={row_batch_size})",),
                    n_workers=n_workers,
        )
    else:
        raise ValueError(f"Unknown feature bags 2nn dim probe: {name}")