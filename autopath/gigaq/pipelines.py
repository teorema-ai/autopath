import os
from typing import Optional, List

import torch.multiprocessing as mp

from ..pancan.features import FeatureBag, FeatureBags
from ..pancan.probe import (
    FeatureBagsProbe, 
    FeaturesPairwiseDistances,
    FeaturesSortedDistances,
    Features2NNDistances,
    Features2NNDim,
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
        return FeatureBag(spec=dict(tilebag="$autopath.pancan.pipelines.pancan_tile_bag('CPTAC_SAMPLE')",))
    else:
        raise ValueError(f"Unknown feature bag: {name}")


# dbx "autopath.gigaq.pipelines.gigapath_feature_bags('GIGAPATH_BASELINE_CPTAC_8020_TEST').build()"
def gigapath_feature_bags(name) -> FeatureBags:
    def get_extractor(name, sideband: bool = False, capture_blocks: Optional[List[int]] = None):
        if sideband: 
            return f"$autopath.gigaq.pipelines.gigapath_backbone_evaluator({repr(name)}, sideband=True, capture_blocks={repr(capture_blocks)})"
        else:
            return f"$autopath.gigaq.pipelines.gigapath_backbone_evaluator({repr(name)})"

    if name == "GIGAPATH_BASELINE_SIDEBAND_CPTAC_9802_TEST":
        extractor=get_extractor('GIGAPATH_BASELINE_BACKBONE_EVALUATOR')
        tilebags="$autopath.pancan.pipelines.pancan_tile_fold('CPTAC_9802_TEST')"
    elif name == "GIGAPATH_BASELINE_CPTAC_9802_TEST":
        extractor=get_extractor('GIGAPATH_BASELINE_BACKBONE_EVALUATOR')
        tilebags="$autopath.pancan.pipelines.pancan_tile_fold('CPTAC_9802_TEST')"
    elif name == "GIGAPATH_BASELINE_CPTAC_8020_TEST":
        extractor=get_extractor('GIGAPATH_BASELINE_BACKBONE_EVALUATOR')
        tilebags="$autopath.pancan.pipelines.pancan_tile_fold('CPTAC_8020_TEST')"
    else:
        raise ValueError(f"Unknown feature bags: {name}")
    return FeatureBags(spec=dict(extractor=extractor, tilebags=tilebags))


# dbx "autopath.gigaq.pipelines.gigapath_feature_bags_probe('GIGAPATH_BASELINE_CPTAC_8020_TEST', n_bins=2).build()"
def gigapath_feature_bags_probe(name, n_bins: int = 2) -> FeatureBagsProbe:
    if name == "GIGAPATH_BASELINE_CPTAC_8020_TEST":
        return FeatureBagsProbe(
                    spec=dict(featurebags="$autopath.gigaq.pipelines.gigapath_feature_bags('GIGAPATH_BASELINE_CPTAC_8020_TEST')",
                              n_bins=n_bins,
        ))
    else:
        raise ValueError(f"Unknown feature bags probe: {name}")

# git commit -am "gigaq: FeaturesPairwiseDistances: BUILD"; dbx.print "autopath.gigaq.pipelines.gigapath_features_pairwise_distances('GIGAPATH_BASELINE_CPTAC_8020_TEST_10_4000_4000').set(n_devices=3).build()"
def gigapath_features_pairwise_distances(name) -> FeaturesPairwiseDistances:
    if name == "GIGAPATH_BASELINE_CPTAC_8020_TEST_10_4000_4000":
        return FeaturesPairwiseDistances(
                    spec=dict(featurebags=f"$autopath.gigaq.pipelines.gigapath_feature_bags('GIGAPATH_BASELINE_CPTAC_8020_TEST')",
                              n_chunks=10,
                              row_chunk_size=4000,
                              col_chunk_size=4000,
                    ), 
        )
    else:
        raise ValueError(f"Unknown feature bags pairwise distances datablock: {name}")


# git commit -am "gigaq: FeaturesSortedDistances: BUILD"; dbx.print "autopath.gigaq.pipelines.gigapath_features_sorted_distances('GIGAPATH_BASELINE_CPTAC_8020_TEST_4000_05_005',).set(n_workers=3, use_gpus=True).build()"
# git commit -am "gigaq: FeaturesSortedDistances: BUILD"; dbx "autopath.gigaq.pipelines.gigapath_features_sorted_distances('GIGAPATH_BASELINE_CPTAC_8020_TEST_1000_05_05',).set(n_workers=3, use_gpus=True).build()"
# git commit -am "gigaq: FeaturesSortedDistances: BUILD"; dbx "autopath.gigaq.pipelines.gigapath_features_sorted_distances('GIGAPATH_BASELINE_CPTAC_8020_TEST_1000_05_005',).set(n_workers=3, use_gpus=True).build()"
def gigapath_features_sorted_distances(name) -> FeaturesSortedDistances:
    if name == "GIGAPATH_BASELINE_CPTAC_8020_TEST_4000_05_005":
        return FeaturesSortedDistances(
                    spec=dict(features_pairwise_distances=f"$autopath.gigaq.pipelines.gigapath_features_pairwise_distances('GIGAPATH_BASELINE_CPTAC_8020_TEST_4000_05_05')",
                              col_subsample_fraction=0.1,
                    ),
        )
    elif name == "GIGAPATH_BASELINE_CPTAC_8020_TEST_1000_05_05":
        return FeaturesSortedDistances(
                    spec=dict(features_pairwise_distances=f"$autopath.gigaq.pipelines.gigapath_features_pairwise_distances('GIGAPATH_BASELINE_CPTAC_8020_TEST_1000_05_05')",
                              col_subsample_fraction=1.0,
                    ),
        )
    else:
        raise ValueError(f"Unknown feature sorted distances probe: {name}")
    

# dbx "autopath.gigaq.pipelines.gigapath_features_2nn_dim('GIGAPATH_BASELINE_CPTAC_8020_TEST_4000_05_005',).set(n_workers=3).build()"
def gigapath_features_2nn_distances(name) -> Features2NNDistances:
    if name == "GIGAPATH_BASELINE_CPTAC_8020_TEST_4000_05_005":
        return Features2NNDistances(
                    spec=dict(features_sorted_distances=f"$autopath.gigaq.pipelines.gigapath_features_sorted_distances('{name}')",),
        )
    else:
        raise ValueError(f"Unknown feature bags 2nn distances datablock: {name}")
    

# dbx "autopath.gigaq.pipelines.gigapath_features_2nn_dim('GIGAPATH_BASELINE_CPTAC_8020_TEST_4000_05_005',).build()"
def gigapath_features_2nn_dim(name) -> Features2NNDim:
    if name == "GIGAPATH_BASELINE_CPTAC_8020_TEST_4000_05_005":
        return Features2NNDim(
                    spec=dict(features_2nn_distances=f"$autopath.gigaq.pipelines.gigapath_features_2nn_distances('{name}')",),
        )
    else:
        raise ValueError(f"Unknown feature bags 2nn dim datablock: {name}")