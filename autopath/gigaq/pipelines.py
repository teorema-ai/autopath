import os
from typing import Optional, List

import torch.multiprocessing as mp

import dbx

from autopath.pancan.pipelines import (
    pancan_tile_bag,
    pancan_tile_fold,
)

from autopath.features import (
    FeatureShard, 
    FeatureClip,
)

from autopath.pancan.probes import (
    LogisticFeatureBagProbe, 
    #
    FeaturePairwiseDistances,
    FeatureSortedDistances,
    Feature2NNDistances,
    Feature2NNDim,
)

from .dinov2.backbone import (
    BackboneEvaluator, 
    SidebandBackboneEvaluator, 
    GIGAPATH_BACKBONE_DEPTH,
)


mp.set_start_method("spawn", force=True)


def gigapath_backbone_evaluator(name, *, device: str = 'cuda',):
    def select_capture_blocks(n_blocks: int = 1):
        if n_blocks <= 0 or n_blocks > GIGAPATH_BACKBONE_DEPTH:
            return None
        inc = GIGAPATH_BACKBONE_DEPTH // (n_blocks - 1)
        return list(range(0, GIGAPATH_BACKBONE_DEPTH, inc))
    
    if name == "GIGAPATH_BASELINE_BACKBONE_EVALUATOR":
        return BackboneEvaluator(device=device)
    elif name == "GIGAPATH_BASELINE_BACKBONE_5BLOCK_EVALUATOR":
        capture_blocks = select_capture_blocks(n_blocks=5)
        return SidebandBackboneEvaluator(device=device, capture_blocks=capture_blocks)
    else:
        raise ValueError(f"Unknown backbone evaluator: {name}")


# git commit -am "gigaq: FeatureShard: BUILD"; dbx.print "autopath.gigaq.pipelines.gigapath_feature_shard('GIGAPATH_BASELINE_CPTAC_SAMPLE').set(device='cuda', gpu_batch_size=1024).build()"
def gigapath_feature_shard(name) -> FeatureShard:
    if name == "GIGAPATH_BASELINE_CPTAC_SAMPLE":
        return FeatureShard(spec=dict(tilebag=dbx.quote(pancan_tile_bag, 'CPTAC_SAMPLE'),))
    else:
        raise ValueError(f"Unknown feature shard: {name}")


# git commit -am "gigaq: FeatureClip: BUILD"; dbx.print "autopath.gigaq.pipelines.gigapath_feature_clip('GIGAPATH_BASELINE_CPTAC_8020_TEST').build()"
def gigapath_feature_clip(name) -> FeatureClip:
    def get_extractor(name, sideband: bool = False, capture_blocks: Optional[List[int]] = None):
        if sideband: 
            return dbx.quote(gigapath_backbone_evaluator, name, sideband=True, capture_blocks=capture_blocks)
        else:
            return dbx.quote(gigapath_backbone_evaluator, name)

    if name == "GIGAPATH_BASELINE_CPTAC_9802_TEST":
        extractor=get_extractor('GIGAPATH_BASELINE_BACKBONE_EVALUATOR')
        tileclip=dbx.quote(pancan_tile_fold, 'CPTAC_9802_TEST')
    elif name == "GIGAPATH_BASELINE_CPTAC_8020_TEST":
        extractor=get_extractor('GIGAPATH_BASELINE_BACKBONE_EVALUATOR')
        tileclip=dbx.quote(pancan_tile_fold, 'CPTAC_8020_TEST')
    elif name == "GIGAPATH_BASELINE_5B_CPTAC_8020_TEST":
        extractor=get_extractor('GIGAPATH_BASELINE_BACKBONE_5B_EVALUATOR')
        tileclip = dbx.quote(pancan_tile_fold, 'CPTAC_8020_TEST')
    else:
        raise ValueError(f"Unknown feature bags: {name}")
    return FeatureClip(spec=dict(extractor=extractor, tileclip=tileclip))


# git commit -am "gigaq: LogisticFeatureBagProbe: BUILD"; dbx "autopath.gigaq.pipelines.gigapath_logisticfeature_bags_probe('GIGAPATH_BASELINE_CPTAC_8020_TEST', n_bins=2).build()"
def gigapath_logistic_feature_bags_probe(name, n_bins: int = 2) -> LogisticFeatureBagProbe:
    if name == "GIGAPATH_BASELINE_CPTAC_8020_TEST":
        return LogisticFeatureBagProbe(
                    spec=dict(featurebags=dbx.quote(gigapath_feature_clip, 'GIGAPATH_BASELINE_CPTAC_8020_TEST'), n_bins=n_bins,))
    else:
        raise ValueError(f"Unknown feature bags probe: {name}")

# git commit -am "gigaq: FeaturePairwiseDistances: BUILD"; dbx.print "autopath.gigaq.pipelines.gigapath_feature_pairwise_distances('GIGAPATH_BASELINE_CPTAC_8020_TEST_10_4000_4000').set(n_devices=3).build()"
# git commit -am "gigaq: FeaturePairwiseDistances: BUILD"; dbx.print "autopath.gigaq.pipelines.gigapath_feature_pairwise_distances('GIGAPATH_BASELINE_CPTAC_8020_TEST_10_4000_20000').set(n_devices=3).build()"
# git commit -am "gigaq: FeaturePairwiseDistances: BUILD"; dbx.print "autopath.gigaq.pipelines.gigapath_feature_pairwise_distances('GIGAPATH_BASELINE_CPTAC_8020_TEST_20_4000_20000').set(n_devices=3).build()"
def gigapath_feature_pairwise_distances(name) -> FeaturePairwiseDistances:
    if name == "GIGAPATH_BASELINE_CPTAC_8020_TEST_10_4000_4000":
        return FeaturePairwiseDistances(
                    spec=dict(featurebags=dbx.quote(gigapath_feature_clip, 'GIGAPATH_BASELINE_CPTAC_8020_TEST'),
                              n_chunks=10,
                              row_chunk_size=4000,
                              col_chunk_size=4000,
                    ), 
        )
    elif name == "GIGAPATH_BASELINE_CPTAC_8020_TEST_10_4000_20000":
        return FeaturePairwiseDistances(
                    spec=dict(featurebags=dbx.quote(gigapath_feature_clip, 'GIGAPATH_BASELINE_CPTAC_8020_TEST'),
                              n_chunks=10,
                              row_chunk_size=4000,
                              col_chunk_size=20000,
                    ), 
        )
    elif name == "GIGAPATH_BASELINE_CPTAC_8020_TEST_20_4000_20000":
        return FeaturePairwiseDistances(
                    spec=dict(featurebags=dbx.quote(gigapath_feature_clip, 'GIGAPATH_BASELINE_CPTAC_8020_TEST'),
                              n_chunks=20,
                              row_chunk_size=4000,
                              col_chunk_size=20000,
                    ), 
        )
    elif name == "GIGAPATH_BASELINE_CPTAC_8020_TEST_5_4000_20000":
        return FeaturePairwiseDistances(
                    spec=dict(featurebags=dbx.quote(gigapath_feature_clip, 'GIGAPATH_BASELINE_CPTAC_8020_TEST'),
                              n_chunks=5,
                              row_chunk_size=4000,
                              col_chunk_size=20000,
                    ), 
        )
    elif name == "GIGAPATH_BASELINE_CPTAC_8020_TEST_5_4000_4000":
        return FeaturePairwiseDistances(
                    spec=dict(featurebags=dbx.quote(gigapath_feature_clip, 'GIGAPATH_BASELINE_CPTAC_8020_TEST'),
                              n_chunks=5,
                              row_chunk_size=4000,
                              col_chunk_size=4000,
                    ), 
        )
    elif name == "GIGAPATH_BASELINE_CPTAC_8020_TEST_5_20000_4000":
        return FeaturePairwiseDistances(
                    spec=dict(featurebags=dbx.quote(gigapath_feature_clip, 'GIGAPATH_BASELINE_CPTAC_8020_TEST'),
                              n_chunks=5,
                              row_chunk_size=20000,
                              col_chunk_size=4000,
                    ), 
        )
    elif name == "GIGAPATH_BASELINE_CPTAC_8020_TEST_5_1000_5000":
        return FeaturePairwiseDistances(
                    spec=dict(featurebags=dbx.quote(gigapath_feature_clip, 'GIGAPATH_BASELINE_CPTAC_8020_TEST'),
                              n_chunks=5,
                              row_chunk_size=1000,
                              col_chunk_size=5000,
                    ), 
        )
    elif name == "GIGAPATH_BASELINE_CPTAC_8020_TEST_5_1000_10000":
        return FeaturePairwiseDistances(
                    spec=dict(featurebags=dbx.quote(gigapath_feature_clip, 'GIGAPATH_BASELINE_CPTAC_8020_TEST'),
                              n_chunks=5,
                              row_chunk_size=1000,
                              col_chunk_size=10000,
                    ), 
        )
    elif name == "GIGAPATH_BASELINE_CPTAC_8020_TEST_5_1000_20000":
        return FeaturePairwiseDistances(
                    spec=dict(featurebags=dbx.quote(gigapath_feature_clip, 'GIGAPATH_BASELINE_CPTAC_8020_TEST'),
                              n_chunks=5,
                              row_chunk_size=1000,
                              col_chunk_size=20000,
                    ), 
        )
    elif name == "GIGAPATH_BASELINE_CPTAC_8020_TEST_5_1000_50000":
        return FeaturePairwiseDistances(
                    spec=dict(featurebags=dbx.quote(gigapath_feature_clip, 'GIGAPATH_BASELINE_CPTAC_8020_TEST'),
                              n_chunks=5,
                              row_chunk_size=1000,
                              col_chunk_size=50000,
                    ), 
        )
    elif name == "GIGAPATH_BASELINE_CPTAC_8020_TEST_5_1000_100000":
        return FeaturePairwiseDistances(
                    spec=dict(featurebags=dbx.quote(gigapath_feature_clip, 'GIGAPATH_BASELINE_CPTAC_8020_TEST'),
                              n_chunks=5,
                              row_chunk_size=1000,
                              col_chunk_size=100000,
                    ), 
        )
    elif name == "GIGAPATH_BASELINE_CPTAC_8020_TEST_5_1000_200000":
        return FeaturePairwiseDistances(
                    spec=dict(featurebags=dbx.quote(gigapath_feature_clip, 'GIGAPATH_BASELINE_CPTAC_8020_TEST'),
                              n_chunks=5,
                              row_chunk_size=1000,
                              col_chunk_size=200000,
                    ), 
        )
    elif name == "GIGAPATH_BASELINE_CPTAC_8020_TEST_5_1000_400000":
        return FeaturePairwiseDistances(
                    spec=dict(featurebags=dbx.quote(gigapath_feature_clip, 'GIGAPATH_BASELINE_CPTAC_8020_TEST'),
                              n_chunks=5,
                              row_chunk_size=1000,
                              col_chunk_size=400000,
                    ), 
        )
    elif name == "GIGAPATH_BASELINE_CPTAC_8020_TEST_50_100_800000":
        return FeaturePairwiseDistances(
                    spec=dict(featurebags=dbx.quote(gigapath_feature_clip, 'GIGAPATH_BASELINE_CPTAC_8020_TEST'),
                              n_chunks=50,
                              row_chunk_size=100,
                              col_chunk_size=800000,
                    ), 
        )
    elif name == "GIGAPATH_BASELINE_CPTAC_8020_TEST_5_1000_150000":
        return FeaturePairwiseDistances(
                    spec=dict(featurebags=dbx.quote(gigapath_feature_clip, 'GIGAPATH_BASELINE_CPTAC_8020_TEST'),
                              n_chunks=5,
                              row_chunk_size=1000,
                              col_chunk_size=150000,
                    ), 
        )
    else:
        raise ValueError(f"Unknown feature bags pairwise distances datablock: {name}")


# git commit -am "gigaq: FeatureSortedDistances: BUILD"; dbx.print "autopath.gigaq.pipelines.gigapath_feature_sorted_distances('GIGAPATH_BASELINE_CPTAC_8020_TEST_10_4000_4000',).set(n_workers=3,).build()"
# git commit -am "gigaq: FeatureSortedDistances: BUILD"; dbx.print "autopath.gigaq.pipelines.gigapath_feature_sorted_distances('GIGAPATH_BASELINE_CPTAC_8020_TEST_10_4000_20000',).set(n_workers=3,).build()"
def gigapath_feature_sorted_distances(name) -> FeatureSortedDistances:
    return FeatureSortedDistances(
                    spec=dict(features_pairwise_distances=dbx.quote(gigapath_feature_pairwise_distances, name),)
    )
    

# git commit -am "gigaq: Features=2NNDistances: BUILD"; dbx.print "autopath.gigaq.pipelines.gigapath_feature_2nn_distances('GIGAPATH_BASELINE_CPTAC_8020_TEST_10_4000_4000',).set(n_workers=3).build()"
# git commit -am "gigaq: Feature2NNDistances: BUILD"; dbx.print "autopath.gigaq.pipelines.gigapath_feature_2nn_distances('GIGAPATH_BASELINE_CPTAC_8020_TEST_10_4000_20000',).set(n_workers=3).build()"
def gigapath_feature_2nn_distances(name) -> Feature2NNDistances:
    return Feature2NNDistances(
                    spec=dict(features_sorted_distances=dbx.quote(gigapath_feature_sorted_distances, name),),
        )
    

# git commit -am "gigaq: Feature2NNDim: BUILD"; dbx.print "autopath.gigaq.pipelines.gigapath_feature_2nn_dim('GIGAPATH_BASELINE_CPTAC_8020_TEST_10_4000_4000',).build_tree().dim"
# git commit -am "gigaq: Feature2NNDim: BUILD"; dbx.print "autopath.gigaq.pipelines.gigapath_feature_2nn_dim('GIGAPATH_BASELINE_CPTAC_8020_TEST_10_4000_20000',).build_tree().dim"
def gigapath_feature_2nn_dim(name) -> Feature2NNDim:
    return Feature2NNDim(
                    spec=dict(features_2nn_distances=dbx.quote(gigapath_feature_2nn_distances, name),),
        )