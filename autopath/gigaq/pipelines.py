import os

from ..pancan.features import FeatureShard, FeatureShards
import dbx

from .dinov2.backbone import BackboneEvaluator


def gigapath_backbone_evaluator(name, *, device: str = 'cuda'):
    if name == "GIGAPATH_BASELINE_BACKBONE_EVALUATOR":
        return BackboneEvaluator(device=device)
    else:
        raise ValueError(f"Unknown backbone evaluator: {name}")

# dbx "autopath.gigaq.pipelines.gigapath_feature_shard('GIGAPATH_BASELINE_CPTAC_SAMPLE').build()"
def gigapath_feature_shard(name, *, device = 'cuda', gpu_batch_size=2014,) -> FeatureShard:
    if name == "GIGAPATH_BASELINE_CPTAC_SAMPLE":
        return FeatureShard(spec=dict(tileshard="@autopath.pancan.pipelines.pancan_tile_bag('CPTAC_SAMPLE')#",))
    else:
        raise ValueError(f"Unknown feature shard: {name}")


# dbx "autopath.gigaq.pipelines.gigapath_feature_shards('GIGAPATH_BASELINE_CPTAC_8020_TEST')"
def gigapath_feature_shards(name, *, device = 'cuda', gpu_batch_size=512) -> FeatureShards:
    if name == "GIGAPATH_BASELINE_CPTAC_9802_TEST":
        return FeatureShards(
                    device=device,
                    gpu_batch_size=gpu_batch_size,
                    spec=dict(extractor="@autopath.gigaq.pipelines.gigapath_backbone_evaluator('GIGAPATH_BASELINE_BACKBONE_EVALUATOR')#",
                              tileshards="@autopath.pancan.pipelines.pancan_tile_fold('CPTAC_9802_TEST')#",
        ))
    elif name == "GIGAPATH_BASELINE_CPTAC_8020_TEST":
        return FeatureShards(
                    device=device,
                    gpu_batch_size=gpu_batch_size,
                    spec=dict(extractor="@autopath.gigaq.pipelines.gigapath_backbone_evaluator('GIGAPATH_BASELINE_BACKBONE_EVALUATOR')#",
                              tileshards="@autopath.pancan.pipelines.pancan_tile_fold('CPTAC_8020_TEST')#",
        ))
    else:
        raise ValueError(f"Unknown feature shards: {name}")


