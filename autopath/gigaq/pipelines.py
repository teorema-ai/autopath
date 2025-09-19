import os

from ..pancan.features import FeatureBag, FeatureBags
import dbx

from .dinov2.backbone import BackboneEvaluator


def gigapath_backbone_evaluator(name, *, device: str = 'cuda'):
    if name == "GIGAPATH_BASELINE_BACKBONE_EVALUATOR":
        return BackboneEvaluator(device=device)
    else:
        raise ValueError(f"Unknown backbone evaluator: {name}")

# dbx "autopath.gigaq.pipelines.gigapath_feature_bag('GIGAPATH_BASELINE_CPTAC_SAMPLE').build()"
def gigapath_feature_bag(name, *, device = 'cuda', gpu_batch_size=1024,) -> FeatureBag:
    if name == "GIGAPATH_BASELINE_CPTAC_SAMPLE":
        return FeatureBag(spec=dict(tilebag="@autopath.pancan.pipelines.pancan_tile_bag('CPTAC_SAMPLE')#",))
    else:
        raise ValueError(f"Unknown feature bag: {name}")


# dbx "autopath.gigaq.pipelines.gigapath_feature_bags('GIGAPATH_BASELINE_CPTAC_8020_TEST').build()"
def gigapath_feature_bags(name, *, devices = 'cuda', gpu_batch_size=1536) -> FeatureBags:
    if name == "GIGAPATH_BASELINE_CPTAC_9802_TEST":
        return FeatureBags(
                    devices=devices,
                    gpu_batch_size=gpu_batch_size,
                    spec=dict(extractor="@autopath.gigaq.pipelines.gigapath_backbone_evaluator('GIGAPATH_BASELINE_BACKBONE_EVALUATOR')#",
                              tilebags="@autopath.pancan.pipelines.pancan_tile_fold('CPTAC_9802_TEST')#",
        ))
    elif name == "GIGAPATH_BASELINE_CPTAC_8020_TEST":
        return FeatureBags(
                    devices=devices,
                    gpu_batch_size=gpu_batch_size,
                    spec=dict(extractor="@autopath.gigaq.pipelines.gigapath_backbone_evaluator('GIGAPATH_BASELINE_BACKBONE_EVALUATOR')#",
                              tilebags="@autopath.pancan.pipelines.pancan_tile_fold('CPTAC_8020_TEST')#",
        ))
    else:
        raise ValueError(f"Unknown feature bags: {name}")


