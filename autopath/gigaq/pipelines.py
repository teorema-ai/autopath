import os

from ..pancan.features import FeatureBag, FeatureBags
import dbx

from .dinov2.backbone import BackboneEvaluator


def gigapath_backbone_evaluator(name, *, device: str = 'cuda'):
    if name == "GIGAPATH_BASELINE_BACKBONE_EVALUATOR":
        return BackboneEvaluator(device=device)
    else:
        raise ValueError(f"Unknown backbone evaluator: {name}")


def gigapath_feature_bag(name, *, device = 'cuda', gpu_batch_size=512,) -> FeatureBag:
    if name == "GIGAPATH_BASELINE_CPTAC_SAMPLE":
        return FeatureBag(device=device,
                          gpu_batch_size=gpu_batch_size,
                          spec=dict(extractor="@autopath.gigaq.pipelines.gigapath_backbone_evaluator('GIGAPATH_BASELINE_BACKBONE_EVALUATOR')#",
                                    tilebag="@autopath.pancan.pipelines.pancan_tile_bag('PANCAN_CPTAC_SAMPLE')#",
        ))


def gigapath_feature_bags(name, *, device = 'cuda', gpu_batch_size=1024) -> FeatureBags:
    if name == "GIGAPATH_BASELINE_PANCAN_CPTAC_9802_TEST":
        return FeatureBags(
                    device=device,
                    gpu_batch_size=gpu_batch_size,
                    spec=dict(extractor="@autopath.gigaq.pipelines.gigapath_backbone_evaluator('GIGAPATH_BASELINE_BACKBONE_EVALUATOR')#",
                              tilebatch="@autopath.pancan.pipelines.pancan_tile_fold('PANCAN_CPTAC_9802_TEST')#",
        ))


