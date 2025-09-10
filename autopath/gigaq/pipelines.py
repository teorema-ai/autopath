import os

from ..pancan.features import FeatureBag
from .dinov2.backbone import BackboneEvaluator


GIGAPATH_DATASPACE = os.environ.get("GIGAPATH_DATASPACE", "/mnt/labshare/PERSONAL/dmitry/dbx") 


def gigapath_backbone_evaluator(name, *, device: str = 'cuda'):
        if name == "GIGAPATH_BASELINE_BACKBONE_EVALUATOR":
            return BackboneEvaluator(device=device)
        else:
            raise ValueError(f"Unknown backbone evaluator: {name}")


def gigapath_feature_bag(name, *, root=GIGAPATH_DATASPACE, device = 'cuda', gpu_batch_size=1024, verbose=False, debug=False,
) -> FeatureBag:
        if name == "GIGAPATH_BASELINE_CPTAC_SAMPLE":
            return FeatureBag(root,
                      device=device,
                      gpu_batch_size=gpu_batch_size,
                      verbose=verbose,
                      debug=debug,
                      spec=dict(extractor="@autopath.gigaq.pipelines.gigapath_backbone_evaluator('GIGAPATH_BASELINE_BACKBONE_EVALUATOR')#",
                                tilebag="@autopath.pancan.pipelines.pancan_tilebag('PANCAN_CPTAC_SAMPLE')#",
            ))

def gigapath_feature_bags(name, *, root=GIGAPATH_DATASPACE, device = 'cuda', gpu_batch_size=1024, verbose=False, debug=False,
) -> FeatureBags:
        if name == "GIGAPATH_BASELINE_PANCAN_CPTAC":
            return FeatureBags(root,
                      device=device,
                      gpu_batch_size=gpu_batch_size,
                      verbose=verbose,
                      debug=debug,
                      spec=dict(extractor="@autopath.gigaq.pipelines.gigapath_backbone_evaluator('GIGAPATH_BASELINE_BACKBONE_EVALUATOR')#",
                                tilebags="@autopath.pancan.pipelines.pancan_tilebags('PANCAN_CPTAC_9802_TEST')#",
            ))
"""
def gigapath_features(
    name,
    *,
    root=GIGAPATH_DATASPACE,
    device = 'cuda',
    gpu_batch_size=1024,
    shard_size: int = 1024,
    verbose=False,
    debug=False,
) -> FeatureBatch:
        if name == "GIGAPATH_BASELINE_CPTAC_8020_TEST_FEATURES":
            return FeatureBatch(root,
                      device=device,
                      gpu_batch_size=gpu_batch_size,
                      verbose=verbose,
                      debug=debug,
                      spec=dict(extractor="@autopath.gigaq.pipelines.gigapath_backbone_evaluator('GIGAPATH_BASELINE_BACKBONE_EVALUATOR')#",
                                tileset="@autopath.pancan.pipelines.pancan_tileset('PANCAN_CPTAC_8020_TEST')#",
                                shard_size=shard_size,
            ))

def gigapath_features(
    name,
    *,
    root=GIGAPATH_DATASPACE,
    device = 'cuda',
    gpu_batch_size=1024,
    shard_size: int = 1024,
    verbose=False,
    debug=False,
) -> FeatureBatch:
        if name == "GIGAPATH_BASELINE_CPTAC_8020_TEST_FEATURES":
            return FeatureBatch(root,
                      device=device,
                      gpu_batch_size=gpu_batch_size,
                      verbose=verbose,
                      debug=debug,
                      spec=dict(extractor="@autopath.gigaq.pipelines.gigapath_backbone_evaluator('GIGAPATH_BASELINE_BACKBONE_EVALUATOR')#",
                                tileset="@autopath.pancan.pipelines.pancan_tileset('PANCAN_CPTAC_8020_TEST')#",
                                shard_size=shard_size,
            ))
        elif name == "GIGAPATH_BASELINE_CPTAC_9802_TEST_FEATURES":
            return FeatureBatch(root,
                      device=device,
                      gpu_batch_size=gpu_batch_size,
                      verbose=verbose,
                      debug=debug,
                      spec=dict(extractor="@autopath.gigaq.pipelines.gigapath_backbone_evaluator('GIGAPATH_BASELINE_BACKBONE_EVALUATOR')#",
                                tileset="@autopath.pancan.pipelines.pancan_tileset('PANCAN_CPTAC_9802_TEST')#",
                                shard_size=shard_size,
            ))
        else:
            raise ValueError(f"Unknown featureset: {name}")
"""

