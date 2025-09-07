import os

from ..pancan.features import FeatureBatch, FeatureShard
from .dinov2.backbone import BackboneEvaluator


GIGAPATH_DATASPACE = os.environ.get("GIGAPATH_DATASPACE", "/mnt/labshare/PERSONAL/dmitry/dbx") 


def gigapath_backbone_evaluator(name, *, device: str = 'cuda'):
    match name:
        case "GIGAPATH_BASELINE_BACKBONE_EVALUATOR":
            return BackboneEvaluator(device=device)
        case _:
            raise ValueError(f"Unknown backbone evaluator: {name}")


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
    match name:
        case "GIGAPATH_BASELINE_CPTAC_8020_TEST_FEATURES":
            return FeatureBatch(root,
                      device=device,
                      gpu_batch_size=gpu_batch_size,
                      verbose=verbose,
                      debug=debug,
                      spec=dict(extractor="@autopath.gigaq.pipelines.gigapath_backbone_evaluator('GIGAPATH_BASELINE_BACKBONE_EVALUATOR')#",
                                tileset="@autopath.pancan.pipelines.pancan_tileset('PANCAN_CPTAC_8020_TEST')#",
                                shard_size=shard_size,
            ))
        case "GIGAPATH_BASELINE_CPTAC_9802_TEST_FEATURES":
            return FeatureBatch(root,
                      device=device,
                      gpu_batch_size=gpu_batch_size,
                      verbose=verbose,
                      debug=debug,
                      spec=dict(extractor="@autopath.gigaq.pipelines.gigapath_backbone_evaluator('GIGAPATH_BASELINE_BACKBONE_EVALUATOR')#",
                                tileset="@autopath.pancan.pipelines.pancan_tileset('PANCAN_CPTAC_9802_TEST')#",
                                shard_size=shard_size,
            ))
        case _:
            raise ValueError(f"Unknown featureset: {name}")


