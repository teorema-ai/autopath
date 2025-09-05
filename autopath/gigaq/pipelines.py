import os

from ..pancan.features import FeatureBatch, FeatureSet
from .dinov2.backbone import BackboneEvaluator
from ..pancan.features import FeatureSet
from ..pancan.features import FeatureBatch # DEPRECATED


GIGAQ_DATASPACE = os.environ.get("GIGAQ_DATASPACE", "/mnt/labshare/PERSONAL/dmitry/dbx") 

GIGAPATH_BASELINE_BACKBONE_EVALUATOR = BackboneEvaluator


def GIGAPATH_BASELINE_CPTAC_8020_TEST_FEATURES(
    root=GIGAQ_DATASPACE,
    *,
    device = 'cuda',
    gpu_batch_size=128,
    shard_size: int = 1024,
    verbose=False,
    debug=False,
) -> FeatureSet:
    return FeatureSet(root,
                      device=device,
                      gpu_batch_size=gpu_batch_size,
                      verbose=verbose,
                      debug=debug,
                      spec=dict(extractor="@autopath.gigaq.pipelines.GIGAPATH_BASELINE_BACKBONE_EVALUATOR()",
                                tileset="@autopath.pancan.pipelines.PANCAN_CPTAC_8020_TEST()",
                                shard_size=shard_size,
    ))


# DEPRECATED BELOW THIS LINE


def GIGAPATH_BASELINE_CPTAC_FEATURES(
    *,
    num_gpus=None,
    gpu_batch_size=16,
    split='test',
    max_bag_count=None,
    verbose=True,
    debug=True,
):
    if num_gpus is not None:
        builder='@autopath.pancan.features.TorchMultiprocessingDatabatchBuilder(num_gpus={num_gpus})'
    else:
        builder='@dbx.DatabatchBuilder(verbose=True)'
    featurebatch = FeatureBatch(
                spec=dict(extractor="@autopath.gigaq.pipelines.GIGAPATH_BASELINE_BACKBONE_EVALUATOR()",
                         slidebatch="@autopath.pancan.pipelines.PANCAN_CPTAC_SLIDE_BATCH()", 
                         split=split,
                ),
                builder=builder,
                gpu_batch_size=gpu_batch_size, 
                verbose=verbose, 
                debug=debug, 
    )
    return featurebatch


def GIGAPATH_BASELINE_CPTAC_TEST_FEATURES(
    *,
    num_gpus=None,
    gpu_batch_size=16,
    split='test',
    max_bag_count=None,
    verbose=True,
    debug=False,
):
    return GIGAPATH_BASELINE_CPTAC_FEATURES(
        num_gpus=num_gpus,
        gpu_batch_size=gpu_batch_size,
        split='test',
        max_bag_count=None,
        verbose=verbose,
        debug=debug
    )
