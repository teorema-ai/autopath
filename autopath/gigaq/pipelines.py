from ..pancan.features import FeatureBatch
from .dinov2.backbone import BackboneEvaluator


GIGAPATH_BASELINE_BACKBONE_EVALUATOR = BackboneEvaluator

def GIGAPATH_BASELINE_CPTAC_FEATURES_(
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
        builder='@dbx.DatabatchBuilder()'
    featurebatch = FeatureBatch(
                cfg=dict(extractor="@autopath.gigaq.pipelines.GIGAPATH_BASELINE_BACKBONE_EVALUATOR()",
                         slidebatch="@autopath.pancan.piplines.PANCAN_CPTAC_SLIDE_BATCH()", 
                         split=split,
                ),
                builder=builder,
                gpu_batch_size=gpu_batch_size, 
                verbose=verbose, 
                debug=debug, 
    )
    return featurebatch


def GIGAPATH_BASELINE_CPTAC_FEATURES_TEST(
    *,
    num_gpus=None,
    gpu_batch_size=16,
    split='test',
    max_bag_count=None,
    verbose=True,
    debug=False,
):
    return GIGAPATH_BASELINE_CPTAC_FEATURES_(
        num_gpus=num_gpus,
        gpu_batch_size=gpu_batch_size,
        split='test',
        max_bag_count=None,
        verbose=verbose,
        debug=debug
    )
