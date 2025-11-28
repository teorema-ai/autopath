import os
from typing import Optional, List

import torch
import torch.multiprocessing as mp

import dbx

from autopath.pancan.pipelines import (
    pancan_tile_bag,
    pancan_tile_fold,
)

from autopath.features import (
    FeatureShard, 
    FeatureClip,
    featureset,
)

from autopath.pancan.probes import (
    LogisticFeatureBagProbe, 
    #
    FeaturePairwiseDistances,
    FeatureSortedDistances,
    Feature2NNDistances,
    Feature2NNDim,
)

from autopath.gigaq.dinov2.backbone import (
    BackboneEvaluator, 
    SidebandBackboneEvaluator, 
    GIGAPATH_BACKBONE_DEPTH,
)


from autopath.models.vred import (
    Classifier, 
    ClassMultiscaleLatentGaussians2D,
    VariationalReDecoder,
    VariationalReDecoderEvaluator,
    VariationalReDecoderLightning,
    VariationalReDecoderStill,
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
    elif name == "GIGAPATH_BASELINE_CPTAC_9802_TRAIN":
        extractor=get_extractor('GIGAPATH_BASELINE_BACKBONE_EVALUATOR')
        tileclip=dbx.quote(pancan_tile_fold, 'CPTAC_9802_TRAIN')
    elif name == "GIGAPATH_BASELINE_CPTAC_8020_TEST":
        extractor=get_extractor('GIGAPATH_BASELINE_BACKBONE_EVALUATOR')
        tileclip=dbx.quote(pancan_tile_fold, 'CPTAC_8020_TEST')
    elif name == "GIGAPATH_BASELINE_CPTAC_8020_TRAIN":
        extractor=get_extractor('GIGAPATH_BASELINE_BACKBONE_EVALUATOR')
        tileclip=dbx.quote(pancan_tile_fold, 'CPTAC_8020_TRAIN')
    elif name == "GIGAPATH_BASELINE_5B_CPTAC_8020_TEST":
        extractor=get_extractor('GIGAPATH_BASELINE_BACKBONE_5B_EVALUATOR')
        tileclip = dbx.quote(pancan_tile_fold, 'CPTAC_8020_TEST')
    else:
        raise ValueError(f"Unknown gigapath_feature_clip: {repr(name)}")
    return FeatureClip(spec=dict(extractor=extractor, tileclip=tileclip))


# git commit -am "gigaq: Featureset: TEST"; dbx.print "autopath.gigaq.pipelines.gigapath_featureset('GIGAPATH_BASELINE_CPTAC_8020_TEST')[0]"
# git commit -am "gigaq: Featureset: TEST"; dbx.print "autopath.gigaq.pipelines.gigapath_featureset('GIGAPATH_BASELINE_CPTAC_9802_TEST')[0]"
def gigapath_featureset(name) -> torch.utils.data.Dataset:
    featureclip = gigapath_feature_clip(name)
    quoted_featureclip = dbx.quote(featureclip)
    dbx.Logger().debug(f"===================> {featureclip=}\n{quoted_featureclip=}")
    return featureset(quoted_featureclip)


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


def gigapath_featureset_dataloader(name, *args, **kwargs):
    featureset = gigapath_featureset(name)
    return torch.utils.data.DataLoader(featureset, *args, **kwargs)


def gigapath_featureset_dataloader_sample(name, *args, **kwargs):
    featureset = gigapath_featureset(name)
    dataloader = torch.utils.data.DataLoader(featureset, *args, **kwargs)
    return next(iter(dataloader))


# git commit -am "gigaq: VRED"; dbx.print "autopath.gigaq.pipelines.gigapath_vred('GIGAPATH_VRED_2HDN_100CLS_5CHN')"
def gigapath_vred(name):
    if name == "GIGAPATH_VRED_2HDN_100CLS_5CHN":
        n_hidden_layers=2
        n_classes=100
        n_channels=5
    else:
        raise ValueError(f"Unknown gigapath_vred: {name}")
    
    input_dim = 1536
    classifier = Classifier(
        input_dim=input_dim,
        n_hidden_layers=n_hidden_layers,
        n_classes=n_classes,
    )
    latent_gaussians = ClassMultiscaleLatentGaussians2D(
        n_classes=n_classes,
        n_channels=n_channels,
        input_dim=input_dim,
        n_hidden_layers=n_hidden_layers,
        fine_scale=256,
        n_scales=4,
    )
    vred = VariationalReDecoder(
        classifier=classifier,
        latent_gaussians=latent_gaussians,
    )
    return vred
    
# git commit -am "gigaq: VRED EVAL"; dbx.print "autopath.gigaq.pipelines.gigapath_vred_evaluator('GIGAPATH_VRED_2HDN_100CLS_5CHN_BASELINE_CPTAC_8020_TEST', batch_size=1).to('cuda').samples(1)"
# git commit -am "gigaq: VRED EVAL"; dbx.print "autopath.gigaq.pipelines.gigapath_vred_evaluator('GIGAPATH_VRED_2HDN_100CLS_5CHN_BASELINE_CPTAC_8020_TEST', batch_size=1).to('cuda').samples(2)"
# git commit -am "gigaq: VRED EVAL"; dbx.print "autopath.gigaq.pipelines.gigapath_vred_evaluator('GIGAPATH_VRED_2HDN_100CLS_5CHN_BASELINE_CPTAC_8020_TEST', batch_size=2).to('cuda').samples(1)"
# git commit -am "gigaq: VRED EVAL"; dbx.print "autopath.gigaq.pipelines.gigapath_vred_evaluator('GIGAPATH_VRED_2HDN_100CLS_5CHN_BASELINE_CPTAC_8020_TEST', batch_size=2).to('cuda').samples(2)"
#
# git commit -am "gigaq: VRED EVAL"; dbx.print "autopath.gigaq.pipelines.gigapath_vred_evaluator('GIGAPATH_VRED_2HDN_100CLS_5CHN_BASELINE_CPTAC_8020_TEST', batch_size=1).to('cuda').losses(1)"
# git commit -am "gigaq: VRED EVAL"; dbx.print "autopath.gigaq.pipelines.gigapath_vred_evaluator('GIGAPATH_VRED_2HDN_100CLS_5CHN_BASELINE_CPTAC_8020_TEST', batch_size=1).to('cuda').losses(2)"
# git commit -am "gigaq: VRED EVAL"; dbx.print "autopath.gigaq.pipelines.gigapath_vred_evaluator('GIGAPATH_VRED_2HDN_100CLS_5CHN_BASELINE_CPTAC_8020_TEST', batch_size=2).to('cuda').losses(1)"
# git commit -am "gigaq: VRED EVAL"; dbx.print "autopath.gigaq.pipelines.gigapath_vred_evaluator('GIGAPATH_VRED_2HDN_100CLS_5CHN_BASELINE_CPTAC_8020_TEST', batch_size=2).to('cuda').losses(2)"
def gigapath_vred_evaluator(vred_dataset_name, **dataloader_kwargs):
    vredname, _clipname = vred_dataset_name.split('_BASELINE_CPTAC_')
    clipname = "GIGAPATH_BASELINE_CPTAC_" + _clipname
    vred = gigapath_vred(vredname)
    featureloader = gigapath_featureset_dataloader(clipname, **dataloader_kwargs)
    vred_evaluator = VariationalReDecoderEvaluator(vred, featureloader)
    return vred_evaluator

#TODO: #REMOVE?
def gigapath_vred_lightning(vred_name, learning_rate: float = 0.03):
    return VariationalReDecoderLightning(spec=dict(vred=dbx.quote(gigapath_vred, vred_name), learning_rate=learning_rate))

# git commit -am "gigaq: VRED EVAL"; dbx.print "autopath.gigaq.pipelines.gigapath_vred_still('GIGAPATH_VRED_2HDN_100CLS_5CHN_BASELINE_CPTAC_8020_TEST').to('cuda')"
def gigapath_vred_still(vred_dataset_name, *, learning_rate: float = 0.03, max_steps: int = None, init_ckpt_path: str = None, n_devices: int = 1, batch_size: int = 1, shuffle: bool = False,):
    evaluator = gigapath_vred_evaluator(vred_dataset_name, batch_size=batch_size, shuffle=shuffle)
    lightning = VariationalReDecoderLightning(spec=dict(vred=evaluator.spec.vred, learning_rate=learning_rate))
    still = VariationalReDecoderStill(spec=dict(
        lightning=lightning, 
        dataloader=evaluator.spec.dataloader,
        max_steps=max_steps,
        init_ckpt_path=init_ckpt_path,
        ),
        n_devices=n_devices,
    )
    return still
    
