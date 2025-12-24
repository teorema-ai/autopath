from typing import Optional, List

from tqdm import tqdm

import torch
import torch.multiprocessing as mp

import dbx

from autopath.databits import ClipDataLoaderBuilder

from autopath.pancan.pipelines import (
    pancan_tile_bag,
    pancan_tile_bag_clip,
    pancan_tile_bag_fold,
)

from autopath.features import (
    FeatureBag, 
    FeatureBagClip,
    featurebagset,
    FeatureShardClip,
    featureshardset,
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
    VariationalReEncoderDecoder,
    VariationalReEncoderDecoderEvaluator,
    VariationalReEncoderDecoderLightning,
    VariationalReEncoderDecoderStill,
)


mp.set_start_method("spawn", force=True)

def quote_extractor(name, sideband: bool = False, capture_blocks: Optional[List[int]] = None):
        if sideband: 
            return dbx.quote(gigapath_backbone_evaluator, name, sideband=True, capture_blocks=capture_blocks)
        else:
            return dbx.quote(gigapath_backbone_evaluator, name)
        

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

# git commit -am "gigaq: FeatureBag: BUILD"; dbx.print "autopath.gigaq.pipelines.gigapath_feature_bag('GIGAPATH_BASELINE_CPTAC_SAMPLE').set(device='cuda', gpu_batch_size=1024).build()"
def gigapath_feature_bag(name: str = None, *, root: str = None) -> FeatureBag:
    if name is None:
        return FeatureBag
    elif name == "GIGAPATH_BASELINE_CPTAC_SAMPLE":
        return FeatureBag(
            root,
            spec=dict(
                tilebag=dbx.quote(pancan_tile_bag, 'CPTAC_SAMPLE'),
                extractor=quote_extractor('GIGAPATH_BASELINE_BACKBONE_EVALUATOR'),
        ))
    else:
        raise ValueError(f"Unknown feature shard: {name}")

# git commit -am "gigaq: FeatureBagClip: BUILD"; dbx.print "autopath.gigaq.pipelines.gigapath_feature_bag_clip('GIGAPATH_BASELINE_CPTAC', n_devices=3, n_threads=16).set(gpu_batch_size=1024).build()"
# git commit -am "gigaq: FeatureBagClip: BUILD"; dbx.print "autopath.gigaq.pipelines.gigapath_feature_bag_clip('GIGAPATH_BASELINE_CPTAC_9802_TEST', n_threads=16).build()"
# git commit -am "gigaq: FeatureBagClip: BUILD"; dbx.print "autopath.gigaq.pipelines.gigapath_feature_bag_clip('GIGAPATH_BASELINE_CPTAC_9802_TRAIN', n_threads=16).build()"
# git commit -am "gigaq: FeatureBagClip: BUILD"; dbx.print "autopath.gigaq.pipelines.gigapath_feature_bag_clip('GIGAPATH_BASELINE_CPTAC_8020_TEST', n_threads=16).build()"
# git commit -am "gigaq: FeatureBagClip: BUILD"; dbx.print "autopath.gigaq.pipelines.gigapath_feature_bag_clip('GIGAPATH_BASELINE_CPTAC_8020_TRAIN', n_threads=16).build()"
def gigapath_feature_bag_clip(name:str = None, *, root:str = None, n_devices: int = 1, n_threads: int = 1) -> FeatureBagClip:
    devices = [f'cuda:{i}' for i in range(n_devices)]
    if name is None:
        return FeatureBagClip
    if name == "GIGAPATH_BASELINE_CPTAC":
        extractor=quote_extractor('GIGAPATH_BASELINE_BACKBONE_EVALUATOR')
        tilebagclip=dbx.quote(pancan_tile_bag_clip, 'CPTAC')
    elif name == "GIGAPATH_BASELINE_CPTAC_9802_TEST":
        extractor=quote_extractor('GIGAPATH_BASELINE_BACKBONE_EVALUATOR')
        tilebagclip=dbx.quote(pancan_tile_bag_fold, 'CPTAC_9802_TEST')
    elif name == "GIGAPATH_BASELINE_CPTAC_9802_TRAIN":
        extractor=quote_extractor('GIGAPATH_BASELINE_BACKBONE_EVALUATOR')
        tilebagclip=dbx.quote(pancan_tile_bag_fold, 'CPTAC_9802_TRAIN')
    elif name == "GIGAPATH_BASELINE_CPTAC_8020_TEST":
        extractor=quote_extractor('GIGAPATH_BASELINE_BACKBONE_EVALUATOR')
        tilebagclip=dbx.quote(pancan_tile_bag_fold, 'CPTAC_8020_TEST')
    elif name == "GIGAPATH_BASELINE_CPTAC_8020_TRAIN":
        extractor=quote_extractor('GIGAPATH_BASELINE_BACKBONE_EVALUATOR')
        tilebagclip=dbx.quote(pancan_tile_bag_fold, 'CPTAC_8020_TRAIN')
    elif name == "GIGAPATH_BASELINE_5B_CPTAC_8020_TEST":
        extractor=quote_extractor('GIGAPATH_BASELINE_BACKBONE_5B_EVALUATOR')
        tilebagclip = dbx.quote(pancan_tile_bag_fold, 'CPTAC_8020_TEST')
    else:
        raise ValueError(f"Unknown gigapath_feature_clip: {repr(name)}")
    return FeatureBagClip(root, spec=dict(extractor=extractor, tilebagclip=tilebagclip), devices=devices, n_threads=n_threads)


# git commit -am "gigaq: Featurebagset: TEST"; dbx.print "autopath.gigaq.pipelines.gigapath_featurebagset('GIGAPATH_BASELINE_CPTAC')[0]"
# git commit -am "gigaq: Featurebagset: TEST"; dbx.print "autopath.gigaq.pipelines.gigapath_featurebagset('GIGAPATH_BASELINE_CPTAC_8020_TEST')[0]"
# git commit -am "gigaq: Featurebagset: TEST"; dbx.print "autopath.gigaq.pipelines.gigapath_featurebagset('GIGAPATH_BASELINE_CPTAC_9802_TEST')[0]"
def gigapath_featurebagset(name, *, root: str = None, shuffle_bags_seed: int = None) -> torch.utils.data.Dataset:
    featureclip = gigapath_feature_bag_clip(name, root=root)
    quoted_featureclip = dbx.quote(featureclip)
    dbx.Logger().debug(f"===================> {featureclip=}\n{quoted_featureclip=}")
    return featurebagset(quoted_featureclip, bags_shuffle_seed=shuffle_bags_seed)


# git commit -am "gigaq: FeatureShardClip: BUILD"; dbx.print "autopath.gigaq.pipelines.gigapath_feature_shard_clip('GIGAPATH_BASELINE_CPTAC', shard_size=32, n_threads=16).build()"
def gigapath_feature_shard_clip(name: str = None, *, shard_size: int = 32, n_threads: int = 1) -> FeatureShardClip:
    if name is None:
        clip = FeatureShardClip
    else:
        try:
            featureshardset=dbx.quote(gigapath_featureshardset, name)
            clip = FeatureShardClip(spec=dict(featureset=featureshardset, shard_size=shard_size), n_threads=n_threads)
        except Exception as e:
            raise ValueError(f"Failed to instantiate gigapath_feature_shard_clip: {repr(name)}") from e
    return clip


# git commit -am "gigaq: Featureshardset: TEST"; dbx.print "autopath.gigaq.pipelines.gigapath_featureshardset('GIGAPATH_BASELINE_CPTAC_32')[0]"
def gigapath_featureshardset(name) -> torch.utils.data.Dataset:
    featureshardclip = gigapath_feature_shard_clip(name)
    quoted_featureshardclip = dbx.quote(featureshardclip)
    dbx.Logger().debug(f"===================> {featureshardclip=}\n{quoted_featureshardclip=}")
    return featureshardset(quoted_featureshardclip)


# git commit -am "gigaq: LogisticFeatureBagProbe: BUILD"; dbx "autopath.gigaq.pipelines.gigapath_logisticfeature_bags_probe('GIGAPATH_BASELINE_CPTAC_8020_TEST', n_bins=2).build()"
def gigapath_logistic_feature_bags_probe(name, n_bins: int = 2) -> LogisticFeatureBagProbe:
    if name == "GIGAPATH_BASELINE_CPTAC_8020_TEST":
        return LogisticFeatureBagProbe(
                    spec=dict(featurebags=dbx.quote(gigapath_feature_bag_clip, 'GIGAPATH_BASELINE_CPTAC_8020_TEST'), n_bins=n_bins,))
    else:
        raise ValueError(f"Unknown feature bags probe: {name}")

# git commit -am "gigaq: FeaturePairwiseDistances: BUILD"; dbx.print "autopath.gigaq.pipelines.gigapath_feature_pairwise_distances('GIGAPATH_BASELINE_CPTAC_8020_TEST_10_4000_4000').set(n_devices=3).build()"
# git commit -am "gigaq: FeaturePairwiseDistances: BUILD"; dbx.print "autopath.gigaq.pipelines.gigapath_feature_pairwise_distances('GIGAPATH_BASELINE_CPTAC_8020_TEST_10_4000_20000').set(n_devices=3).build()"
# git commit -am "gigaq: FeaturePairwiseDistances: BUILD"; dbx.print "autopath.gigaq.pipelines.gigapath_feature_pairwise_distances('GIGAPATH_BASELINE_CPTAC_8020_TEST_20_4000_20000').set(n_devices=3).build()"
def gigapath_feature_pairwise_distances(name) -> FeaturePairwiseDistances:
    if name == "GIGAPATH_BASELINE_CPTAC_8020_TEST_10_4000_4000":
        return FeaturePairwiseDistances(
                    spec=dict(featurebags=dbx.quote(gigapath_feature_bag_clip, 'GIGAPATH_BASELINE_CPTAC_8020_TEST'),
                              n_chunks=10,
                              row_chunk_size=4000,
                              col_chunk_size=4000,
                    ), 
        )
    elif name == "GIGAPATH_BASELINE_CPTAC_8020_TEST_10_4000_20000":
        return FeaturePairwiseDistances(
                    spec=dict(featurebags=dbx.quote(gigapath_feature_bag_clip, 'GIGAPATH_BASELINE_CPTAC_8020_TEST'),
                              n_chunks=10,
                              row_chunk_size=4000,
                              col_chunk_size=20000,
                    ), 
        )
    elif name == "GIGAPATH_BASELINE_CPTAC_8020_TEST_20_4000_20000":
        return FeaturePairwiseDistances(
                    spec=dict(featurebags=dbx.quote(gigapath_feature_bag_clip, 'GIGAPATH_BASELINE_CPTAC_8020_TEST'),
                              n_chunks=20,
                              row_chunk_size=4000,
                              col_chunk_size=20000,
                    ), 
        )
    elif name == "GIGAPATH_BASELINE_CPTAC_8020_TEST_5_4000_20000":
        return FeaturePairwiseDistances(
                    spec=dict(featurebags=dbx.quote(gigapath_feature_bag_clip, 'GIGAPATH_BASELINE_CPTAC_8020_TEST'),
                              n_chunks=5,
                              row_chunk_size=4000,
                              col_chunk_size=20000,
                    ), 
        )
    elif name == "GIGAPATH_BASELINE_CPTAC_8020_TEST_5_4000_4000":
        return FeaturePairwiseDistances(
                    spec=dict(featurebags=dbx.quote(gigapath_feature_bag_clip, 'GIGAPATH_BASELINE_CPTAC_8020_TEST'),
                              n_chunks=5,
                              row_chunk_size=4000,
                              col_chunk_size=4000,
                    ), 
        )
    elif name == "GIGAPATH_BASELINE_CPTAC_8020_TEST_5_20000_4000":
        return FeaturePairwiseDistances(
                    spec=dict(featurebags=dbx.quote(gigapath_feature_bag_clip, 'GIGAPATH_BASELINE_CPTAC_8020_TEST'),
                              n_chunks=5,
                              row_chunk_size=20000,
                              col_chunk_size=4000,
                    ), 
        )
    elif name == "GIGAPATH_BASELINE_CPTAC_8020_TEST_5_1000_5000":
        return FeaturePairwiseDistances(
                    spec=dict(featurebags=dbx.quote(gigapath_feature_bag_clip, 'GIGAPATH_BASELINE_CPTAC_8020_TEST'),
                              n_chunks=5,
                              row_chunk_size=1000,
                              col_chunk_size=5000,
                    ), 
        )
    elif name == "GIGAPATH_BASELINE_CPTAC_8020_TEST_5_1000_10000":
        return FeaturePairwiseDistances(
                    spec=dict(featurebags=dbx.quote(gigapath_feature_bag_clip, 'GIGAPATH_BASELINE_CPTAC_8020_TEST'),
                              n_chunks=5,
                              row_chunk_size=1000,
                              col_chunk_size=10000,
                    ), 
        )
    elif name == "GIGAPATH_BASELINE_CPTAC_8020_TEST_5_1000_20000":
        return FeaturePairwiseDistances(
                    spec=dict(featurebags=dbx.quote(gigapath_feature_bag_clip, 'GIGAPATH_BASELINE_CPTAC_8020_TEST'),
                              n_chunks=5,
                              row_chunk_size=1000,
                              col_chunk_size=20000,
                    ), 
        )
    elif name == "GIGAPATH_BASELINE_CPTAC_8020_TEST_5_1000_50000":
        return FeaturePairwiseDistances(
                    spec=dict(featurebags=dbx.quote(gigapath_feature_bag_clip, 'GIGAPATH_BASELINE_CPTAC_8020_TEST'),
                              n_chunks=5,
                              row_chunk_size=1000,
                              col_chunk_size=50000,
                    ), 
        )
    elif name == "GIGAPATH_BASELINE_CPTAC_8020_TEST_5_1000_100000":
        return FeaturePairwiseDistances(
                    spec=dict(featurebags=dbx.quote(gigapath_feature_bag_clip, 'GIGAPATH_BASELINE_CPTAC_8020_TEST'),
                              n_chunks=5,
                              row_chunk_size=1000,
                              col_chunk_size=100000,
                    ), 
        )
    elif name == "GIGAPATH_BASELINE_CPTAC_8020_TEST_5_1000_200000":
        return FeaturePairwiseDistances(
                    spec=dict(featurebags=dbx.quote(gigapath_feature_bag_clip, 'GIGAPATH_BASELINE_CPTAC_8020_TEST'),
                              n_chunks=5,
                              row_chunk_size=1000,
                              col_chunk_size=200000,
                    ), 
        )
    elif name == "GIGAPATH_BASELINE_CPTAC_8020_TEST_5_1000_400000":
        return FeaturePairwiseDistances(
                    spec=dict(featurebags=dbx.quote(gigapath_feature_bag_clip, 'GIGAPATH_BASELINE_CPTAC_8020_TEST'),
                              n_chunks=5,
                              row_chunk_size=1000,
                              col_chunk_size=400000,
                    ), 
        )
    elif name == "GIGAPATH_BASELINE_CPTAC_8020_TEST_50_100_800000":
        return FeaturePairwiseDistances(
                    spec=dict(featurebags=dbx.quote(gigapath_feature_bag_clip, 'GIGAPATH_BASELINE_CPTAC_8020_TEST'),
                              n_chunks=50,
                              row_chunk_size=100,
                              col_chunk_size=800000,
                    ), 
        )
    elif name == "GIGAPATH_BASELINE_CPTAC_8020_TEST_5_1000_150000":
        return FeaturePairwiseDistances(
                    spec=dict(featurebags=dbx.quote(gigapath_feature_bag_clip, 'GIGAPATH_BASELINE_CPTAC_8020_TEST'),
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


def gigapath_featurebagset_dataloader(name, root: str = None, shuffle_bags_seed: int = None, **dataloader_kwargs):
    featureset = gigapath_featurebagset(name, root=root,shuffle_bags_seed=shuffle_bags_seed)
    return ClipDataLoaderBuilder(spec=dict(
                            clip_dataset=featureset,
                            batch_size=dataloader_kwargs.get('batch_size', None),
                            shuffle=dataloader_kwargs.get('shuffle', False),
                          ),  
                          dataloader_kwargs=dataloader_kwargs,
    ).dataloader()


# git commit -am "gigaq: FeaturebagsetDataloader: SAMPLES"; dbx.print "autopath.gigaq.pipelines.gigapath_featurebagset_dataloader_samples('GIGAPATH_BASELINE_CPTAC_8020_TRAIN', 10, batch_size=1, num_workers=1)"
# git commit -am "gigaq: FeaturebagsetDataloader: SAMPLES"; dbx.print "autopath.gigaq.pipelines.gigapath_featurebagset_dataloader_samples('GIGAPATH_BASELINE_CPTAC_8020_TRAIN', 20, batch_size=1, num_workers=2)"
# git commit -am "gigaq: FeaturebagsetDataloader: SAMPLES"; dbx.print "autopath.gigaq.pipelines.gigapath_featurebagset_dataloader_samples('GIGAPATH_BASELINE_CPTAC_8020_TRAIN', 100, batch_size=1, num_workers=1)"
# git commit -am "gigaq: FeaturebagsetDataloader: SAMPLES"; dbx.print "autopath.gigaq.pipelines.gigapath_featurebagset_dataloader_samples('GIGAPATH_BASELINE_CPTAC_8020_TRAIN', 100, batch_size=1, num_workers=2)"
# git commit -am "gigaq: FeaturebagsetDataloader: SAMPLES"; dbx.print "autopath.gigaq.pipelines.gigapath_featurebagset_dataloader_samples('GIGAPATH_BASELINE_CPTAC_8020_TRAIN', 100, batch_size=1, num_workers=1, prefetch_factor=1)" 2.71s/it
# git commit -am "gigaq: FeaturebagsetDataloader: SAMPLES"; dbx.print "autopath.gigaq.pipelines.gigapath_featurebagset_dataloader_samples('GIGAPATH_BASELINE_CPTAC_8020_TRAIN', 100, batch_size=1, num_workers=4, prefetch_factor=1)" 2.61s/it
# git commit -am "gigaq: FeaturebagsetDataloader: SAMPLES"; dbx.print "autopath.gigaq.pipelines.gigapath_featurebagset_dataloader_samples('GIGAPATH_BASELINE_CPTAC_8020_TRAIN', 100, batch_size=1, num_workers=2, prefetch_factor=None)" 2.58s/it
# git commit -am "gigaq: FeaturebagsetDataloader: SAMPLES"; dbx.print "autopath.gigaq.pipelines.gigapath_featurebagset_dataloader_samples('GIGAPATH_BASELINE_CPTAC_8020_TRAIN', 100, batch_size=1, num_workers=2, prefetch_factor=1)" 2.47s/it
# git commit -am "gigaq: FeaturebagsetDataloader: SAMPLES"; dbx.print "autopath.gigaq.pipelines.gigapath_featurebagset_dataloader_samples('GIGAPATH_BASELINE_CPTAC_8020_TRAIN', 400, batch_size=8, num_workers=1, prefetch_factor=2)" 2.47s/it
# git commit -am "gigaq: FeaturebagsetDataloader: SAMPLES"; dbx.print "autopath.gigaq.pipelines.gigapath_featurebagset_dataloader_samples('GIGAPATH_BASELINE_CPTAC_8020_TRAIN', 400, batch_size=8, num_workers=1, prefetch_factor=1,)" 2.38s/it
# git commit -am "gigaq: FeaturebagsetDataloader: SAMPLES"; dbx.print "autopath.gigaq.pipelines.gigapath_featurebagset_dataloader_samples('GIGAPATH_BASELINE_CPTAC_8020_TRAIN', 400, batch_size=4, num_workers=4, prefetch_factor=1,)" 2.28s/it
# git commit -am "gigaq: FeaturebagsetDataloader: SAMPLES"; dbx.print "autopath.gigaq.pipelines.gigapath_featurebagset_dataloader_samples('GIGAPATH_BASELINE_CPTAC_8020_TRAIN', 400, batch_size=4, num_workers=1, prefetch_factor=1,)" 2.25s/it
# unset DBXREPO; git commit -am "gigaq: FeaturebagsetDataloader: SAMPLES"; dbx.print "autopath.gigaq.pipelines.gigapath_featurebagset_dataloader_samples('GIGAPATH_BASELINE_CPTAC_8020_TRAIN', 400, root='/tmp/dmitry/datalake', batch_size=4, num_workers=4, prefetch_factor=None)" 1.39s/it
# unset DBXREPO; git commit -am "gigaq: FeaturebagsetDataloader: SAMPLES"; dbx.print "autopath.gigaq.pipelines.gigapath_featurebagset_dataloader_samples('GIGAPATH_BASELINE_CPTAC_8020_TRAIN', 400, root='/tmp/dmitry/datalake', batch_size=4, num_workers=8, prefetch_factor=1)" ~=
# unset DBXREPO; git commit -am "gigaq: FeaturebagsetDataloader: SAMPLES"; dbx.print "autopath.gigaq.pipelines.gigapath_featurebagset_dataloader_samples('GIGAPATH_BASELINE_CPTAC_8020_TRAIN', 400, root='/tmp/dmitry/datalake', batch_size=4, num_workers=2, prefetch_factor=2)" ~>
# unset DBXREPO; git commit -am "gigaq: FeaturebagsetDataloader: SAMPLES"; dbx.print "autopath.gigaq.pipelines.gigapath_featurebagset_dataloader_samples('GIGAPATH_BASELINE_CPTAC_8020_TRAIN', 400, root='/tmp/dmitry/datalake', batch_size=4, num_workers=2, prefetch_factor=1)" ~=
# unset DBXREPO; git commit -am "gigaq: FeaturebagsetDataloader: SAMPLES"; dbx.print "autopath.gigaq.pipelines.gigapath_featurebagset_dataloader_samples('GIGAPATH_BASELINE_CPTAC_8020_TRAIN', 400, root='/tmp/dmitry/datalake', batch_size=8, num_workers=2, prefetch_factor=1)" ~=
def gigapath_featurebagset_dataloader_samples(name, n, root: str = None, shuffle_bags: bool = False, return_last: bool = False, **dataloader_kwargs):
    batch_size = dataloader_kwargs.get('batch_size', None)
    dataloader = gigapath_featurebagset_dataloader(name, root=root, shuffle=shuffle_bags, **dataloader_kwargs)
    progress = tqdm(total=n)
    for i, _ in enumerate(dataloader):
        progress.update(batch_size if batch_size is not None else 1)
        if i*batch_size >= n-1:
            break
    if return_last:
        return _


def gigapath_featureshardset_dataloader(name, *args, **kwargs):
    featureset = gigapath_featureshardset(name)
    return torch.utils.data.DataLoader(featureset, *args, **kwargs)


def gigapath_featureshardset_dataloader_samples(name, n, *args, return_last: bool = False, **kwargs):
    dataloader = gigapath_featureshardset_dataloader(name, *args, **kwargs)
    itor = tqdm(iter(dataloader), total=n)
    for i, _ in enumerate(itor):
        if i >= n-1:
            break
    if return_last:
        return _


def gigapath_featureshardset_dataloader_sample(name, *args, **kwargs):
    featureset = gigapath_featureshardset(name)
    dataloader = torch.utils.data.DataLoader(featureset, *args, **kwargs)
    return next(iter(dataloader))


# git commit -am "gigaq: VRED"; dbx.print "autopath.gigaq.pipelines.gigapath_vred('GIGAPATH_VRED_2HDN_100CLS_5CHN')"
def gigapath_vred(name, capture_mixture_distributions: bool = False):
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
    vred = VariationalReEncoderDecoder(
        classifier=classifier,
        latent_gaussians=latent_gaussians,
        capture_mixture_distributions=capture_mixture_distributions,
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
def gigapath_vred_evaluator(vred_dataset_name, *, use_bags: bool = True, shuffle_bags: bool = True, capture_mixture_distributions: bool = False, **dataloader_kwargs):
    vredname, _clipname = vred_dataset_name.split('_BASELINE_CPTAC_')
    clipname = "GIGAPATH_BASELINE_CPTAC_" + _clipname
    vred = dbx.quote(gigapath_vred, vredname, capture_mixture_distributions=capture_mixture_distributions)
    if use_bags:
        featureloader = dbx.quote(gigapath_featurebagset_dataloader, clipname, shuffle_bags=shuffle_bags, **dataloader_kwargs)
    else:
        featureloader = dbx.quote(gigapath_featureshardset_dataloader, clipname, **dataloader_kwargs)
    vred_evaluator = VariationalReEncoderDecoderEvaluator(spec=dict(vred=vred, dataloader=featureloader))
    return vred_evaluator


# git commit -am 'gigaq: VRED: STILL: BUILD'; dbx.print 'autopath.gigaq.pipelines.gigapath_vred_still("GIGAPATH_VRED_2HDN_100CLS_5CHN_BASELINE_CPTAC_8020_TEST", max_epochs=3, max_steps=100, batch_size=8, shuffle=False, loss_capture_distribution=True, logs="/home/t-9dkarp/autopath/tensorboard/vred", n_devices=1, learning_rate=8e-3).build()'
# git commit -am 'gigaq: VRED: STILL: BUILD'; dbx.print 'autopath.gigaq.pipelines.gigapath_vred_still("GIGAPATH_VRED_2HDN_100CLS_5CHN_BASELINE_CPTAC_8020_TEST", max_epochs=1, max_steps=100, batch_size=4, shuffle=True, log_mixture_distributions=True, logs="/home/t-9dkarp/autopath/tensorboard/vred", n_devices=1, learning_rate=1e-6).set(capture_output=True).build()'
# git commit -am 'gigaq: VRED: STILL: BUILD'; dbx.print 'autopath.gigaq.pipelines.gigapath_vred_still("GIGAPATH_VRED_2HDN_100CLS_5CHN_BASELINE_CPTAC_8020_TEST", max_epochs=1, max_steps=100, batch_size=4, shuffle=True, log_mixture_distributions=True, logs="/home/t-9dkarp/autopath/tensorboard/vred", n_devices=1, learning_rate=1e-3).set(capture_output=True).build()'
# git commit -am 'gigaq: VRED: STILL: BUILD'; dbx.print 'autopath.gigaq.pipelines.gigapath_vred_still("GIGAPATH_VRED_2HDN_100CLS_5CHN_BASELINE_CPTAC_8020_TEST", max_epochs=1, max_steps=100, batch_size=4, shuffle=True, log_mixture_distributions=True, logs="/home/t-9dkarp/autopath/tensorboard/vred", n_devices=1, learning_rate=1e-2).set(capture_output=True).build()'
# git commit -am 'gigaq: VRED: STILL: BUILD'; dbx.print 'autopath.gigaq.pipelines.gigapath_vred_still("GIGAPATH_VRED_2HDN_100CLS_5CHN_BASELINE_CPTAC_8020_TEST", max_epochs=1, max_steps=100, batch_size=4, shuffle=False, log_mixture_distributions=True, logs="/home/t-9dkarp/autopath/tensorboard/vred", n_devices=1, learning_rate=1e-2).set(capture_output=True).build()'
# git commit -am 'gigaq: VRED: STILL: BUILD'; dbx.print 'autopath.gigaq.pipelines.gigapath_vred_still("GIGAPATH_VRED_2HDN_100CLS_5CHN_BASELINE_CPTAC_8020_TEST", max_epochs=1, max_steps=100, batch_size=1, use_bags=True, shuffle_bags_seed=42, log_mixture_distributions=True, log_weights=False, log_gradients=False, logs="/home/t-9dkarp/autopath/tensorboard/vred", n_devices=2, learning_rate=1e-5, gradient_clip_algorithm="norm", gradient_clip_val=1.0,).set(capture_output=True).build()'
# git commit -am 'gigaq: VRED: STILL: BUILD'; dbx.print 'autopath.gigaq.pipelines.gigapath_vred_still("GIGAPATH_VRED_2HDN_100CLS_5CHN_BASELINE_CPTAC_8020_TEST", max_epochs=1, max_steps=100, batch_size=2, use_bags=True, shuffle_bags_seed=42, log_mixture_distributions=True, log_weights=False, log_gradients=False, logs="/home/t-9dkarp/autopath/tensorboard/vred", n_devices=2, learning_rate=1e-5, gradient_clip_algorithm="norm", gradient_clip_val=1.0,).set(capture_output=True).build()'
# UNSET DBXREPO; git commit -am "gigaq: VRED: STILL: BUILD"; dbx.print "autopath.gigaq.pipelines.gigapath_vred_still('GIGAPATH_VRED_2HDN_100CLS_5CHN_BASELINE_CPTAC_8020_TEST', root='/tmp/dmitry/datalake', max_epochs=3, max_steps=None, ckpt_every_n_steps=100, use_bags=True, shuffle_bags_seed=42, log_mixture_distributions=True, log_weights=False, log_gradients=False, logs='/home/t-9dkarp/autopath/tensorboard/vred', n_devices=3, learning_rate=1e-5, gradient_clip_algorithm='norm', gradient_clip_val=1.0, batch_size=8, num_workers=2, prefetch_factor=1).set(capture_output=True).build()"
# UNSET DBXREPO; git commit -am "gigaq: VRED: STILL: BUILD"; dbx.print "autopath.gigaq.pipelines.gigapath_vred_still('GIGAPATH_VRED_2HDN_100CLS_5CHN_BASELINE_CPTAC_8020_TRAIN', root='/tmp/dmitry/datalake', max_epochs=3, max_steps=None, ckpt_every_n_steps=100, use_bags=True, shuffle_bags_seed=42, log_mixture_distributions=True, log_weights=False, log_gradients=False, logs='/home/t-9dkarp/autopath/tensorboard/vred', n_devices=3, learning_rate=1e-5, gradient_clip_algorithm='norm', gradient_clip_val=1.0, batch_size=8, num_workers=2, prefetch_factor=1, pin_memory=True, precision='medium',).set(capture_output=True).build()"
#
"""
UNSET DBXREPO; git commit -am "gigaq: VRED: STILL: BUILD"; dbx.print "autopath.gigaq.pipelines.gigapath_vred_still('GIGAPATH_VRED_2HDN_100CLS_5CHN_BASELINE_CPTAC_8020_TRAIN', root='/tmp/dmitry/datalake', ckpt='autopath.models.vred.VariationalReEncoderDecoderStill/c074fe8c90427a27a5c13cebfda7007b1e4391cad734dfc73cee6a1af6af96f2/ckpts/epoch=0-step=1900.ckpt', \
       max_epochs=3, max_steps=None, ckpt_every_n_steps=100, use_bags=True, shuffle_bags_seed=42, log_mixture_distributions=True, log_weights=False, log_gradients=False, logs='/home/t-9dkarp/autopath/tensorboard/vred', n_devices=3, learning_rate=1e-5, gradient_clip_algorithm='norm', gradient_clip_val=1.0, batch_size=6, num_workers=2, prefetch_factor=1, pin_memory=True, precision='medium',).set(capture_output=True).build()"

"""
def gigapath_vred_still(vred_dataset_name = None, 
                        *, 
                        root: str = None, 
                        tag: str = None,
                        use_bags: bool = True,
                        shuffle_bags_seed: int = None,
                        learning_rate: float = 0.03, 
                        precision: str = None, 
                        scheduler: str = 'cosine', 
                        max_epochs: int = 3, 
                        max_steps: int = 1, 
                        ckpt_every_n_steps: int = 1000, 
                        ckpt: str = None, 
                        n_devices: int = 1, 
                        log_mixture_distributions: bool = False, 
                        log_weights: bool = False, 
                        log_gradients: bool = False, 
                        skip_invalid_gradients: bool = True,
                        gradient_clip_val: float = 1.0,
                        gradient_clip_algorithm: str = 'norm',
                        logs: str = None,
                        **dataloader_kwargs,
    ):
    if vred_dataset_name is None:
        still = VariationalReEncoderDecoderStill
    else:
        vredname, _clipname = vred_dataset_name.split('_BASELINE_CPTAC_')
        clipname = "GIGAPATH_BASELINE_CPTAC_" + _clipname
        tag = f"{vred_dataset_name}_{tag}" if tag else vred_dataset_name
        vred = dbx.quote(gigapath_vred, vredname, capture_mixture_distributions=log_mixture_distributions)
        if use_bags:
            featureloader = dbx.quote(gigapath_featurebagset_dataloader, clipname, root=root, shuffle_bags_seed=shuffle_bags_seed, **dataloader_kwargs)
        else:
            featureloader = dbx.quote(gigapath_featureshardset_dataloader, clipname, root=root, **dataloader_kwargs)
        lightning = dbx.quote(VariationalReEncoderDecoderLightning, spec=dict(vred=vred, learning_rate=learning_rate, scheduler=scheduler))
        still = VariationalReEncoderDecoderStill(spec=dict(
                    lightning=lightning, 
                    dataloader=featureloader,
                    max_epochs=max_epochs,
                    max_steps=max_steps,
                    ckpt_every_n_steps=ckpt_every_n_steps,
                    skip_invalid_gradients=skip_invalid_gradients,
                    log_weights=log_weights,
                    log_gradients=log_gradients,
                    init_ckpt_path_or_anchor=ckpt,
                    gradient_clip_val=gradient_clip_val,
                    gradient_clip_algorithm=gradient_clip_algorithm,
                    precision=precision,
                ),
            n_devices=n_devices,
            logs=logs,
            tag=tag,
        )
    return still
    
