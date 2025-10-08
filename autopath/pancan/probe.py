
from dataclasses import dataclass
import datetime
import functools
import gc
import math
from typing import Optional, Union, Tuple


import numpy as np
import pandas as pd
import torch


from sklearn.metrics import classification_report
from sklearn.linear_model import LogisticRegression, LinearRegression


from dbx import (
	Logger,
    Datashard,
	Datablock,
    write_tensor, 
    read_tensor,
    write_npz,
    read_npz,
    write_pickle,
    read_pickle,
    TorchMultithreadingDatashardBatchBuilder,
    TorchMultiprocessingDatashardBatchBuilder,
)


from .features import FeatureBags


class FeatureProbe:
    @staticmethod
    def ndarray(X: Union[np.ndarray, list, torch.Tensor, pd.DataFrame]):
        if isinstance(X, list):
            X = np.array(X)
        elif isinstance(X, torch.Tensor):
            X = X.numpy()
        elif isinstance(X, pd.DataFrame):
            X = X.values
        elif isinstance(X, np.ndarray):
            pass
        else:
            raise ValueError(f"Unknown ndarray input type: {type(X)}")
        return X

    @staticmethod
    def discretize_features(X: Union[np.ndarray, list, torch.Tensor, pd.DataFrame], d:int = 4) -> np.ndarray:
        features = FeatureProbe.ndarray(X)
        return pd.DataFrame(features).apply(lambda c: pd.qcut(c, d, labels=False, duplicates='drop')).fillna(0.0).values

    @staticmethod
    def split_train_test(X: Union[np.ndarray, list, torch.Tensor, pd.DataFrame], 
                         y: Optional[Union[np.ndarray, list, torch.Tensor, pd.DataFrame]] = None, 
                         train_fraction:float=0.8
    ):
        #TODO: split_slides_labels_train_test() -> split_features_labels_train_test()
        X = FeatureProbe.ndarray(X)
        if y is not None:
            y = FeatureProbe.ndarray(y)
        N = y.shape[0]
        permutation = permutation = np.random.permutation(range(N))
        n = int(math.floor(N*train_fraction))
        train = list(permutation[:n])
        test = list(permutation[n:])
        if y is not None:
            return (X[train, :], y[train]), (X[test, :], y[test])
        else:
            return X[train, :], X[test, :]

    @staticmethod
    def evaluate_features(Xy: Tuple[Union[np.ndarray, list, torch.Tensor, pd.DataFrame], 
                                    Union[np.ndarray, list, torch.Tensor, pd.DataFrame],], 
                          *, 
                          fraction=0.8
    ):
        features, labels = Xy
        features = FeatureProbe.ndarray(features)
        labels = FeatureProbe.ndarray(labels)
        N = len(labels)
        ntrain = int(N*fraction)
        randomidx = np.random.permutation(list(range(N)))
        trainidx = randomidx[:ntrain]
        testidx = randomidx[ntrain:]
       
        X_train = features[trainidx, :]
        y_train = labels[trainidx]

        X_test = features[testidx, :]
        y_test = labels[testidx]

        clf = LogisticRegression()
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)
        report = classification_report(y_test, y_pred)
        return report

    @staticmethod
    def evaluate_features2(Xy1,
                           Xy2,
                           *, 
                           fraction=0.8,
                           label1="(1)",
                           label2="(2)",
                           log: Logger = Logger(),
    ):
        log.verbose(f"Evaluating features {label1}: started at {datetime.datetime.now()}")
        report1 = FeatureProbe.evaluate_features(Xy1, fraction=fraction)
        log.verbose(f"Evaluating features {label1}: finished at {datetime.datetime.now()}")
        log.verbose(f"Evaluating features {label2}: started at {datetime.datetime.now()}")
        report2 = FeatureProbe.evaluate_features(Xy2, fraction=fraction)
        log.verbose(f"Evaluating features {label2}: finished at {datetime.datetime.now()}")

        rstr = f"---------- {label1} ------------\n{report1}\n---------- {label2} ------------\n{report2}"
        log.verbose(rstr)
        return report1, report2

    def plot_features_umap(self,
                           Xy,
                           *, 
                           fraction = 0.01, 
                           title="", 
                           label="",
                           use_umap_plot=False,
                           output_path=None,
                           log: Logger = Logger(),
    ):
        features, labels = Xy
        import umap
        import umap.plot
        from matplotlib import pyplot as plt
        # UMAP visualization of original features (1% subsample)
        um = umap.UMAP(n_components=2, random_state=42)
        umap_features = um.fit_transform(features)
        ulabels = set(labels)
        ilablesmap = {l: i for i, l in enumerate(ulabels)}
        ilabels = [ilablesmap[l] for l in labels]
        plt.scatter(umap_features[:, 0], umap_features[:, 1], c=ilabels, label=labels)

        if output_path is not None:
            plt.savefig(output_path)
            log.verbose(f"Wrote figure to '{output_path}'")
        else:
            plt.show()
        return umap_features


class FeatureBagsProbe(Datablock, FeatureProbe):
    TOPICFILES = {
        'labels': 'labels.npz',
        'cdf': 'cdf.pt',
        'aggregated_features': 'aggregated_features.pt',
        'discretized_features': 'discretized_features.pt',
        'cdf_umap': 'cdf_umap.png',
        'discretized_features_umap': 'discretized_features_umap.png',
        'evaluation_reports': 'evaluation_reports.pkl',
    }
    @dataclass
    class CONFIG:
        featurebags: FeatureBags
        n_bins: int = 2
        evaluation_fraction: float = 0.8
        umap_fraction: float = 0.01
        aggregation: str = "mean"

    def __post_init__(self):
        assert self.config.aggregation in ["mean"], f"Unknown aggregation: {self.config.aggregation}"
        return self

    def __build__(self):
        # labels
        labels = np.array([bag.label for bag in self.config.featurebags.bags])
        write_npz(self.path('labels', ensure_dirpath=True), labels=labels)

        feature_list = []
        for featurebag in self.config.featurebags.bags:
            feature_list.append(torch.mean(featurebag.features(), dim=0))
        aggregated_features = torch.stack(feature_list)
        assert len(labels) == len(aggregated_features), f"len(labels) != len(aggregate_features): {len(labels)} != {len(aggregated_features)}"
        write_tensor(aggregated_features, self.path('aggregated_features', ensure_dirpath=True))

        # cdf
        quantiles = np.arange(0.0, 1.0, 1.0/self.config.n_bins)
        cdf = torch.tensor(np.percentile(aggregated_features.numpy(), quantiles, axis=0))
        write_tensor(cdf, self.path('cdf', ensure_dirpath=True))

        # discretized_features
        discretized_features = torch.Tensor(self.discretize_features(aggregated_features.numpy(), self.config.n_bins))
        write_tensor(discretized_features, self.path('discretized_features', ensure_dirpath=True))
        
        # evaluation_reports
        continuous, discretized = self.evaluate_features2(
            (aggregated_features, labels), 
            (discretized_features, labels),
            fraction=self.config.evaluation_fraction,
            label1='continuous',
            label2='discretized',
            log=self.log,
        )
        evaluation_reports = {'continuous': continuous, 'discretized': discretized}
        write_pickle(evaluation_reports, self.path('evaluation_reports', ensure_dirpath=True))
        return self

    def __read__(self, topic):
        if topic == 'labels':
            result = read_npz(self.path('labels'), 'labels')
        elif topic == 'aggregated_features':
            result = read_tensor(self.path('aggregated_features'))
        elif topic == 'discretized_features':
            result = read_tensor(self.path('discretized_features'))
        elif topic == 'cdf':
            result = read_tensor(self.path('cdf'))
        elif topic == 'evaluation_reports':
            result = read_pickle(self.path('evaluation_reports'))
        else:
            raise ValueError(f"Unknown topic: {topic}")
        return result


class FeaturePairwiseDistancesChunk(Datablock):
    TOPICFILE = "pairwise_distances.pt"

    @dataclass
    class CONFIG:
        featurebags: FeatureBags
        featurebags_size: int
        row_batch_offset: int
        row_batch_size: int
        sideband_layer: Optional[str] = None

    def __post_init__(self):
        self.rows = list(range(self.config.row_batch_offset, 
                               min(self.config.row_batch_offset + self.config.row_batch_size,
                                   self.config.featurebags_size)))
        return self

    def __build__(self, features):
        self.log.detailed(f"Building FeaturePairwiseDistancesChunk with rows {self.rows} on device {features.device}")
        pairwise_distances = torch.cdist(features[self.rows], features).to('cpu')
        self.log.debug(f"Built FeatureDistanceChunk at offset {self.config.row_batch_offset} with shape {pairwise_distances.shape} on device {features.device}")
        write_tensor(pairwise_distances, self.path(ensure_dirpath=True))
        del pairwise_distances
        return self
    
    def __read__(self):
        result = read_tensor(self.path())
        return result
    
    @functools.cached_property
    def tensor(self):
        return self.read()
        

class FeatureBagsPairwiseDistancesProbe(Datablock):
    TOPICFILES = {"features_size": "features_size.pt"}
    @dataclass
    class CONFIG:
        featurebags: FeatureBags
        row_batch_size: int
        sideband_layer: Optional[str] = None

    def __init__(self, *args, row_batch_size: int = 1, devices: list[str] = ['cuda:0'], **kwargs):
        super().__init__(*args, row_batch_size=row_batch_size, devices=devices, **kwargs)
        self._features = None
        
    def features(self):
        if self._features is None:
            self.log.debug(f"Reading features from {len(self.config.featurebags.bags)} feature bags")
            feature_list = []
            for featurebag in self.config.featurebags.bags:
                feature_list.extend(featurebag.features)
            self._features = torch.stack(feature_list)
            self.log.debug(f"Combined features: shape: {self._features.shape}")
            #REMOVE: DEADLOCK
            #assert len(features) == self.config.featurebags.size(), f"len(features) != self.config.featurebags.size(): {len(features)} != {self.config.featurebags.size()}"
        return self._features

    @functools.cached_property
    def chunks(self):
        return [FeaturePairwiseDistancesChunk(spec=dict(
                    featurebags=self.spec['featurebags'],
                    featurebags_size=self.features_size, 
                    row_batch_offset=i, 
                    row_batch_size=self.spec['row_batch_size'])) for i in range(0, self.features_size, self.spec['row_batch_size'])]
    
    @property
    def num_chunks(self):
        return len(self.chunks)
    
    @functools.cached_property
    def features_size(self):
        if self.valid():
            self.log.debug(f"Reading features_size from {self.path('features_size')}")
            return read_tensor(self.path('features_size')).item()
        else:
            self.log.debug(f"Calculating features_size")
            return len(self.features())

    def __build__(self):
        chunks = self.chunks
        self.log.debug("Retrieving features")
        features = self.features()
        features_size = self.features_size
        self.log.debug(f"Forming FeaturePairwiseDistancesChunks with size {features_size} and batch size {self.spec['row_batch_size']}")
        
        self.log.debug(f"Found {len(chunks)} FeaturePairwiseDistancesChunks.  Looking for missing chunks")
        missing_chunks = [chunk for chunk in chunks if not chunk.valid()]
        self.log.debug(f"Found {len(missing_chunks)} missing chunks")
        self.log.debug(f"Building all missingpairwise feature distance chunks")
        built_chunks = TorchMultithreadingDatashardBatchBuilder(devices=self.devices, log=self.log).build_shards(missing_chunks, features)
        self.log.verbose(f"Built all pairwise feature distance chunks: {len(built_chunks)}")
        write_tensor(torch.tensor([features_size]), self.path('features_size', ensure_dirpath=True))
        return self
    
    def __read__(self, topic):
        if topic == 'features_size':
            result = read_tensor(self.path('features_size'))    
        return result
    

class FeatureBagsUniquePairwiseDistancesChunk(Datablock):
    TOPICFILES = {"subsample_indices": "subsample_indices.npy",
                  "original_order_indices": "original_order_indices.npy",
                  "unique_value_indices": "unique_value_indices.npy"
    }
    
    @dataclass
    class CONFIG:
        featuredist_chunk: FeaturePairwiseDistancesChunk
        subsample_fraction: float = 1.0
        selfdist_eps: float = 1e-6

    def __build__(self):
        _chunk = self.config.featuredist_chunk.read()
        if self.config.subsample_fraction < 1.0:
            subsample_permutation = np.random.permutation(len(_chunk))
            subsample_indices = subsample_permutation[:int(len(_chunk)*self.config.subsample_fraction)]
            chunk = _chunk[subsample_indices]
            self.log.debug(f"Subsampled tensor down to shape {chunk.shape} on device {self.device}")
        else:
            chunk = _chunk
            subsample_indices = torch.arange(0)
        write_tensor(subsample_indices, self.path('subsample_indices', ensure_dirpath=True))
        del subsample_indices
        self.log.debug(f"Replacing distances below {self.config.selfdist_eps} with {torch.inf}")
        chunk[chunk < self.config.selfdist_eps] = torch.inf
        self.log.debug(f"Sorting pairwise distances in subsampled FeaturePairwiseDistancesChunk {self.config.featuredist_chunk.hashpath()} of shape {chunk.shape} on device {self.device}")
        sorted_chunk, original_order_indices = torch.sort(chunk, dim=-1, descending=False)
        write_tensor(original_order_indices, self.path('original_order_indices', ensure_dirpath=True))
        del chunk
        del original_order_indices
        gc.collect()
        self.log.debug(f"Finding unique pairwise distances in subsampled FeaturePairwiseDistancesChunk {self.config.featuredist_chunk.hashpath()} of shape {sorted_chunk.shape} on device {self.device}")
        unique_distances, unique_value_indices = torch.unique_consecutive(sorted_chunk, return_inverse=True)
        write_tensor(unique_value_indices, self.path('unique_value_indices', ensure_dirpath=True))
        del sorted_chunk
        del unique_distances
        del unique_value_indices
        gc.collect()    
        self.log.debug(f"Built FeatureBagsUniquePairwiseDistancesChunk {self.hashpath()} at offset {self.config.featuredist_chunk.config.row_batch_offset} with shape {unique_distances.shape} on device {self.device}")
        return self
    
    def __read__(self, topic):
        result = read_tensor(self.path(topic))
        return result
    
    @functools.cached_property
    def tensor(self):
        subsample_indices = self.read('subsample_indices')
        original_order_indices = self.read('original_order_indices')
        _chunk = self.config.featuredist_chunk.read()
        tensor = _chunk[subsample_indices][original_order_indices]
        return tensor
    
    @functools.cached_property
    def unique_value_indices(self):
        return self.read('unique_value_indices')
        

class FeatureBagsUniquePairwiseDistancesProbe(Datablock):
    TOPICFILE = "breadcumbs"

    @dataclass
    class CONFIG:
        featurebags_pairwise_distances_probe: FeatureBagsPairwiseDistancesProbe
        subsample_fraction: float = 1.0
        selfdist_eps: float = 1e-6

    def __init__(self, *args, devices: list[str] = ['cuda:0'], **kwargs):
        super().__init__(*args, devices=devices, **kwargs)

    def __build__(self):
        featuredist_chunks = self.config.featurebags_pairwise_distances_probe.chunks
        uniquedist_chunks = [FeatureBagsUniquePairwiseDistancesChunk(spec=dict(featuredist_chunk=featuredist_chunk, subsample_fraction=self.config.subsample_fraction, selfdist_eps=self.config.selfdist_eps)) 
                            for featuredist_chunk in featuredist_chunks
        ]
        self.log.debug(f"Formed {len(uniquedist_chunks)} FeatureBagsUniquePairwiseDistancesChunks")
        missing_uniquedist_chunks = [chunk for chunk in uniquedist_chunks if not chunk.valid()]
        self.log.debug(f"Found {len(missing_uniquedist_chunks)} missing FeatureBagsUniquePairwiseDistancesChunks")
        self.log.debug(f"Building {len(missing_uniquedist_chunks)} FeatureBagsUniquePairwiseDistancesChunks")
        built_uniquedist_chunks = TorchMultiprocessingDatashardBatchBuilder(devices=self.devices, log=self.log).build_shards(missing_uniquedist_chunks)
        self.leave_breadcrumbs()
        self.log.verbose(f"Built all missing FeatureBagsUniquePairwiseDistancesChunks: {len(built_uniquedist_chunks)}")
        return self
    

class FeatureBags2NNDistanceChunk(Datablock):
    TOPICFILE = "twonn_distances.pt"
    @dataclass
    class CONFIG:
        featuredist_chunk: FeaturePairwiseDistancesChunk
        selfdist_eps: float = 1e-6

    def __build__(self):
        featuredist_chunk_tensor = self.config.featuredist_chunk.read()
        self.log.debug(f"Replacing distances below {self.config.selfdist_eps} with {torch.inf}")
        featuredist_chunk_tensor[featuredist_chunk_tensor < self.config.selfdist_eps] = torch.inf
        self.log.debug(f"Locating smallest pairwise distances in FeaturePairwiseDistancesChunk {self.config.featuredist_chunk.hashpath()} of shape {featuredist_chunk_tensor.shape}")
        firstdist = torch.kthvalue(featuredist_chunk_tensor, 1, dim=1)
        if self.debug:
            firstdist_min, firstdist_max = firstdist.values.min(), firstdist.values.max()
            self.log.debug(f"firstdist_min: {firstdist_min}, firstdist_max: {firstdist_max}")
        self.log.debug(f"Locating second smallest pairwise distances in FeaturePairwiseDistancesChunk {self.config.featuredist_chunk.hashpath()} of shape {featuredist_chunk_tensor.shape}")
        seconddist = torch.kthvalue(featuredist_chunk_tensor, 2, dim=1)
        if self.debug:
            seconddist_min, seconddist_max = seconddist.values.min(), seconddist.values.max()
            self.log.debug(f"seconddist_min: {seconddist_min}, seconddist_max: {seconddist_max}")
        assert all(firstdist.values > 0.0), f"firstdist must be > 0.0: {firstdist.values}"
        assert all(seconddist.values > 0.0), f"seconddist must be > 0.0: {seconddist.values}"
        assert all(firstdist.values < torch.inf), f"firstdist must be < torch.inf: {firstdist.values}"
        assert all(seconddist.values < torch.inf), f"seconddist must be < torch.inf: {seconddist.values}"
        del featuredist_chunk_tensor
        twonn_distances = torch.stack([firstdist.values, seconddist.values], dim=-1)
        self.log.debug(f"Built FeatureBags2NNDistanceChunk {self.hashpath()} at offset {self.config.featuredist_chunk.config.row_batch_offset} with shape {twonn_distances.shape}")
        write_tensor(twonn_distances, self.path(ensure_dirpath=True))
        return self
    
    def __read__(self):
        result = read_tensor(self.path())
        return result
    
    @functools.cached_property
    def tensor(self):
        return self.read()
    

class FeatureBags2NNDimProbe(Datablock):
    TOPICFILES = {
        "twonn_distances": "twonn_distances.pt", 
        "dimension": "dimension.pt",
        "model": "model.pkl",
    }
    @dataclass
    class CONFIG:
        featurebags_pairwise_distances_probe: FeatureBagsPairwiseDistancesProbe
        selfdist_eps: float = 1e-6

    def __init__(self, *args, n_workers: int = 1, **kwargs):
        super().__init__(*args, n_workers=n_workers, **kwargs)

    def __build__(self):
        featuredist_chunks = self.config.featurebags_pairwise_distances_probe.chunks
        twonndist_chunks = [FeatureBags2NNDistanceChunk(spec=dict(featuredist_chunk=chunk, selfdist_eps=self.config.selfdist_eps)) 
                            for chunk in featuredist_chunks
        ]
        self.log.debug(f"Found {len(twonndist_chunks)} FeaturePairwiseDistancesChunks")
        missing_twonndist_chunks = [chunk for chunk in twonndist_chunks if not chunk.valid()]
        self.log.debug(f"Found {len(missing_twonndist_chunks)} missing FeaturePairwiseDistancesChunks")
        self.log.debug(f"Building {len(missing_twonndist_chunks)} FeatureBags2NNDistanceChunks")
        built_twonndist_chunks = TorchMultiprocessingDatashardBatchBuilder(n_workers=self.n_workers, log=self.log).build_shards(missing_twonndist_chunks)
        self.log.verbose(f"Built all missing FeatureBags2NNDistance chunks: {len(built_twonndist_chunks)}")
        twonndists = torch.cat([chunk.tensor for chunk in twonndist_chunks], dim=0)
        mus = twonndists[:, 1]/twonndists[:, 0]
        assert all(mus >= 1.0), f"mus must be >= 1.0: {mus}"
        sortedmus, _ = torch.sort(mus, descending=False)
        del mus
        logsortedmus = torch.log(sortedmus)
        cumprob = torch.arange(0, len(logsortedmus))/len(logsortedmus)
        x = logsortedmus
        y = -torch.log(1.0 - cumprob)
        model = LinearRegression()
        self.log.debug(f"Fitting linear model on {len(x)} mu points")
        model.fit(x, y)
        self.log.debug(f"Built linear model of dimension {model.coef_[0]}")
        write_tensor(torch.tensor([model.coef_[0]]), self.path('dimension', ensure_dirpath=True))
        write_pickle(model, self.path('model', ensure_dirpath=True))
        return self
                            




