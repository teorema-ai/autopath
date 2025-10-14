
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


class FeaturesProbe:
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
        features = FeaturesProbe.ndarray(X)
        return pd.DataFrame(features).apply(lambda c: pd.qcut(c, d, labels=False, duplicates='drop')).fillna(0.0).values

    @staticmethod
    def split_train_test(X: Union[np.ndarray, list, torch.Tensor, pd.DataFrame], 
                         y: Optional[Union[np.ndarray, list, torch.Tensor, pd.DataFrame]] = None, 
                         train_fraction:float=0.8
    ):
        #TODO: split_slides_labels_train_test() -> split_features_labels_train_test()
        X = FeaturesProbe.ndarray(X)
        if y is not None:
            y = FeaturesProbe.ndarray(y)
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
        features = FeaturesProbe.ndarray(features)
        labels = FeaturesProbe.ndarray(labels)
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
        report1 = FeaturesProbe.evaluate_features(Xy1, fraction=fraction)
        log.verbose(f"Evaluating features {label1}: finished at {datetime.datetime.now()}")
        log.verbose(f"Evaluating features {label2}: started at {datetime.datetime.now()}")
        report2 = FeaturesProbe.evaluate_features(Xy2, fraction=fraction)
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


class FeatureBagsProbe(Datablock, FeaturesProbe):
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


class FeaturesPairwiseDistancesChunk(Datablock):
    VERSION = 1
    TOPICFILES = {
        'rows': 'rows.npz',
        'cols': 'cols.npz',
        'distances': "distances.pt"
    }

    @dataclass
    class CONFIG:
        featurebags: FeatureBags
        chunk_idx: int
        seed: int = 42
        row_subsample_fraction: float = 1.0
        col_subsample_fraction: float = 1.0
        sideband_layer: Optional[str] = None

    def __build__(self, features):
        rng = np.random.default_rng(self.config.seed)
        M = features.shape[0]
        rows = rng.permutation(M)
        row_batch_size = int(M*self.config.row_subsample_fraction)
        _rows = rows[row_batch_size*self.config.chunk_idx:min(M, row_batch_size*(self.config.chunk_idx+1))]
        del rows
        cols = rng.permutation(M)
        col_batch_size = int(M*self.config.col_subsample_fraction)
        _cols = cols[col_batch_size*self.config.chunk_idx:min(M, col_batch_size*(self.config.chunk_idx+1))]
        del cols
        self.log.detailed(f"Building FeaturePairwiseDistancesChunk at index {self.config.chunk_idx} with rows {_rows} and cols {_cols} on device {features.device}")
        pairwise_distances = torch.cdist(features[_rows], features[_cols]).to('cpu')
        self.log.debug(f"Built FeatureDistanceChunk at index {self.config.chunk_idx} with shape {pairwise_distances.shape} on device {features.device}")
        write_npz(self.path('rows', ensure_dirpath=True), rows=_rows)
        write_npz(self.path('cols', ensure_dirpath=True), cols=_cols)
        write_tensor(pairwise_distances, self.path('distances', ensure_dirpath=True))
        del _rows
        del _cols
        del pairwise_distances
        return self
    
    def __read__(self, topic):
        if topic == 'rows':
            result = read_npz(self.path('rows'), 'rows')
        elif topic == 'cols':
            result = read_npz(self.path('cols'), 'cols')
        elif topic == 'distances':
            result = read_tensor(self.path('distances'))
        else:
            raise ValueError(f"Unknown topic: {topic}") 
        return result
    
    @functools.cached_property
    def rows(self):
        return self.read('rows')
    
    @functools.cached_property
    def cols(self):
        return self.read('cols')

    @functools.cached_property
    def distances(self):
        return self.read('distances')
    
    @functools.cached_property
    def tensor(self):
        return self.distances
    

class FeaturesPairwiseDistances(Datablock):
    TOPICFILES = {
        "features_shape": "features_shape.npy",
        "n_chunks": "n_chunks.npy",
    }
    @dataclass
    class CONFIG:
        featurebags: FeatureBags
        chunk_size: int = 1024
        row_subsample_fraction: float = 1.0
        col_subsample_fraction: float = 1.0
        seed: int = 42
        sideband_layer: Optional[str] = None

    def __init__(self, *args, n_devices: int = 1, **kwargs):
        super().__init__(*args, n_devices=n_devices, **kwargs)

    def __post_init__(self):
        self.devices = [f"cuda:{i}" for i in range(self.n_devices)]
        return self
        
    def features(self):
        self.log.debug(f"Reading features from {len(self.config.featurebags.bags)} feature bags")
        feature_list = []
        for featurebag in self.config.featurebags.bags:
            feature_list.extend(featurebag.features)
        features = torch.stack(feature_list)
        self.log.debug(f"Combined features: shape: {features.shape}")
        return features

    @functools.cached_property
    def chunks(self):
        return self._chunks(self.features_shape)

    def _chunks(self, shape):
        M, _ = shape
        m = int(M*self.config.row_subsample_fraction)
        n_chunks = math.ceil(m/self.config.chunk_size)
        return [FeaturesPairwiseDistancesChunk(spec=dict(
                    featurebags=self.spec['featurebags'],
                    chunk_idx=i,
                    seed=self.config.seed,
                    row_subsample_fraction=self.config.row_subsample_fraction,
                    col_subsample_fraction=self.config.col_subsample_fraction,
                    sideband_layer=self.config.sideband_layer,
                    )) 
                for i in range(n_chunks)
            ]
    
    @property
    def n_chunks(self):
        if self.valid():
            self.log.debug(f"Reading n_chunks from {self.path('n_chunks')}")
            return read_tensor(self.path('n_chunks')).item()
        else:
            self.log.debug(f"Calculating n_chunks")
            return len(self.chunks)
    
    @functools.cached_property
    def features_shape(self):
        if self.valid():
            self.log.debug(f"Reading features_shape from {self.path('features_shape')}")
            return read_tensor(self.path('features_shape'))
        else:
            self.log.debug(f"Calculating features_shape")
            return self.features().shape

    def __build__(self):
        self.log.debug("Retrieving features")
        features = self.features()
        chunks = self._chunks(features.shape)
        self.log.debug(f"Forming FeaturesPairwiseDistancesChunks from features of shape {features.shape}, chunk_size {self.config.chunk_size}, "
                       f"row_subsample_fraction {self.config.row_subsample_fraction}, col_subsample_fraction {self.config.col_subsample_fraction}"
        )
        self.log.debug(f"Formed {len(chunks)} FeaturesPairwiseDistancesChunks.  Looking for missing chunks")
        missing_chunks = [chunk for chunk in chunks if not chunk.valid()]
        self.log.debug(f"Found {len(missing_chunks)} missing chunks")
        self.log.debug(f"Building all missing pairwise feature distance chunks")
        built_chunks = TorchMultithreadingDatashardBatchBuilder(devices=self.devices, log=self.log).build_shards(missing_chunks, features)
        self.log.verbose(f"Built all missing pairwise feature distance chunks: {len(built_chunks)}")
        write_tensor(torch.tensor([len(chunks)]), self.path('n_chunks', ensure_dirpath=True))
        write_tensor(torch.tensor(features.shape), self.path('features_shape', ensure_dirpath=True))
        return self
    
    def __read__(self, topic):
        if topic == 'features_shape':
            result = read_tensor(self.path('features_shape'))
        elif topic == 'n_chunks':
            result = read_tensor(self.path('n_chunks')).item()   
        return result
    

class FeaturesSortedDistancesChunk(Datablock):
    VERSION = 3
    TOPICFILES = {"rows": "rows.npz",
                  "cols": "cols.npz",
                  "original_order_indices": "original_order_indices.npy",
    }
    
    @dataclass
    class CONFIG:
        distchunk: FeaturesPairwiseDistancesChunk
        row_subsample_fraction: float = 1.0
        col_subsample_fraction: float = 1.0
        seed: int = 42

    def __build__(self):
        _chunk = self.config.distchunk.distances.to(self.device)
        #
        rng = np.random.default_rng(self.config.seed)
        if self.config.row_subsample_fraction < 1.0:
            row_subsample_permutation = rng.permutation(_chunk.shape[0])
            row_subsample_indices = row_subsample_permutation[:int(_chunk.shape[0]*self.config.row_subsample_fraction)]
            _chunk_ = _chunk[row_subsample_indices, :]
            self.log.debug(f"Subsampled tensor rows down to shape {_chunk_.shape} on device {self.device}")
        else:
            _chunk_ = _chunk
            row_subsample_indices = torch.arange(_chunk.shape[0])
        del _chunk
        write_npz(self.path('rows', ensure_dirpath=True), rows=row_subsample_indices)
        del row_subsample_indices
        #
        if self.config.col_subsample_fraction < 1.0:
            col_subsample_permutation = rng.permutation(_chunk_.shape[1])
            col_subsample_indices = col_subsample_permutation[:int(_chunk_.shape[1]*self.config.col_subsample_fraction)]
            chunk = _chunk_[:, col_subsample_indices]
            self.log.debug(f"Subsampled tensor cols down to shape {chunk.shape} on device {self.device}")
        else:
            chunk = _chunk_
            col_subsample_indices = torch.arange(_chunk_.shape[1])
        del _chunk_
        write_npz(self.path('cols', ensure_dirpath=True), cols=col_subsample_indices)
        del col_subsample_indices
        gc.collect()
        
        subsampled = "subsampled " if self.config.row_subsample_fraction < 1.0 or self.config.col_subsample_fraction < 1.0 else ''
        self.log.debug(f"Sorting pairwise distances in {subsampled}FeaturePairwiseDistancesChunk {self.config.distchunk.hashpath()} of shape {chunk.shape} on device {self.device}")
        sorted_chunk, original_order_indices = torch.sort(chunk, dim=-1, descending=False)
        write_tensor(original_order_indices.to('cpu'), self.path('original_order_indices', ensure_dirpath=True))
        del chunk
        del sorted_chunk
        del original_order_indices
        gc.collect()
        self.log.debug(f"Built FeaturesSortedDistancesChunk {self.hashpath()} on device {self.device}")
        return self
    
    def __read__(self, topic):
        if topic == 'rows':
            result = read_npz(self.path('rows'), 'rows')
        elif topic == 'cols':
            result = read_npz(self.path('cols'), 'cols')
        elif topic == 'original_order_indices':
            result = read_tensor(self.path(topic))
        else:
            raise ValueError(f"Unknown topic: {topic}")
        return result
    
    @functools.cached_property
    def rows(self):
        return self.read('rows')
    
    @functools.cached_property
    def cols(self):
        return self.read('cols')
    
    @functools.cached_property
    def original_order_indices(self):
        return self.read('original_order_indices')

    @functools.cached_property
    def tensor(self):
        _chunk = self.config.distchunk.read()
        _tensor = torch.squeeze(_chunk[self.rows, :])
        _tensor_ = torch.squeeze(_tensor[:, self.cols])
        #
        tensor = torch.zeros_like(_tensor_)
        for i in range(_tensor.shape[0]):
            row = _tensor_[i, self.original_order_indices[i,:]]
            tensor[i, :] = row
        return tensor
        

class FeaturesSortedDistances(Datablock):
    VERSION = 3
    TOPICFILES = {"chunk_indices": "chunk_indices.npz"}

    @dataclass
    class CONFIG:
        features_pairwise_distances: FeaturesPairwiseDistances
        max_n_chunks: int = None
        row_subsample_fraction: float = 1.0
        col_subsample_fraction: float = 1.0
        chunk_seed: int = 42
        seed: Optional[int] = None

    def __init__(self, *args, n_workers: int = 1, use_gpus: bool = False, **kwargs):
        super().__init__(*args, n_workers=n_workers, use_gpus=use_gpus, **kwargs)

    def __post_init__(self):
        self.devices = [f"cuda:{i}" if self.use_gpus else f"cpu" for i in range(self.n_workers)]
        return self

    def __build__(self):
        select_sorteddist_chunks = self.chunks
        chunk_indices = self.chunk_indices
        self.log.debug(f"Selected {len(select_sorteddist_chunks)} FeaturesSortedDistancesChunks")
        #
        missing_sorteddist_chunks = [chunk for chunk in select_sorteddist_chunks if not chunk.valid()]
        self.log.debug(f"Found among them {len(missing_sorteddist_chunks)} missing FeaturesSortedDistancesChunks")
        self.log.debug(f"Building {len(missing_sorteddist_chunks)} FeaturesSortedDistancesChunks")
        built_sorteddist_chunks = TorchMultiprocessingDatashardBatchBuilder(devices=self.devices, log=self.log).build_shards(missing_sorteddist_chunks)
        write_npz(self.path('chunk_indices', ensure_dirpath=True), chunk_indices=chunk_indices)
        self.log.verbose(f"Built all missing FeaturesSortedDistancesChunks: {len(built_sorteddist_chunks)}")
        return self
    
    def __read__(self, topic):
        if topic == 'chunk_indices':
            result = read_npz(self.path('chunk_indices'), 'chunk_indices')[0]
        return result
    
    @functools.cached_property
    def chunks(self):
        distchunks = self.config.features_pairwise_distances.chunks
        chunk_indices = self.chunk_indices
        distchunks = [distchunks[i] for i in chunk_indices]
        select_sorteddist_chunks = [FeaturesSortedDistancesChunk(
                                spec=dict(distchunk=distchunk, 
                                          row_subsample_fraction=self.config.row_subsample_fraction, 
                                          col_subsample_fraction=self.config.col_subsample_fraction),
                                          seed=self.config.chunk_seed,
                                ) 
                            for distchunk in distchunks
        ]
        return select_sorteddist_chunks
    
    @functools.cached_property
    def chunk_indices(self):
        if self.valid():
            self.log.debug(f"Reading chunks_indices from {self.path('chunk_indices')}")
            return self.read('chunk_indices')
        else:
            n_chunks = self.config.features_pairwise_distances.n_chunks
            if self.config.seed is not None:
                rng = np.random.default_rng(self.config.seed)
                permutation = rng.permutation(n_chunks)
            else:
                permutation = np.arange(n_chunks)
            max_n_chunks = self.config.max_n_chunks if self.config.max_n_chunks is not None else n_chunks
            return permutation[:max_n_chunks]


class Features2NNDistancesChunk(Datablock):
    TOPICFILE = "twonn_distances.pt"
    @dataclass
    class CONFIG:
        sorted_distchunk: FeaturesSortedDistancesChunk

    def __build__(self):
        distchunk = self.config.sorted_distchunk
        self.log.debug(f"Locating smallest nonzero pairwise distances in FeatureSortedDistancesChunk {self.config.sorted_distchunk.hashpath()} of shape {sorted_distchunk_tensor.shape}")
        firstdist = sorted_distchunk_tensor[:, 0]
        if self.debug:
            firstdist_min, firstdist_max = firstdist.values.min(), firstdist.values.max()
            self.log.debug(f"firstdist_min: {firstdist_min}, firstdist_max: {firstdist_max}")
        self.log.debug(f"Locating second smallest pairwise distances in FeaturePairwiseDistancesChunk {self.config.featuredist_chunk.hashpath()} of shape {featuredist_chunk_tensor.shape}")
        seconddist = sorted_distchunk_tensor[:, 1]
        if self.debug:
            seconddist_min, seconddist_max = seconddist.values.min(), seconddist.values.max()
            self.log.debug(f"seconddist_min: {seconddist_min}, seconddist_max: {seconddist_max}")
        del sorted_distchunk_tensor
        twonn_distances = torch.stack([firstdist.values, seconddist.values], dim=-1)
        self.log.debug(f"Built Features2NNDistanceChunk {self.hashpath()} at offset {self.config.dist_chunk.config.row_batch_offset} with shape {twonn_distances.shape}")
        write_tensor(twonn_distances, self.path(ensure_dirpath=True))
        return self
    
    def __read__(self):
        result = read_tensor(self.path())
        return result
    
    @functools.cached_property
    def tensor(self):
        return self.read()
    

class Features2NNDim(Datablock):
    TOPICFILES = {
        "twonn_distances": "twonn_distances.pt", 
        "dimension": "dimension.pt",
        "model": "model.pkl",
    }
    @dataclass
    class CONFIG:
        features_sorted_distances: FeaturesSortedDistances

    def __init__(self, *args, n_workers: int = 1, **kwargs):
        super().__init__(*args, n_workers=n_workers, **kwargs)

    def __build__(self):
        sorted_distchunks = self.config.features_sorted_distances.chunks
        twonndist_chunks = [Features2NNDistancesChunk(spec=dict(featuredist_chunk=sorted_distchunk)) 
                            for sorted_distchunk in sorted_distchunks
        ]
        self.log.debug(f"Formed {len(twonndist_chunks)} Features2NNDistancesChunks")
        missing_twonndist_chunks = [chunk for chunk in twonndist_chunks if not chunk.valid()]
        self.log.debug(f"Found {len(missing_twonndist_chunks)} missing Features2NNDistancesChunks")
        self.log.debug(f"Building {len(missing_twonndist_chunks)} Features2NNDistancesChunks")
        built_twonndist_chunks = TorchMultiprocessingDatashardBatchBuilder(n_workers=self.n_workers, log=self.log).build_shards(missing_twonndist_chunks)
        self.log.verbose(f"Built all missing Features2NNDistancesChunks: {len(built_twonndist_chunks)}")
        twonndists = torch.cat([chunk.tensor for chunk in twonndist_chunks], dim=0)
        mus = twonndists[:, 1]/twonndists[:, 0]
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
                            




