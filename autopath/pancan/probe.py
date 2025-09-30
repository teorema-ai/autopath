
from dataclasses import dataclass, asdict
import datetime
import functools
import json
import math
import os
import pickle
import sys
import time
from typing import List, Dict, Optional, Union, Tuple

import fsspec

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import torch

import scipy as sp


from sklearn.metrics import accuracy_score, f1_score, classification_report
from sklearn.linear_model import LogisticRegression

import dbx
from dbx import (
	Logger,
    Databag,
	Datablock,
    write_tensor, 
    read_tensor,
    write_npz,
    read_npz,
    write_pickle,
    read_pickle,
    MultiDeviceDatabagBuilder,
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
    FILES = {
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


class FeatureBagsDimProbe(Datablock, FeatureProbe):
    FILES = {
        'pairwise_distances': 'pairwise_distances.npz',
        'dimension_fit_report': 'dimension_fit_report',
    }
    @dataclass
    class CONFIG:
        featurebags: FeatureBags
        sideband_layer: Optional[str] = None

    class FeatureDistances(Databag):
        def __init__(self, features: torch.Tensor, rows: List[int], log: Logger):
            self.features = features
            self.rows = rows
            self.pairwise_distances = None
            self.log = log
        
        def __str__(self):
            return f"FeatureDistances({self.rows})"

        def to(self, device):
            self.features = self.features.to(device)
            return self

        def build(self):
            if self.pairwise_distances is None:
                self.log.verbose(f"Building FeatureDistances with rows {self.rows} on device {self.features.device}")
                self.pairwise_distances = torch.cdist(self.features[self.rows], self.features)
                self.log.verbose(f"Build FeatureDistances with shape {self.pairwise_distances.shape} on device {self.features.device}")
            return self

    def __init__(self, *args, row_batch_size: int = 1, devices: list[str] = ['cuda:0'], **kwargs):
        super().__init__(*args, row_batch_size=row_batch_size, devices=devices, **kwargs)
        
    @functools.cached_property
    def features(self):
        feature_list = []
        for featurebag in self.config.featurebags.bags:
            feature_list.extend(featurebag.features)
        features = torch.stack(feature_list)
        return features

    def __build__(self):
        rows_bounds = torch.arange(0, len(self.features), self.row_batch_size).tolist()
        rows_list = [list(range(rows_bounds[i], rows_bounds[i+1])) for i in range(len(rows_bounds)-1)]
        feature_distances_list = [self.FeatureDistances(self.features, rows, log=self.log) for rows in rows_list]
        feature_distances_list = MultiDeviceDatabagBuilder(devices=self.devices).build(feature_distances_list)
        pairwise_distances = torch.cat([feature_distances.pairwise_distances for feature_distances in feature_distances_list])
        self.log.verbose(f"Built pairwise_distances with shape: {pairwise_distances.shape}")
        write_tensor(pairwise_distances, self.path('pairwise_distances', ensure_dirpath=True))
        return self
    
    def __read__(self, topic):
        if topic == 'pairwise_distances':
            result = read_tensor(self.path('pairwise_distances'))
        elif topic == 'dimension_fit_report':
            result = read_pickle(self.path('dimension_fit_report'))
        return result
