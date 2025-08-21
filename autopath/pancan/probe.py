
from dataclasses import dataclass, asdict
import datetime
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

import scipy as sp


from sklearn.metrics import accuracy_score, f1_score, classification_report
from sklearn.linear_model import LogisticRegression

DBKSPACE = os.environ.get("DBKSPACE", "/mnt/labshare/PROJECTS/GIGAQ/dbx")
DBKREPO = os.environ.get("DBKREPO", f"{os.environ.get('HOME')}/autopath")

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
        features = Evaluator.ndarray(X)
        return pd.DataFrame(features).apply(lambda c: pd.qcut(c, d, labels=False, duplicates='drop')).fillna(0.0).values

    @staticmethod
    def split_train_test(X: Union[np.ndarray, list, torch.Tensor, pd.DataFrame], 
                         y: Optional[Union[np.ndarray, list, torch.Tensor, pd.DataFrame]] = None, 
                         train_fraction:float=0.8
    ):
        #TODO: split_slides_labels_train_test() -> split_features_labels_train_test()
        X = Evaluator.ndarray(X)
        if y is not None:
            y = Evaluator.ndarray(y)
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
        features = Evaluator.ndarray(features)
        labels = Evaluator.ndarray(labels)
        N = len(labels)
        ntrain = int(N*fraction)
        randomidx = np.random.permutation(list(range(N)))
        trainidx = randomidx[:ntrain]
        testidx = randomidx[ntrain:]
       
        X_train = features[trainidx, :]
        y_train = labels[trainidx, :]

        X_test = features[testidx, :]
        y_test = labels[testidx, :]

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
                           verbose=False,
    ):
        if verbose:
            print(f"Evaluating features {label1}: started at {datetime.datetime.now()}")
        report1 = Evaluator.evaluate_features(Xy1, fraction=fraction)
        if verbose:
            print(f"Evaluating features {label1}: finished at {datetime.datetime.now()}")
        if verbose:
            print(f"Evaluating features {label2}: started at {datetime.datetime.now()}")
        report2 = Evaluator.evaluate_features(Xy2, fraction=fraction)
        if verbose:
            print(f"Evaluating features {label2}: finished at {datetime.datetime.now()}")

        if verbose:
            print(f"---------- {label1} ------------")
            print(report1)
            print(f"---------- {label2} ------------")
            print(report2)
        return report1, report2

    def plot_features_umap(self,
                           Xy,
                           *, 
                           fraction = 0.01, 
                           title="", 
                           label="",
                           use_umap_plot=False,
                           output_path=None,
                           verbose=False,
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
            if verbose:
                print(f"Wrote figure to '{output_path}'")
        else:
            plt.show()
        return umap_features


#>>> #TODO: #TEST
class DiscreteProbe(Datablock, FeatureProbe):
    TOPICS = {
        'labels': 'labels.parquet',
        'cdf': 'cdf.pt',
        'discretized_features': 'discretized_features.pt',
        'cdf_umap': 'cdf_umap.png',
        'discretized_features_umap': 'discretized_features_umap.png',
        'evaluation_reports': 'evaluation_reports.pkl',
    }
    @dataclass
    class CONFIG:
        features: pd.DataFrame
        slide_sources: dict[str, str]
        n_bins: int = 2
        evaluation_fraction: float = 0.8
        umap_fraction: float = 0.01

    def build(self):
        # labels
        labels = pd.DataFrame({'labels': [scope.slide_sources[s] for s in scope.features.index]})
        labels_path = self.path('labels')
        labels_fs, _ = fsspec.url_to_fs(labels_path)
        with labels_fs.open(labels_path, 'wb') as f:
            labels.to_parquet(f)
        self.log.verbose(f"build: wrote {len(labels)} labels to {labels_path}")

        # cdf
        quantiles = np.arange(0.0, 1.0, 1.0/scope.n_bins)
        cdf = torch.tensor(np.percentile(scope.features, quantiles, axis=0))
        cdf_path = self.path('cdf')
        cdf_fs, _ = fsspec.url_to_fs(cdf_path)
        with cdf_fs.open(cdf_path, 'wb') as f:
            torch.save(cdf, f)
        self.log.verbose(f"build: wrote cdf Tensor of shape {cdf.shape} to {cdf_path}")

        # discretized_features
        discretized_features = torch.Tensor(self.discretize_features(scope.features, scope.n_bins))
        discretized_features_path = self.path(scope, roots, 'discretized_features')
        discretized_features_fs, _ = fsspec.url_to_fs(discretized_features_path)
        with discretized_features_fs.open(discretized_features_path, 'wb') as f:
            torch.save(discretized_features, f)
        self.log.verbose(f"build: wrote {len(discretized_features)} discretized_features to {discretized_features_path}")
        
        # evaluation_reports
        continuous, discretized = self.evaluate_features2(
            (scope.features, labels), 
            (discretized_features, labels),
            fraction=scope.evaluation_fraction,
            label1='continuous',
            label2='discretized',
            verbose=self.verbose,
        )
        evaluation_reports = {'continuous': continuous, 'discretized': discretized}
        evaluation_reports_path = self.path(scope, roots, 'evaluation_reports')
        evaluation_reports_fs, _ = fsspec.url_to_fs(evaluation_reports_path)
        with evaluation_reports_fs.open(evaluation_reports_path, 'wb') as f:
            pickle.dump(evaluation_reports, f)

    def read(self, topic):
        if topic not in self.TOPICS:
            raise ValueError(f"Unknown topic {repr(topic)}: not in {list(self.TOPICS.keys())}")
        if topic == 'labels':
            labels_path = self.path('labels')
            labels_fs, _ = fsspec.url_to_fs(labels_path)
            with labels_fs.open(labels_path, 'rb') as f:
                labels = pd.read_parquet(f)
            result = labels.values
        if topic == 'discretized_features':
            discretized_features_path = self.path('discretized_features')
            discretized_features_fs, _ = fsspec.url_to_fs(discretized_features_path)
            with discretized_features_fs.open(discretized_features_path, 'rb') as f:
                discretized_features = torch.load(f).numpy()
            result = discretized_features
        if topic == 'cdf':
            cdf_path = self.path('cdf')
            cdf_fs, _ = fsspec.url_to_fs(cdf_path)
            with cdf_fs.open(cdf_path, 'rb') as f:
                cdf = torch.load(f).numpy()
            result = cdf
        if topic == 'evaluation_reports':
            evaluation_reports_path = self.path('evaluation_reports')
            evaluation_reports_fs, _ = fsspec.url_to_fs(evaluation_reports_path)
            with evaluation_reports_fs.open(evaluation_reports_path, 'rb') as f:
                evaluation_reports = pickle.load(f)
            result = evaluation_reports
        return result
