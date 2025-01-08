"""
    > Background

        Contains code applying prov-gigapath foundation histopathology model to TCGA pancancer WSI data.
        The (prov-)gigapath model is described in https://www.nature.com/articles/s41586-024-07441-w
        Xu, H., Usuyama, N., Bagga, J. et al. "A whole-slide foundation model for digital pathology from real-world data.",
         Nature 630, 181–188 (2024). https://doi.org/10.1038/s41586-024-07441-w
        Code is available at https://github.com/prov-gigapath/prov-gigapath, 
        pretrained model can be obtained from HuggingFace: https://huggingface.co/prov-gigapath/prov-gigapath.

    > rhubarb
        * cluster info
        ```
            # basis capabilities:
            scontrol show nodes
            # current state
            sinfo -N -l
            # jobs
            squeue
        ```
        * start an interactive job:
        ```
            # alloc
            salloc --nodelist rhubarb -N 1 -n 1 --mem=64G
            # job params:
            env | grep SLURM_
            env | grep SLURM_JOBID
            # start interactive shell:
            srun --jobid=$SLURM_JOBID --pty bash

            # on rhubarb: BEGIN
            nvidia-smi
            htop
            conda activate pancan-gigapath
            # ...
            exit
            # on rhubarb: END

            scancel $SLURM_JOBID
    > SETUP:
    ```
        export SEYE=$HOME/seye
        export DATALAKE="/mnt/labshare/PANCAN-GIGAPATH"
        export VERSION=v1
    ```
    > GIGAPATH-PANCAN
        * ALL
            ```
                dbx.print "DBX.show_datablocks()
            ```
        * CPTAC
            ```
                export CUDA_VISIBLE_DEVICES=0,1,2,3
                export CPIMGS="DBX('seye.gigapath.IMGS', 'CPIMGS', repo='${SEYE}', revision='gigapath/pancan/${VERSION}').SCOPE(slides=DBX.Path('/mnt/labshare/SLIDES/CPTAC_downloads'), origin='CPTAC', tile_px=256, tile_um=256)"
                dbx.print "$CPIMGS.IMGS(verbose=True).Databuilder(throw=True).intent"
                dbx.print "$CPIMGS.IMGS(verbose=True).Databuilder(throw=True).extent"
                dbx.print "$CPIMGS.IMGS(verbose=True).Databuilder(throw=True).build()"
                dbx.print "$CPIMGS.IMGS(verbose=True).Databuilder(throw=True).show_records()"
                dbx.print "$CPIMGS.read('project')"
                dbx.print "$CPIMGS.read('dataset')"
                dbx.print "$CPIMGS.read('dataset').summary()"
                dbx.print "$CPIMGS.read('dataset').manifest()"
                dbx.print "$CPIMGS.read('slides')"
                dbx.print "$CPIMGS.read('slide_path_map')"
                dbx.print "$CPIMGS.read('slide_source_map')"
                dbx.print "$CPIMGS.read('slide_slide_map')"
                dbx.print "$CPIMGS.read('slide_paths')"
                dbx.print "$CPIMGS.read('slide_sources')"
                dbx.print "$CPIMGS.read('tile_paths')"

                export CPBAGS="DBX('seye.gigapath.BAGS', 'CPBAGS', repo='${SEYE}', revision='gigapath/pancan/${VERSION}').SCOPE(dataset=$CPIMGS.READ('dataset'))"
                dbx.print "$CPBAGS.BAGS(num_gpus=4, verbose=True).Databuilder(throw=True).build()"

                export CPEMBS="DBX('seye.gigapath.EMBS', 'CPEMBS', repo='${SEYE}', revision='gigapath/pancan/${VERSION}').SCOPE(slides=$CPIMGS.READ('slides'), bags=$CPBAGS.READ())"
                dbx.print "$CPEMBS.EMBS(verbose=True).Databuilder(throw=True).build()"
                dbx.print "$CPEMBS.EMBS(verbose=True).Databuilder(throw=True).read()"

                export CPEVAL="DBX('seye.gigapath.DEVALUATOR', 'CPEVAL', repo='${SEYE}', revision='gigapath/pancan/${VERSION}').SCOPE(features=$CPEMBS.READ(), slide_sources=$CPIMGS.READ('slide_sources'), n_bins=4)"
                dbx.print "$CPEVAL.DEVAL(verbose=True).Databuilder(throw=True).register()"
                dbx.print "$CPEVAL.DEVAL(verbose=True).Databuilder(throw=True).build()"

                #dbx.print "DBX.Transcribe($CPIMGS, $CPBAGS, $CPEMBS, with_build=True)"
            ```
        * TCGA
            ```
                export CUDA_VISIBLE_DEVICES=0,1,2,3
                export TCIMGS="DBX('seye.gigapath.IMGS', 'TCIMGS', repo='${SEYE}', revision='gigapath/pancan/${VERSION}').SCOPE(slides=DBX.Path('/mnt/labshare/SLIDES'), origin='TCGA', tile_px=256, tile_um=256)"
                dbx.print "$TCIMGS.IMGS(verbose=True).Databuilder(throw=True).intent"
                dbx.print "$TCIMGS.IMGS(verbose=True).Databuilder(throw=True).extent"
                dbx.print "$TCIMGS.IMGS(verbose=True).Databuilder(throw=True).build()"
                dbx.print "$TCIMGS.read('project')"
                dbx.print "$TCIMGS.read('dataset')"
                dbx.print "$TCIMGS.read('dataset').summary()"
                dbx.print "$TCIMGS.read('dataset').manifest()"
                dbx.print "$TCIMGS.read('slides')"

                export TCBAGS="DBX('seye.gigapath.BAGS', 'TCBAGS', repo='${SEYE}', revision='gigapath/pancan/${VERSION}').SCOPE(dataset=$TCIMGS.READ('dataset'))"
                dbx.print "$TCBAGS.BAGS(num_gpus=4, verbose=True).Databuilder(throw=True).intent"
                dbx.print "$TCBAGS.BAGS(num_gpus=4, verbose=True).Databuilder(throw=True).extent"
                dbx.print "$TCBAGS.BAGS(num_gpus=4, verbose=True).Databuilder(throw=True).build()"

                export TCEMBS="DBX('seye.gigapath.EMBS', 'TCEMBS', repo='${SEYE}', revision='gigapath/pancan/${VERSION}').SCOPE(slides=$TCIMGS.READ('slides'), bags=$TCBAGS.READ())"
                dbx.print "$TCEMBS.EMBS(verbose=True).Databuilder(throw=True).build()"
                dbx.print "$TCEMBS.EMBS(verbose=True).Databuilder(throw=True).read()"

                export TCEVAL="DBX('seye.gigapath.DEVALUATOR', 'TCEVAL', repo='${SEYE}', revision='gigapath/pancan/${VERSION}').SCOPE(features=$TCEMBS.READ(), slide_sources=$TCIMGS.READ('slide_sources'), n_bins=4)"
                dbx.print "$TCEVAL.DEVAL(verbose=True).Databuilder(throw=True).register()"
                dbx.print "$TCEVAL.DEVAL(verbose=True).Databuilder(throw=True).build()"

                #dbx.print "DBX.Transcribe($TCIMGS, $TCBAGS, $TCEMBS, with_build=True)"
            ```
"""


import argparse
from dataclasses import dataclass, asdict
import datetime
import json
import math
import os
import pdb #DEBUG
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


import ray
import torch

from sklearn.metrics import accuracy_score, f1_score, classification_report
from sklearn.linear_model import LogisticRegression

import slideflow as sf

from datablocks.dbx import Datablock


class Evaluator:
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
                         y: Union[np.ndarray, list, torch.Tensor, pd.DataFrame], 
                         train_fraction:float=0.8
    ):
        #TODO: split_slides_labels_train_test() -> split_features_labels_train_test()
        X = Evaluator.ndarray(X)
        y = Evaluator.ndarray(y)
        N = y.shape[0]
        permutation = permutation = np.random.permutation(range(N))
        n = int(math.floor(N*train_fraction))
        train = list(permutation[:n])
        test = list(permutation[n:])
        return (X[train, :], y[train]), (X[test, :], y[test])

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


class IMGS(Datablock):
    @dataclass
    class SCOPE:
        tile_px: int
        tile_um: int
        slides: str
        origin: str = 'TCGA'

    TOPICS = {
              'project': 'settings.json',
              'dataset': '.dataset',
              'slides': 'slides.json',
              'slide_path_map': 'slide_path_map.json',
              'slide_source_map': 'slide_source_map.json',
              'source_slides_map': 'source_slides_map.json',
              'slide_paths': '.slide_paths',
              'slide_sources': '.slide_sources',
              'tile_paths': 'tile_paths.json',
    }

    def __init__(self, *args, force_extract: bool = False, **kwargs):
        super().__init__(*args, **kwargs)
        self.force_extract = force_extract

    def slide_paths(self, scope, roots):
        dataset = self.dataset(scope, roots)
        # Iterate through each slide.
        slide_paths = list(dataset.slide_paths())
        
        if self.verbose:
            print(f"Obtained {len(slide_paths)} paths")
        return slide_paths

    def project(self, scope, roots):
        #DEBUG
        #pdb.set_trace()
        return sf.Project(roots['project'])

    def dataset(self, scope, roots):
        return self.project(scope, roots).dataset(tile_px=scope.tile_px, tile_um=scope.tile_um)

    def build(self, scope, roots):
        root = roots['project']
        if not self.filesystem.isfile(os.path.join(root, 'settings.json')) or \
            not self.filesystem.isfile(os.path.join(root, 'datasets.json')) or \
            not self.filesystem.isfile(os.path.join(root, 'annotations.csv')):
            sf.create_project(root=root)
            if scope.origin == 'TCGA':
                slide_folders = {os.path.basename(d): d for d in self.filesystem.ls(scope.slides)}
                slide_paths = {
                    basename: path for basename, path in slide_folders.items() 
                        if basename.startswith('TCGA-') and basename.find('test') == -1
                }
            elif scope.origin == 'CPTAC':
                if self.debug:
                    print(f"DEBUG: IMGS: build: building project from scope.slides: {scope.slides}") 
                slide_folders = {os.path.basename(d): d for d in self.filesystem.ls(scope.slides)}
                if self.debug:
                    print(f"DEBUG: IMGS: build: computed slide_folders: {slide_folders}")
                slide_paths = {}
                for basename, path in slide_folders.items():
                    images = os.path.join(path, 'images')
                    #gcs will return True to isdir(path) for any extant path, it seems
                    if self.filesystem.exists(images) and not self.filesystem.isfile(images) and\
                        basename.find('TEST') == -1:
                        slide_paths[basename] = path
                if self.debug:
                    print(f"DEBUG: IMGS: build: computed slide_paths: {slide_paths}")
            else:
                raise ValueError(f"Unknown slide origin: '{scope.origin}'")
            datasets = {
                    basename: {'slides': os.path.join(path, 'images'),
                                'tfrecords': os.path.join(path, 'tfrecords'),
                                'roi': os.path.join(path, 'ROI')
                            } 
                            for basename, path in slide_paths.items()
            }
            datasets_path = os.path.join(root, 'datasets.json')
            with self.filesystem.open(datasets_path, 'w') as f:
                json.dump(datasets, f)
            if self.debug:
                print(f"DEBUG: IMGS: build: wrote datasets to path {datasets_path}:\n{datasets}")
            settings_path = os.path.join(root, 'settings.json')
            with self.filesystem.open(settings_path, 'r') as f:
                settings = json.load(f)
            settings['name'] = "gigapath-pancan"
            # insert `list(datasets.keys())` into {PROJECT}/settings.json:sources. 
            # remember to double-quote all source names
            settings['sources'] = list(datasets.keys())
            with self.filesystem.open(settings_path, 'w') as f:
                json.dump(settings, f)
            if self.debug:
                print(f"DEBUG: IMGS: build: wrote settings to path {settings_path}:\n{datasets}")
            # if necessary, generate blank annotations
            #DEBUG
            #pdb.set_trace()
            annf = os.path.join(root, 'annotations.csv')
            if not self.filesystem.isfile(annf):
                #self.filesystem.rm(annf)
                self.project(scope, roots).create_blank_annotations()
            #HACK: eliminate duplicate patient/slide
            with self.filesystem.open(os.path.join(root, 'annotations.csv'), 'r') as f:
                ann = pd.read_csv(f)
                ann = ann.set_index('patient').groupby(level=0).last().reset_index()
            with self.filesystem.open(annf, 'w') as f:
                ann.to_csv(f)

        '''
              'project': 'settings.json',
              'dataset': '.dataset',
              'slides': 'slides.json',
              'slide_path_map': 'slide_path_map.json',
              'slide_source_map': 'slide_source_map.json',
              'source_slide_map': 'source_slide_map.json',
              'slide_paths': '.slide_paths',
              'slide_sources': '.slide_sources',
              'tile_paths': 'tile_paths.json',
        '''

        if (not self.valid(scope, roots, 'slides')           or
            not self.valid(scope, roots, 'slide_path_map')   or 
            not self.valid(scope, roots, 'slide_source_map') or 
            not self.valid(scope, roots, 'source_slides_map') or
            not self.valid(scope, roots, 'slide_paths')      or
            not self.valid(scope, roots, 'slide_sources')    or
            not self.valid(scope, roots, 'tile_paths')
        ):
            if self.verbose:
                print(f"IMGS: build: generating slides, slide_path_map, slide_source_map, source_slides_map: BEGIN: {datetime.datetime.now()}")
            slide_source_map = {}
            source_slides_map = {}
            slide_path_map = {}
            sources = self.dataset(scope, roots).sources
            for source, paths in sources.items():
                for path in self.filesystem.ls(paths['slides']):
                    basename = os.path.basename(path)
                    slide, ext = os.path.splitext(basename)
                    if ext == '.svs': # exclude . and ..
                        slide_path_map[slide] = path
                        slide_source_map[slide] = source
                        if source not in source_slides_map:
                            source_slides_map[source] = [slide]
                        else:
                            source_slides_map[source].append(slide)
            slides = list(slide_path_map.keys())
            with self.filesystem.open(self.path(scope, roots, 'slides'), 'w') as f:
                json.dump(slides, f)
            with self.filesystem.open(self.path(scope, roots, 'slide_path_map'), 'w') as f:
                json.dump(slide_path_map, f)
            with self.filesystem.open(self.path(scope, roots, 'slide_source_map'), 'w') as f:
                json.dump(slide_source_map, f)
            with self.filesystem.open(self.path(scope, roots, 'source_slides_map'), 'w') as f:
                json.dump(source_slides_map, f)
            if self.verbose:
                    print(f"IMGS: build: generating slide_source_map: finished at {datetime.datetime.now()}")
            if self.verbose:
                print(f"IMGS: build: generating slides, slide_path_map, slide_source_map, source_slides_map: END: {datetime.datetime.now()}")
        if self.verbose:
            print(f"TILES: Extracting tiles: BEING: {datetime.datetime.now()}")
        self.dataset(scope, roots).extract_tiles(skip_extracted=not self.force_extract)
        if self.verbose:
            print(f"TILES: Extracting tiles: END: {datetime.datetime.now()}")
        
    def read(self, scope, roots, topic):
        if topic not in self.TOPICS:
            raise ValueError(f"Unknown topic {repr(topic)}: expected one of {self.TOPICS}")
        
        if topic == 'project':
            return self.project(scope, roots)

        if topic == 'dataset':
            return self.dataset(scope, roots)

        if topic == 'slide_source_map':
            with self.filesystem.open(self.path(scope, roots, 'slide_source_map'), 'r') as f:
                return json.load(f)

        if topic == 'source_slides_map':
            with self.filesystem.open(self.path(scope, roots, 'source_slides_map'), 'r') as f:
                return json.load(f)
    
        if topic == 'slide_path_map':
            with self.filesystem.open(self.path(scope, roots, 'slide_path_map'), 'r') as f:
                return json.load(f)
    
        if topic == 'slides':
            with self.filesystem.open(self.path(scope, roots, 'slides'), 'r') as f:
                return json.load(f)

        if topic == 'slide_paths':
            return list(self.read(scope, roots, 'slide_paths').values())

        if topic == 'slide_sources':
            return list(self.read(scope, roots, 'slide_sources').values())

        if topic == 'tiles':
            #TODO: #FIX: return .tfrecords paths
            return None


class BAGS(Datablock):
    @dataclass
    class SCOPE:
        dataset: sf.Dataset

    PATHNAME: str = ".bags"

    def __init__(self,
        filesystem: fsspec.AbstractFileSystem = fsspec.filesystem("file"),
        *,
        hf_token: str = "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",
        cache: Optional[str] = None,
        tile_encoder_snapshot: str = "8d2b1d2e65832e16bf9ff100a081acf6170a44ca",
        force_regenerate: bool = False,
        num_gpus: int = 1,
        verbose: bool = False,
        debug: bool = False,
    ):
        self.filesystem = filesystem
    
        self.hf_token = hf_token
        self.cache = cache
        if self.cache is None:
            self.cache = os.path.join(os.environ['HOME'], '.cache')
        self.tile_encoder_snapshot = tile_encoder_snapshot
        self.force_regenerate = force_regenerate
        self.num_gpus = num_gpus
        self.verbose = verbose
        self.debug = debug
        
        self._tile_encoder = None

    @property
    def tile_encoder(self):
        if self._tile_encoder is None:
            # Load gigapath
            os.environ['HF_TOKEN'] = self.hf_token
            if self.verbose:
                print(f"Building gigapath tile feature extractor: started at {datetime.datetime.now()}")
            extractor = sf.build_feature_extractor(
                'gigapath.tile',
                weights=f"{self.cache}/huggingface/hub/models--prov-gigapath--prov-gigapath/snapshots/{self.tile_encoder_snapshot}/pytorch_model.bin",
            )
            if self.verbose:
                print(f"Building gigapath tile feature extractor: finished at {datetime.datetime.now()}")
            self._tile_encoder = extractor
        return self._tile_encoder

    def build(self, scope, roots):
        bags_path = self.path(scope, roots)
        kw = {'force_regenerate': self.force_regenerate}
        if self.num_gpus is not None:
            kw['num_gpus'] = self.num_gpus
        if self.verbose:
            print(f"Generating feature bags to {bags_path} with kwargs {kw}: started at {datetime.datetime.now()}")
        scope.dataset.generate_feature_bags(self.tile_encoder, outdir=bags_path, **kw)
        if self.verbose:
            print(f"Generating feature bags to {bags_path}: finished at {datetime.datetime.now()}")  

    def read(self, scope, roots):
        bags = {}
        bags_path = self.path(scope, roots)
        paths = self.filesystem.ls(bags_path)
        ptpaths = [path for path in paths if path.endswith('.pt')]
        for path in ptpaths:
            basename = os.path.basename(path)
            slide, ext = os.path.splitext(basename)
            bags[slide] = path
        return bags


class EMBS(Datablock):
    @dataclass
    class SCOPE:
        bags: list[str]
        slides: list[str]

    PATHNAME: str = ".embs"

    def __init__(self,
        filesystem: fsspec.AbstractFileSystem = fsspec.filesystem("file"),
        *,
        hf_token: str = "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",
        cache: Optional[str] = None,
        verbose: bool = False,
        debug: bool = False,
    ):
        self.filesystem = filesystem

        self.hf_token = hf_token
        self.cache = cache
        if self.cache is None:
            self.cache = os.path.join(os.environ['HOME'], '.cache')
        self.verbose = verbose
        self.debug = debug
        
        self._slide_encoder = None

    @property
    def slide_encoder(self):
        if self._slide_encoder is None:
            # Load gigapath
            os.environ['HF_TOKEN'] = self.hf_token
            if self.verbose:
                print(f"Building gigapath slide feature extractor: started at {datetime.datetime.now()}")
            extractor = sf.build_feature_extractor(
                'gigapath.slide',
                weights=f"{self.cache}/slide_encoder.pth",
                global_pool=True
            )
            if self.verbose:
                print(f"Building gigapath slide feature extractor: finished at {datetime.datetime.now()}")
            self._slide_encoder = extractor
        return self._slide_encoder

    def build(self, scope, root):
        bags = list(scope.bags.values())
        embs_path = self.path(scope, root)
        if self.verbose:
            print(f"Generating slide embeddings from {len(bags)} feature bags to {embs_path}: started at {datetime.datetime.now()}")
        self.slide_encoder.generate_and_save(bags, outdir=embs_path)
        if self.verbose:
            print(f"Generating slide embeddings from {len(bags)} feature bags to {embs_path}: finished at {datetime.datetime.now()}")

    def read(self, scope, root):
        #TODO: parallelize :-)
        embtensors = []
        index = []
        for slide in scope.slides:
            try:
                embtensor = torch.load(os.path.join(self.path(scope, root), f"{slide}.pt"))['last_layer_embed']
                embtensors.append(embtensor)
                index.append(slide)
            except:
                pass
        return pd.DataFrame(torch.cat(embtensors), index=index)

    
class DEVALUATOR(Datablock, Evaluator):
    @dataclass
    class SCOPE:
        features: pd.DataFrame
        slide_sources: dict[str, str]
        n_bins: int = 2
        evaluation_fraction: float = 0.8
        umap_fraction: float = 0.01

    TOPICS = {
        'labels': 'pancan_cptac_labels.parquet',
        'discretized_features': 'pancan_cptac_discretized_features.pt',
        'umap': 'pancan_cptac_discretized_features_umap.png',
        'reports': '.evaluation_reports',
    }

    def __init__(self,
        filesystem: fsspec.AbstractFileSystem = fsspec.filesystem("file"),
        *,
        verbose: bool = False,
        debug: bool = False,
    ):
        self.filesystem = filesystem

        self.verbose = verbose
        self.debug = debug
        
    def build(self, scope, roots):
        # labels
        labels = pd.DataFrame({'labels': [scope.slide_sources[s] for s in scope.features.index]})
        labels_path = self.path(scope, roots, 'labels')
        with self.filesystem.open(labels_path, 'wb') as f:
            labels.to_parquet(f)
        if self.verbose:
            print(f"DEVAL: wrote {len(labels)} labels to {labels_path}")

        # discretized_features
        discretized_features = torch.Tensor(self.discretize_features(scope.features, scope.n_bins))
        discretized_features_path = self.path(scope, roots, 'discretized_features')
        with self.filesystem.open(discretized_features_path, 'wb') as f:
            torch.save(discretized_features, f)
        if self.verbose:
            print(f"DEVAL: wrote {len(discretized_features)} discretized_features to {discretized_features_path}")
        
        #TODO: #FIX
        #TODO: check that this file system is local; other are not supported
        '''
        umap_path = self.path(scope, roots, 'umap')
        self.plot_features_umap(
            (discretized_features, labels),
            title="discretized_features", 
            fraction=scope.umap_fraction,
            output_path=umap_path,
            verbose=self.verbose,
        )
        if self.verbose:
            print(f"DEVAL: wrote discretized_features umap plot to {umap_path}")
        '''

    def read(self, scope, roots, topic):
        if topic not in self.TOPICS:
            raise ValueError(f"Unknown topic {repr(topic)}: not in {list(self.TOPICS.keys())}")
        if topic == 'labels':
            labels_path = self.path(scope, roots, 'labels')
            with self.filesystem.open(labels_path, 'rb') as f:
                labels = pd.read_parquet(f)
            result = labels.values
        if topic == 'discretized_features':
            discretized_features_path = self.path(scope, roots, 'discretized_features')
            with self.filesystem.open(discretized_features_path, 'rb') as f:
                discretized_features = torch.load(f).numpy()
            result = discretized_features
        if topic == 'reports':
            discretized_features = self.read(scope, roots, 'discretized_features')
            labels = self.read(scope, roots, 'labels')
            evaluation_reports = self.evaluate_features2(
                (scope.features, labels), 
                (discretized_features, labels),
                fraction=scope.evaluation_fraction,
                label1='continuous',
                label2='discretized',
                verbose=self.verbose,
            )
            result = evaluation_reports
        if topic == 'umap':
            result = self.path(scope, roots, 'umap')
        return result
