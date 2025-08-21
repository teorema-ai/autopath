"""
    Examples:
        #BASH:
            dbx "autopath.pancan.bags.FeatureBag(batch_size=16, device='cuda', verbose=True, debug=True, cfg=dict(slideshard='@autopath.pancan.tiles.PancanSlideShard()', extractor='@autopath.gigaq.dinov2.models.BackboneEvaluator()', split='test')).build().read()"
        #PYTHON:
            import autopath.pancan.bags;featurebag = autopath.pancan.bags.FeatureBag(
                batch_size=16, device="cuda", verbose=True, debug=True, 
                cfg=dict(slideshard="@autopath.pancan.tiles.PancanSlideShard()", 
                         extractor="@autopath.gigaq.dinov2.models.BackboneEvaluator()",
                         split="test",
            )).build().read()
"""
from dataclasses import dataclass, asdict
import datetime
import json
import math
import os
import pdb #DEBUG
import pickle
import sys
import time
from typing import List, Dict, Optional, Union, Tuple, Callable

import fsspec

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

import ray
import torch

import dbx
from dbx import Logger, Datablock, Databatch
from .tiles import PancanSlideShard, PancanSlideBatch

#from ..gigaq.dinov2.models import BackboneEvaluator

DBKSPACE = os.environ.get("DBKSPACE", "/mnt/labshare/PROJECTS/GIGAQ/dbx")
DBKREPO = os.environ.get("DBKREPO", f"{os.environ.get('HOME')}/autopath")

class FeatureBag(Datablock):
    @dataclass
    class CONFIG:
        extractor: Callable
        slideshard: PancanSlideShard
        split: str = "test"

    def __init__(self, *args, batch_size: int = 16, device: str = 'cuda', **kwargs):
        super().__init__(*args, **kwargs)
        self.batch_size = batch_size
        self.device = device

    def __post_init__(self):
        self.eval = self.config.extractor
        self.tiles, self.origin, self.slide = self.config.slideshard.read(self.config.split)
        self.FILE = f"{self.slide}.pt"

    def build(self):
        feature_list = []
        for k in range(math.ceil(len(self.tiles)/self.batch_size)):
            n = k*self.batch_size
            batch = self.tiles[n:n+self.batch_size].to(self.device)
            self.log.verbose(f"Evaluating batch {k}: device: {self.device}")
            features_ = self.eval(batch).to('cpu')
            torch.cuda.empty_cache()
            self.log.verbose(f"Evaluating batch {k}: done")
            feature_list.append(features_)
        features = torch.cat(feature_list)
        dbx.write_tensor(features, self.path())
        return self

    def read(self):
        features = dbx.read_tensor(self.path())
        return features


class FeatureBags(Databatch):
    DATABLOCK = FeatureBag
    @dataclass
    class CONFIG:
        backbone: torch.nn.Module
        slidebatch: PancanSlideBatch


