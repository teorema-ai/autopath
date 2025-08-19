"""
    Examples:
        dbx.exec("autopath.pancan.bags.FeatureBag(cfg=dict(slideshard='@autopath.pancan.tiles.PancanSlideShard()', backbone='@autopath.gigaq.dinov2.models.gigapath_tile_backbone()'))")
        # with repo check
        dbx.exec(f"autopath.pancan.bags.FeatureBag(cfg=dict(slideshard='@autopath.pancan.tiles.PancanSlideShard()', backbone='@autopath.gigaq.dinov2.models.gigapath_tile_backbone()'), gitrepo='{HOME}/autopath')")

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

DATASPACE = os.environ.get("DATASPACE", "/mnt/labshare/PROJECTS/GIGAQ/dbx")

class FeatureBag(Datablock):
    @dataclass
    class CONFIG:
        backbone: torch.nn.Module
        slideshard: PancanSlideShard
        split: str = "test"

    def __post_init__(self):
        #self.eval = BackboneEvaluator(self.config.backbone)
        self.eval = self.config.backbone
        self.tiles, self.cancer, self.slide = self.config.slideshard.read(self.config.split)
        self.FILE = f"{self.slide}.pt"

    def build(self):
        features = self.eval(self.tiles)
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


