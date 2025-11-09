from dataclasses import dataclass

import torch
from torch import nn


from dbx import Datablock

from autopath.models import vae
from autopath.features import FeatureSet


class VariationalReDecoderStill(Datablock):
    VERSION = 1
    
    @dataclass 
    class CONFIG:
        vae: vae.VariationalDecoder
        features: FeatureSet
        loss: nn.Module

    def __init__(self, *args, n_devices: int = 1, **kwargs):
        super().__init__(*args, **kwargs)

    def make_dataloader(self):
        ...

    