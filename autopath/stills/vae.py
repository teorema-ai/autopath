from dataclasses import dataclass

from torch import nn
import torch.utils.data

import dbx
from dbx import Datablock

from autopath.models import vae


class VariationalReDecoderStill(Datablock):
    VERSION = 1
    
    @dataclass 
    class CONFIG:
        vae: vae.VariationalReDecoder
        features: torch.utils.data.Dataset
        loss: nn.Module

    def __init__(self, *args, n_devices: int = 1, **kwargs):
        super().__init__(*args, n_devices=n_devices, **kwargs)

    ...