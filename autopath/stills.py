from dataclasses import dataclass

from torch import nn
import torch.utils.data

import lightning as L

import dbx
from dbx import Datablock

from autopath.models.vred import VariationalReDecoderLightning


class VariationalReDecoderStill(Datablock):
    VERSION = 1
    TOPICFILES = {'logs': None,
                  'ckpts': None,
    }
    
    @dataclass 
    class CONFIG:
        vred_lightning: VariationalReDecoderLightning
        featureset: torch.utils.data.Dataset
        ckpt_path: str = None
        max_steps: int = None
        batch_size: int = 1
        shuffle: bool = False

    def __init__(self, *args, n_devices: int = 1, **kwargs):
        super().__init__(*args, n_devices=n_devices, **kwargs)
         
    def __post_init__(self):
        
        return self

    def __build__(self):
        logger = L.TensorBoardLogger(save_dir=self.dirpath('logs'))
        trainer = dbx.LightningTrainer(
            default_root_dir=self.dirpath('ckpts'), 
            max_steps=self.cfg.max_steps,
            logger=logger,
        )
        dataloader = torch.utils.data.DataLoader(self.cfg.featureset, batch_size=self.cfg.batch_size, shuffle=self.cfg.shuffle)
        trainer.fit(model=self.cfg.vred_lightning, train_dataloaders=dataloader)
        return self

  