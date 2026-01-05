
import functools
from typing import Optional

import torchvision


import dbx
from dbx import Logger

from autopath.databits import Shard, Bag, Clip, ClipDataset


logger = Logger()


class TileShard(Shard):
    @functools.cached_property
    def tiles(self):
        return self.tensor
    
    @property
    def labels(self):
        raise NotImplementedError()
    

class TileBag(TileShard, Bag):
    def __init__(self, *args, name, label, **kwargs):
        Bag.__init__(self, name=name, label=label)
        TileShard.__init__(self, *args, **kwargs)


def tileset(tilebagclip: Clip, 
            transform: Optional[torchvision.transforms.Compose] = None,
            *,
            debug: bool = False,
            verbose: bool = False,
            log = None,
):
    tilebagclip = dbx.eval_term(tilebagclip)
    transform = dbx.eval_term(transform)
    return ClipDataset(clip=tilebagclip, 
                       transform=transform, 
                       debug=debug, 
                       verbose=verbose,
                       log=log,
    )