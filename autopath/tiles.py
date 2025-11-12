
import functools
from typing import Optional

import torchvision


import dbx
from dbx import Logger

from autopath.databits import Shard, Bag, Clip, Split, Fold, ClipDataset


logger = Logger()


class TileShard(Shard):
    @functools.cached_property
    def tiles(self):
        return self.tensor
    
    @property
    def labels(self):
        raise NotImplementedError()
    

class TileBag(TileShard, Bag):
    def __init__(self, *args, **kwargs):
        TileShard.__init__(self, *args, **kwargs)

class TileClip(Clip):
    ...

class TileSplit(Split):
    ...

class TileFold(Fold):
    ...


def tileset(tileclip: Clip, 
            transform: Optional[torchvision.transforms.Compose] = None,
            *,
            debug: bool = False,
            verbose: bool = False,
            log = None,
):
    tileclip = dbx.eval_term(tileclip)
    transform = dbx.eval_term(transform)
    return ClipDataset(
                          clip=tileclip, 
                          transform=transform, 
                          debug=debug, 
                          verbose=verbose,
                          log=log,
    )