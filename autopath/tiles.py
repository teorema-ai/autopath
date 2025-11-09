
import functools
from typing import Optional

import torchvision


import dbx
from dbx import Logger

from autopath.databits import DataShard, DataBag, DataClip, DataSplit, DataFold, DataClipDataset


logger = Logger()


class TileShard(DataShard):
    @functools.cached_property
    def tiles(self):
        return self.tensor
    
    @property
    def labels(self):
        raise NotImplementedError()
    

class TileBag(TileShard, DataBag):
    def __init__(self):
        TileShard.__init__(self)

class TileClip(DataClip):
    ...

class TileSplit(DataSplit):
    ...

class TileFold(DataFold):
    ...


def tileset(tileclip: DataClip, 
            transform: Optional[torchvision.transforms.Compose] = None,
            *,
            debug: bool = False,
            verbose: bool = False,
            log = None,
):
    tileclip = dbx.eval_term(tileclip)
    transform = dbx.eval_term(transform)
    return DataClipDataset(
                          clip=tileclip, 
                          transform=transform, 
                          debug=debug, 
                          verbose=verbose,
                          log=log,
    )