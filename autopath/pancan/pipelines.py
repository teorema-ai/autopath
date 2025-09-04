import os

from .images import PancanSlideBatch, PancanSlideTilebag, PancanTileDatasets

PANCAN_DATASPACE = os.environ.get("PANCAN_DATASPACE", "/mnt/labshare/PERSONAL/dmitry/dbx") 
PANCAN_CPTAC_ROOT = os.environ.get("PANCAN_CPTAC_ROOT", "/mnt/labshare/SLIDES/CPTAC_downloads")
PANCAN_CPTAC_SAMPLE_TILEBAG_SOURCE = os.path.join(PANCAN_CPTAC_ROOT, "HNSCC/tfrecords/256px_256um/C3L-02621-23.tfrecords")
PANCAN_CPTAC_RESOLUTION = os.environ.get("PANCAN_CPTAC_RESOLUTION", "256px_256um")


def PANCAN_CPTAC_TILE_DATASETS_8020(root=PANCAN_DATASPACE, *, verbose=True, debug=False, gitrepo=None):
    return PancanTileDatasets(root, 
                               spec=dict(source=PANCAN_CPTAC_ROOT, 
                                         resolution=PANCAN_CPTAC_RESOLUTION,
                                         train_fraction=0.8,
                               ),
                               verbose=verbose,
                               debug=debug,
                               gitrepo=gitrepo,
    )



def PANCAN_CPTAC_SAMPLE_TILEBAG(root=PANCAN_DATASPACE):
    return PancanSlideTilebag(root, spec=dict(source=PANCAN_CPTAC_SAMPLE_TILEBAG_SOURCE))


def PANCAN_CPTAC_SLIDE_BATCH(root=PANCAN_DATASPACE):
    return PancanSlideBatch(root, spec=dict(source=PANCAN_CPTAC_ROOT, resolution=PANCAN_CPTAC_RESOLUTION))