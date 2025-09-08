import os

from .images import PancanSlideTilebag, PancanTilesets, PancanTileset
from .images import PancanSlideBatch # DEPRECATED


PANCAN_DATASPACE = os.environ.get("PANCAN_DATASPACE", "/mnt/labshare/PERSONAL/dmitry/dbx") 
PANCAN_CPTAC_ROOT = os.environ.get("PANCAN_CPTAC_ROOT", "/mnt/labshare/SLIDES/CPTAC_downloads")
PANCAN_CPTAC_SAMPLE_TILEBAG_SOURCE = os.path.join(PANCAN_CPTAC_ROOT, "HNSCC/tfrecords/256px_256um/C3L-02621-23.tfrecords")
PANCAN_CPTAC_RESOLUTION = os.environ.get("PANCAN_CPTAC_RESOLUTION", "256px_256um")


def pancan_tilebag(name, *, root=PANCAN_DATASPACE, verbose=True, debug=False, gitrepo=None) -> PancanSlideTilebag:
        if name == "PANCAN_CPTAC_SAMPLE_TILEBAG":     
            return PancanTileBag(root, 
                                      spec=dict(source=PANCAN_CPTAC_SAMPLE_TILEBAG_SOURCE),
                                      verbose=verbose, 
                                      debug=debug, 
                                      gitrepo=gitrepo, 
            )
        else:
            raise ValueError(f"Unknown slide tilebag: {name}")




def pancan_tileset(name, *, root=PANCAN_DATASPACE, verbose=True, debug=False, gitrepo=None) -> PancanTileset:
    if name == "PANCAN_CPTAC_8020_TEST":
            return PancanTileset(root,
                                    spec=dict(
                                        datasets=f"@autopath.pancan.pipelines.pancan_tilesets('PANCAN_CPTAC_8020', verbose={verbose}, debug={debug}, gitrepo={repr(gitrepo)})",
                                        split="test",
                                    ),
                                    verbose=verbose,
                                    debug=debug,
                                    gitrepo=gitrepo,
            )
    elif name == "PANCAN_CPTAC_8020_TRAIN":
            return PancanTileset(root,
                                    spec=dict(
                                        datasets=f"@autopath.pancan.pipelines.pancan_tilesets('PANCAN_CPTAC_8020', verbose={verbose}, debug={debug}, gitrepo={repr(gitrepo)})",
                                        split="train",
                                    ),
                                    verbose=verbose,
                                    debug=debug,
                                    gitrepo=gitrepo,
            )
    elif name == "PANCAN_CPTAC_9802_TEST":
            return PancanTileset(root,
                                    spec=dict(
                                        datasets=f"@autopath.pancan.pipelines.pancan_tilesets('PANCAN_CPTAC_9802', verbose={verbose}, debug={debug}, gitrepo={repr(gitrepo)})",
                                        split="test",
                                    ),
                                    verbose=verbose,
                                    debug=debug,
                                    gitrepo=gitrepo,
            )
    elif name == "PANCAN_CPTAC_9802_TRAIN":
            return PancanTileset(root,
                                    spec=dict(
                                        datasets=f"@autopath.pancan.pipelines.pancan_tilesets('PANCAN_CPTAC_9802', verbose={verbose}, debug={debug}, gitrepo={repr(gitrepo)})",
                                        split="train",
                                    ),
                                    verbose=verbose,
                                    debug=debug,
                                    gitrepo=gitrepo,
            )
    else:
            raise ValueError(f"Unknown tileset: {name}")


def pancan_tilesets(name, *, root=PANCAN_DATASPACE, verbose=True, debug=False, gitrepo=None) -> PancanTilesets:
        if name == "PANCAN_CPTAC_8020":   
            return PancanTilesets(root, 
                                  spec=dict(source=PANCAN_CPTAC_ROOT, 
                                            resolution=PANCAN_CPTAC_RESOLUTION,
                                            train_fraction=0.8,
                                  ),
                                  verbose=verbose,
                                  debug=debug,
                                  gitrepo=gitrepo,
            )
        elif name == "PANCAN_CPTAC_9802":   
            return PancanTilesets(root, 
                                  spec=dict(source=PANCAN_CPTAC_ROOT, 
                                            resolution=PANCAN_CPTAC_RESOLUTION,
                                            train_fraction=0.98,
                                  ),
                                  verbose=verbose,
                                  debug=debug,
                                  gitrepo=gitrepo,
            )
        else:
            raise ValueError(f"Unknown tilesets: {name}")
