import os

from .images import PancanTileBag, PancanTileBags, PancanTileSets, PancanTileSet


PANCAN_DATASPACE = os.environ.get("PANCAN_DATASPACE", "/mnt/labshare/PERSONAL/dmitry/dbx") 
PANCAN_CPTAC = os.environ.get("PANCAN_CPTAC", "/mnt/labshare/SLIDES/CPTAC_downloads")
PANCAN_CPTAC_SAMPLE = os.path.join(PANCAN_CPTAC, "HNSCC/tfrecords/256px_256um/C3L-02621-23.tfrecords")
PANCAN_CPTAC_RESOLUTION = os.environ.get("PANCAN_CPTAC_RESOLUTION", "256px_256um")


def pancan_tilebag(name, *, root=PANCAN_DATASPACE, verbose=True, debug=False, gitrepo=None) -> PancanSlideTileBag:
        if name == "PANCAN_CPTAC_SAMPLE":     
            return PancanTileBag(root, 
                                 spec=dict(source=PANCAN_CPTAC_SAMPLE),
                                 verbose=verbose, 
                                 debug=debug, 
                                 gitrepo=gitrepo, 
            )
        else:
            raise ValueError(f"Unknown slide tilebag: {name}")


def pancan_tilebags(name, *, root=PANCAN_DATASPACE, verbose=True, debug=False, gitrepo=None) -> PancanSlideTileBags:
        if name == "PANCAN_CPTAC":     
            return PancanTileBags(root, 
                                 spec=dict(
                                    source=PANCAN_CPTAC,
                                    resolution=PANCAN_CPTAC_RESOLUTION,
                                 ),
                                 verbose=verbose, 
                                 debug=debug, 
                                 gitrepo=gitrepo, 
            )
        else:
            raise ValueError(f"Unknown slide tilebag: {name}")


def pancan_tilesets(name, *, root=PANCAN_DATASPACE, verbose=True, debug=False, gitrepo=None) -> PancanTileSets:
        if name == "PANCAN_CPTAC_8020":   
            return PancanTileSets(root, 
                                  spec=dict(tilebags=f"@autopath.pancan.pipelines.pancan_tilebags('PANCAN_CPTAC', verbose={verbose}, debug={debug}, gitrepo={repr(gitrepo)})",
                                            train_fraction=0.8,
                                  ),
                                  verbose=verbose,
                                  debug=debug,
                                  gitrepo=gitrepo,
            )
        elif name == "PANCAN_CPTAC_9802":   
            return PancanTilesets(root, 
                                  spec=dict(tilebags=f"@autopath.pancan.pipelines.pancan_tilebags('PANCAN_CPTAC', verbose={verbose}, debug={debug}, gitrepo={repr(gitrepo)})",
                                            train_fraction=0.98,
                                  ),
                                  verbose=verbose,
                                  debug=debug,
                                  gitrepo=gitrepo,
            )
        else:
            raise ValueError(f"Unknown tilesets: {name}")


def pancan_tileset(name, *, root=PANCAN_DATASPACE, verbose=True, debug=False, gitrepo=None) -> PancanTileSet:
    if name == "PANCAN_CPTAC_8020_TEST":
            return PancanTileSet(root,
                                 spec=dict(
                                        datasets=f"@autopath.pancan.pipelines.pancan_tilesets('PANCAN_CPTAC_8020', verbose={verbose}, debug={debug}, gitrepo={repr(gitrepo)})",
                                        split="test",
                                 ),
                                 verbose=verbose,
                                 debug=debug,
                                 gitrepo=gitrepo,
            )
    elif name == "PANCAN_CPTAC_8020_TRAIN":
            return PancanTileSet(root,
                                    spec=dict(
                                        datasets=f"@autopath.pancan.pipelines.pancan_tilesets('PANCAN_CPTAC_8020', verbose={verbose}, debug={debug}, gitrepo={repr(gitrepo)})",
                                        split="train",
                                    ),
                                    verbose=verbose,
                                    debug=debug,
                                    gitrepo=gitrepo,
            )
    elif name == "PANCAN_CPTAC_9802_TEST":
            return PancanTileSet(root,
                                 spec=dict(
                                        datasets=f"@autopath.pancan.pipelines.pancan_tilesets('PANCAN_CPTAC_9802', verbose={verbose}, debug={debug}, gitrepo={repr(gitrepo)})",
                                        split="test",
                                 ),
                                 verbose=verbose,
                                 debug=debug,
                                 gitrepo=gitrepo,
            )
    elif name == "PANCAN_CPTAC_9802_TRAIN":
            return PancanTileSet(root,
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
