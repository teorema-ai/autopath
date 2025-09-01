from .images import PancanSlideBatch, CPTAC_ROOT, CPTAC_RESOLUTION


def PANCAN_CPTAC_SLIDE_BATCH():
    return PancanSlideBatch(cfg=dict(source=CPTAC_ROOT, resolution=CPTAC_RESOLUTION))