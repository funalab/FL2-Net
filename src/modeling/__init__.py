from .embedseg_model import EmbedSeg
from .semantic_segmentation_model import AbstractSemSegModel
from .deep_watershed_model import DeepWatershed
from .deep_watershed_v2_model import DeepWatershed_v2
from .semantic_segmentation_model import AbstractSemSegModel

def get_model(cfg):

    if cfg.MODEL.META_ARCHITECTURE == "DeepWatershed":
        model = DeepWatershed(cfg)

    elif cfg.MODEL.META_ARCHITECTURE == "DeepWatershedV2":
        model = DeepWatershed_v2(cfg)

    elif cfg.MODEL.META_ARCHITECTURE == "EmbedSeg":
        model = EmbedSeg(cfg)
    
    elif cfg.MODEL.META_ARCHITECTURE == "StarDist":
        from .stardist_model import StarDist
        model = StarDist(cfg)
    
    elif cfg.MODEL.META_ARCHITECTURE == "NSN" or cfg.MODEL.META_ARCHITECTURE == "NDN":
        model = AbstractSemSegModel(cfg)

    return model
