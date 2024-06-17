from src.modeling.backbone.unet import *
from src.modeling.backbone.fpn import FPN

def build_backbone(cfg, in_channel=None, out_channel=None):
    model_args = {}
    if cfg.MODEL.BACKBONE.NAME == "UNetEncoder":
        model_args["init_channel"] = cfg.MODEL.BACKBONE.CH_BASE if in_channel==None else in_channel
        model_args["kernel_size"] = cfg.MODEL.BACKBONE.KERNEL_SIZE
        model_args["pool_size"] = cfg.MODEL.BACKBONE.POOL_SIZE
        model_args["ap_factor"] = cfg.MODEL.BACKBONE.AP_FACTOR
        model_args["n_down"] = cfg.MODEL.BACKBONE.NUM_LEVEL-1
        model_args["residual"] = cfg.MODEL.BACKBONE.RESIDUAL

        return UNet_Encoder(**model_args)

    elif cfg.MODEL.BACKBONE.NAME == "RUNetEncoder":
        assert cfg.MODEL.BACKBONE.RECURRENT
        model_args["init_channel"] = cfg.MODEL.BACKBONE.CH_BASE if in_channel==None else in_channel
        model_args["kernel_size"] = cfg.MODEL.BACKBONE.KERNEL_SIZE
        model_args["pool_size"] = cfg.MODEL.BACKBONE.POOL_SIZE
        model_args["ap_factor"] = cfg.MODEL.BACKBONE.AP_FACTOR
        model_args["n_down"] = cfg.MODEL.BACKBONE.NUM_LEVEL-1
        model_args["residual"] = cfg.MODEL.BACKBONE.RESIDUAL
        # RNN parameteres
        model_args["rnn_type"] = cfg.MODEL.BACKBONE.RNN.TYPE
        model_args["seq_len"] = cfg.MODEL.BACKBONE.RNN.LENGTH
        model_args["bidirectional"] = cfg.MODEL.BACKBONE.RNN.BIDIRECTIONAL

        return RUNet_Encoder(**model_args)


def build_pixel_decoder(cfg, in_channel=None, out_channel=None):
    model_args = {}
    if cfg.MODEL.SEM_SEG_HEAD.NAME == 'UNetDecoder':
        ap_factor = cfg.MODEL.BACKBONE.AP_FACTOR
        init_channel = cfg.MODEL.BACKBONE.CH_BASE
        assert cfg.MODEL.SEM_SEG_HEAD.NUM_LEVEL == cfg.MODEL.BACKBONE.NUM_LEVEL
        n_down = cfg.MODEL.SEM_SEG_HEAD.NUM_LEVEL-1
        model_args["in_channel"] = init_channel * (ap_factor ** (n_down + 1)) if in_channel==None else in_channel
        model_args["n_class"] = cfg.MODEL.SEM_SEG_HEAD.NUM_CLASSES if out_channel==None else out_channel
        model_args["kernel_size"] = cfg.MODEL.BACKBONE.KERNEL_SIZE
        model_args["pool_size"] = cfg.MODEL.BACKBONE.POOL_SIZE
        model_args["op_factor"] = cfg.MODEL.BACKBONE.AP_FACTOR
        model_args["n_down"] = n_down
        model_args["residual"] = cfg.MODEL.SEM_SEG_HEAD.RESIDUAL

        return UNet_Decoder(**model_args)

    elif cfg.MODEL.SEM_SEG_HEAD.NAME == 'FPN':
        n_level = cfg.MODEL.SEM_SEG_HEAD.NUM_LEVEL
        ap_factor = cfg.MODEL.BACKBONE.AP_FACTOR
        init_channel = cfg.MODEL.BACKBONE.CH_BASE

        in_feat = [int(init_channel * (ap_factor ** (i+1))) for i in reversed(range(n_level))]
        out_feat = cfg.MODEL.SEM_SEG_HEAD.MULTISCALE_OUT_FEATURES
        if isinstance(out_feat, int):
            out_feat = [out_feat, ] * n_level
        assert len(out_feat) == n_level

        model_args["int_channels"] = in_feat if in_channel==None else in_channel
        model_args["out_channels"] = out_feat if out_channel==None else out_channel
        model_args["n_level"] = n_level

        return FPN(**model_args)

