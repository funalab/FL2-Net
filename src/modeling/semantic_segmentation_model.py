import torch
from torch import nn
from torch.nn import functional as F

from src.utils.memory import retry_if_cuda_oom
from src.utils.misc import round_up_pad
from src.modeling.build import *
from skimage.measure import label
from src.modeling.criterions.segloss import DiceLoss
from src.utils.stitch import stitch


class AbstractSemSegModel(nn.Module):
    """
    Main class for semantic segmentation using Encoder-Decoder model.
    """
    def __init__(self, cfg, backbone=None):
        super().__init__()
        out_channel = cfg.MODEL.SEM_SEG_HEAD.NUM_CLASSES
        # building model
        if (backbone is None):
            self.backbone = build_backbone(cfg)
        else:
            self.backbone = backbone
        
        self.decoder = build_pixel_decoder(cfg)

        # loss function
        self.criterion = DiceLoss(n_class=2)
        self.n_down = cfg.MODEL.BACKBONE.NUM_LEVEL-1
        self.ch_conf = {"mask": 2}

    def forward_features(self, x):
        features = self.backbone(x)
        outputs =  {'mask': self.decoder(features)}
        return outputs
    
    def forward(self, inputs, mode='train', inference=False, patchsize=None):
        images = inputs["image"]
        depth, height, width = images.shape[-3:]
        results = {}

        if mode == 'train':
            images = inputs["image"].to('cuda')
            outputs = self.forward_features(images)
            gt_instances = self.prepare_targets(inputs["instances"])
            results['loss'] = {'loss_sum': self.criterion(outputs['mask'], gt_instances.to('cuda'))}
            print(results['loss'])
        else:
            if patchsize == None:
                images = round_up_pad(images, n_down=self.n_down).to('cuda')
                outputs = self.forward_features(images)
            else:
                outputs = stitch(images, self, patchsize)

        if inference:
            outputs = F.softmax(outputs['mask'], dim=1)
            outputs = label(
                (0 < (outputs[:, 1] - outputs[:, 0])) * 1, 
                connectivity=3
            )
            results['instances'] = outputs[..., :depth, :height, :width]
        else:
            results.update(outputs)

        return results
    
    def prepare_targets(self, targets):
        targets_bin = (targets > 0) * 1
        return targets_bin.to(dtype=torch.long)
