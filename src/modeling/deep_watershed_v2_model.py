import torch
from torch import nn
from torch.nn import functional as F
from skimage.measure import regionprops
import numpy as np
import tifffile

from src.utils.memory import retry_if_cuda_oom
from src.utils.misc import round_up_pad
from src.utils.utils import edt_prob
from src.utils.stitch import stitch
from src.modeling.build import *
from src.modeling.criterions.deepws_v2_loss import SetCriterion
from src.modeling.postprocess.watershed_decode_v2 import WatershedDecoder


class DeepWatershed_v2(nn.Module):
    """
    Main class for instance segmentation using waterhed algorithm.
    """
    def __init__(self, cfg, backbone=None):
        super().__init__()

        out_channel = cfg.MODEL.SEM_SEG_HEAD.NUM_CLASSES
        self.gaussian_radius_ratio = cfg.MODEL.DEEPWATERSHED.RADIUS_RATIO
        self.gaussian_sigma_ratio = cfg.MODEL.DEEPWATERSHED.SIGMA_RATIO

        # building model
        if (backbone is None):
            self.backbone = build_backbone(cfg)
        else:
            self.backbone = backbone

        # loss function
        self.criterion = SetCriterion(cfg)

        # decoder for mask prediction
        self.decoder = nn.Sequential(
            build_pixel_decoder(cfg),
            nn.BatchNorm3d(out_channel),
            nn.ReLU(inplace=True)
        )
        self.mask_head = nn.Conv3d(out_channel, 1, kernel_size=1, stride=1)
        self.center_head = nn.Conv3d(out_channel, 1, kernel_size=1, stride=1)
        self.dist_head = nn.Conv3d(out_channel, 1, kernel_size=1, stride=1)
        
        self.postprocess = WatershedDecoder(cfg)

        self.n_down = cfg.MODEL.BACKBONE.NUM_LEVEL-1
        self.ch_conf = {"mask":1, "center":1, "distance":1}


    def forward_features(self, x):
        features = self.backbone(x)
        features = self.decoder(features)
        outputs = {}
        outputs["mask"] = self.mask_head(features)
        outputs["center"] = self.center_head(features)
        outputs["distance"] = self.dist_head(features)
        return outputs
    

    def forward(self, inputs, mode='train', inference=False, patchsize=None):
        images = inputs["image"]
        depth, height, width = images.shape[-3:]
        results = {}

        if mode == 'train':
            images = inputs["image"].to('cuda')
            outputs = self.forward_features(images)
            gt_instances = self.prepare_targets(inputs["instances"])

            results['loss'] = self.criterion(outputs, gt_instances)
            print(results['loss'])
        else:
            if patchsize == None:
                images = round_up_pad(images, n_down=self.n_down).to('cuda')
                outputs = self.forward_features(images)
            else:
                outputs = stitch(images, self, patchsize)

        if inference:
            pred = self.postprocess.decode(outputs)
            pred['instances'] = pred['instances'][..., :depth, :height, :width]
            results.update(pred)
        else:
            results.update(outputs)

        return results


    def prepare_targets(self, targets):
        '''
        Args:
            targets   : [ B x D x H x W ]
        Returns:
            center    : [ B x D x H x W ]
            mask      : [ B x D x H x W ]
            distance  : [ B x D x H x W ]
        '''
        batch_size, depth, height, width = targets.shape
        center_maps = []
        distance_maps = []

        for bs in range(batch_size):
            target = targets[bs]
            center_coord = []
            size = []

            for prop in regionprops(target.numpy()):
                mind, minh, minw, maxd, maxh, maxw = prop.bbox
                # center coordinates
                center_coord.append([
                    int((maxd + mind) / 2), 
                    int((maxh + minh) / 2), 
                    int((maxw + minw) / 2)
                ])
                # object size
                size.append([
                    maxd - mind, 
                    maxh - minh, 
                    maxw - minw
                ])

            # distnace transformation (and normalization)
            distance_maps.append(
                torch.tensor(edt_prob(target.numpy()))
            )

            # probability map for centroids
            center_maps.append(
                self.generate_score_map(size, center_coord, depth, height, width)
            )

        new_targets = {
            "center": torch.stack(center_maps),
            "mask": (targets > 0).long(),
            "distance": torch.stack(distance_maps)
        }
        return new_targets

    def generate_score_map(self, gt_whd, centers_int, depth, height, width):
        radius = torch.tensor(gt_whd) / self.gaussian_radius_ratio
        radius = torch.clamp_min(radius, 0)
        radius = radius.type(torch.int).cpu().numpy()
        fmap = torch.zeros(depth, height, width)
        for i in range(len(radius)):
            fmap = self.draw_gaussian(fmap, centers_int[i], radius[i])
        return fmap

    def gaussian3D(self, radius, sigma):
        # m, n = [(s - 1.) / 2. for s in shape]
        l, m, n = radius
        z, y, x = np.ogrid[-l:l + 1, -m:m + 1, -n:n + 1]

        gauss = np.exp(-(x * x + y * y + z * z) / (2 * sigma[0] * sigma[1] * sigma[2]))
        gauss[gauss < np.finfo(gauss.dtype).eps * gauss.max()] = 0
        return gauss

    def draw_gaussian(self, fmap, center, radius, k=1):
        diameter = 2 * radius + 1
        gaussian = self.gaussian3D(radius, sigma=diameter / self.gaussian_sigma_ratio)
        gaussian = torch.Tensor(gaussian)
        z, y, x = int(center[0]), int(center[1]), int(center[2])
        depth, height, width = fmap.shape[:3]

        left, right = min(x, radius[2]), min(width - x, radius[2] + 1)
        top, bottom = min(y, radius[1]), min(height - y, radius[1] + 1)
        front, rear = min(z, radius[0]), min(depth - z, radius[0] + 1)

        masked_fmap = fmap[z - front:z + rear, y - top:y + bottom, x - left:x + right]
        masked_gaussian = gaussian[radius[0] - front:radius[0] + rear, radius[1] - top:radius[1] + bottom, radius[2] - left:radius[2] + right]
        if min(masked_gaussian.shape) > 0 and min(masked_fmap.shape) > 0:
            masked_fmap = torch.max(masked_fmap, masked_gaussian * k)
            fmap[z - front:z + rear, y - top:y + bottom, x - left:x + right] = masked_fmap
        return fmap
