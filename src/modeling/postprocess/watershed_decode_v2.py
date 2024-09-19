#!/usr/bin/python3
# -*- coding:utf-8 -*-

import numpy as np
import torch
import torch.nn.functional as F
from scipy import ndimage
from skimage.measure import label
from skimage import morphology
from skimage.segmentation import watershed
from skimage import transform

class WatershedDecoder():

    def __init__(self, cfg):

        super().__init__()
        self.center_thr = cfg.MODEL.TEST.OBJECT_CENTER_THRESHOLD
        self.mask_thr = cfg.MODEL.TEST.OBJECT_MASK_THRESHOLD

    def decode(self, outputs):
        r"""
        decode output feature map to detection results

        Args:
            mask(Tensor): tensor that represents predicted segmentation mask
            center(Tensor): tensor that represents predicted object center probability
        """
        centers = torch.sigmoid(outputs["center"]).squeeze(1)         # B x D x H x W
        masks = torch.sigmoid(outputs["mask"]).squeeze(1)             # B x D x H x W
        distances = torch.sigmoid(outputs["distance"]).squeeze(1)     # B x D x H x W

        centers = WatershedDecoder.pseudo_nms(centers)
        centers = (centers > self.center_thr).cpu().numpy().astype(np.uint8)
        masks = (masks > self.mask_thr).cpu().numpy().astype(np.uint8)
        distances = distances.cpu().detach().numpy()

        instances = []
        for center, mask, distance in zip(centers, masks, distances):
            if center.sum() > 0:
                det = label(center, connectivity=3)
                wsimage = watershed(-distance, det, mask=mask)
            else:
                wsimage = mask
            instances.append(wsimage)

        instances = np.stack(instances)

        predictions = {
            "instances": torch.from_numpy(instances),
        }
        return predictions


    @staticmethod
    def pseudo_nms(fmap, pool_size=3):
        r"""
        apply max pooling to get the same effect of nms

        Args:
            fmap(Tensor): output tensor of previous step
            pool_size(int): size of max-pooling
        """
        pad = (pool_size - 1) // 2
        fmap_max = F.max_pool3d(fmap, pool_size, stride=1, padding=pad)
        keep = (fmap_max == fmap).float()
        return fmap * keep
