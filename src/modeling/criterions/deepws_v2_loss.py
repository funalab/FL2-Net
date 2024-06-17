import torch
import torch.nn as nn
import torch.nn.functional as F
from src.modeling.criterions.segloss import BinaryDiceLoss

class SetCriterion(nn.Module):

    def __init__(self, cfg):
        super().__init__()
        self.center_loss = BinaryDiceLoss()
        self.mask_loss = BinaryDiceLoss()
        self.dist_loss = BinaryDiceLoss()
        self.center_weight = cfg.MODEL.LOSS.CENTER_WEIGHT
        self.mask_weight = cfg.MODEL.LOSS.MASK_WEIGHT
        self.dist_weight = cfg.MODEL.LOSS.DIST_WEIGHT


    def forward(self, outputs, targets):
        """This performs the loss computation.
        Args:
            outputs:
                mask     [ B x 1 x D x H x W ]
                center   [ B x 1 x D x H x W ]
                dist     [ B x 1 x D x H x W ]

            targets:
                mask     [ B x     D x H x W ]
                center   [ B x     D x H x W ]
                dist     [ B x     D x H x W ]
        """
        pred_masks = outputs["mask"]
        pred_centers = outputs["center"]
        pred_dists = outputs["distance"]

        gt_masks = targets["mask"]
        gt_centers = targets["center"]
        gt_dists = targets["distance"]

        cur_device = pred_centers.device

        loss_center = self.center_loss(pred_centers, gt_centers.to(cur_device))
        loss_mask = self.mask_loss(pred_masks, gt_masks.to(cur_device))
        loss_dist = self.dist_loss(pred_dists, gt_dists.to(cur_device))

        loss_center_weighted = self.center_weight * loss_center
        loss_mask_weighted = self.mask_weight * loss_mask
        loss_dist_weighted = self.dist_weight * loss_dist

        return {
            "loss_center": loss_center_weighted,
            "loss_mask": loss_mask_weighted,
            "loss_dist": loss_dist_weighted,
            "loss_sum": loss_mask_weighted + loss_center_weighted + loss_dist_weighted
        }

class DistanceLoss(nn.Module):
    def __init__(self, p=1e-4):
        super().__init__()
        self.p = p

    def forward(self, prediction_r, target_r, target_p):
        loss_unweighted = F.l1_loss(
            torch.sigmoid(prediction_r), target_r, reduction='none'
        ) # calculate loss for each pixel
        mask = (target_p > 0) * 1 + (target_p == 0) * self.p
        loss = loss_unweighted * mask
        return torch.mean(loss)
