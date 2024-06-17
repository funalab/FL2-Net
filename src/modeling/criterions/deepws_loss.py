import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat
from src.modeling.criterions.segloss import BinaryDiceLoss

class SetCriterion(nn.Module):

    def __init__(self, cfg):
        super().__init__()
        #self.center_loss = nn.BCEWithLogitsLoss()
        self.center_loss = BinaryDiceLoss()
        self.mask_loss = BinaryDiceLoss()
        #self.mask_loss = nn.BCELoss()
        self.center_weight = cfg.MODEL.LOSS.CENTER_WEIGHT
        self.mask_weight = cfg.MODEL.LOSS.MASK_WEIGHT


    def forward(self, outputs, targets):
        """This performs the loss computation.
        Args:
            outputs:
                mask     [ B x 1 x D x H x W ]
                center   [ B x 1 x D x H x W ]

            targets:
                mask     [ B x     D x H x W ]
                center   [ B x     D x H x W ]
        """
        pred_masks = outputs["mask"]
        pred_centers = outputs["center"]

        gt_masks = targets["mask"]
        gt_centers = targets["center"]

        cur_device = pred_centers.device

        loss_center = self.center_loss(pred_centers, gt_centers.to(cur_device))
        loss_mask = self.mask_loss(pred_masks, gt_masks.to(cur_device))

        loss_center_weighted = self.center_weight * loss_center
        loss_mask_weighted = self.mask_weight * loss_mask

        return {
            "loss_center": loss_center_weighted,
            "loss_mask": loss_mask_weighted,
            "loss_sum": loss_mask_weighted + loss_center_weighted
        }
