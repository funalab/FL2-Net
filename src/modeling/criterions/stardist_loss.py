import torch
import torch.nn as nn
import torch.nn.functional as F

class L1BCELoss(nn.Module):
    
    def __init__(self,scale_d=0.1, scale_reg=1e-4):
        super().__init__()
        self.scale_reg = scale_reg
        self.scale_d = scale_d
        self.dloss_func = DistanceLoss()
        self.oloss_func = nn.BCEWithLogitsLoss()
        
    def forward(self, prediction_p, prediction_r, gt_instances):
        target_r = gt_instances['distances'].to('cuda')
        target_p = gt_instances['probs'].to('cuda')
        #Predicted distances errors are weighted by object prob
        distloss = self.dloss_func(prediction_r, target_r, target_p) * self.scale_d
        objloss = self.oloss_func(prediction_p, target_p)
        distloss *= self.scale_d

        return {
            "loss_dist": distloss,
            "loss_obj": objloss,
            "loss_sum": objloss + distloss
        }


class DistanceLoss(nn.Module):
    def __init__(self, p=1e-4):
        super().__init__()
        self.p = p
    
    def forward(self, prediction_r, target_r, target_p):
        loss_unweighted = F.l1_loss(prediction_r, target_r, reduction='none') # calculate loss for each pixel
        mask = (target_p > 0) * 1 + (target_p == 0) * self.p
        loss = loss_unweighted * mask
        return torch.mean(loss)
