import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

class DiceLoss(nn.Module):
    def __init__(self, n_class=2, smooth=1e-7, p=2, weight=None, size_average=True):
        super(DiceLoss, self).__init__()
        self.smooth = smooth
        self.p = p
        self.n_class = n_class

    def forward(self, inputs, targets):
        '''
        [inputs]
        input: (B, C, D, H, W)
        targets: (B, D, H, W)
        '''
        batch, channel = input.shape[:2]
        inputs = F.softmax(inputs, dim=1)
        inputs = input.view(batch, channel, -1)

        targets = F.one_hot(targets, num_classes=self.n_class)
        targets = targets.view(batch, channel, -1)

        intersection = (inputs * targets).sum(-1)
        union = (inputs.pow(self.p) + targets.pow(self.p)).sum(-1)
        
        dice = (intersection + self.smooth) / (union + self.smooth)
        
        return torch.mean(1 - dice.sum(-1))

class BinaryDiceLoss(nn.Module):

    def __init__(self, p=2, smooth=1e-7):
        super().__init__()
        self.smooth = smooth
        self.p = p

    def forward(self, inputs, targets):
        '''
        Args:
            inputs:   [ B x 1 x D x H x W ]
            targets:  [ B x     D x H x W ]
        '''
        batch_size = inputs.shape[0]
        inputs = torch.sigmoid(inputs)
        inputs = inputs.view(batch_size, -1)
        targets = targets.view(batch_size, -1)
        
        intersection = (inputs * targets).sum(dim=1)
        union = (inputs.pow(self.p) + targets.pow(self.p)).sum(dim=1)

        dice = (2.*intersection + self.smooth) / (union + self.smooth)
        
        return torch.mean(1 - dice)


class FocalLoss(nn.Module):
    def __init__(self, gamma=2, alpha=0.25, size_average=True):
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha
        if isinstance(alpha, (float, int)): self.alpha = torch.Tensor([alpha, 1-alpha])
        if isinstance(alpha, list): self.alpha = torch.Tensor(alpha)
        self.size_average = size_average

    def forward(self, input, target):
        target = target.long()
        if input.dim() > 2:
            input = input.view(input.size(0), input.size(1), -1)  # N,C,D,H,W => N,C,D*H*W
            input = input.transpose(1, 2)    # N,C,D*H*W => N,D*H*W,C
            input = input.contiguous().view(-1, input.size()[2])  # N,D*H*W,C => N*D*H*W,C
            target = target.view(-1, 1)

            logpt = F.log_softmax(input)
            logpt = logpt.gather(1, target)
            logpt = logpt.view(-1)
            pt = Variable(logpt.data.exp())

            if self.alpha is not None:
                if self.alpha.type() != input.data.type():
                    self.alpha = self.alpha.type_as(input.data)
                at = self.alpha.gather(0, target.data.view(-1))
                logpt = logpt * Variable(at)

            loss = -(1 * (1 - pt) ** self.gamma) * logpt


            if self.size_average:
                return loss.mean()
            else:
                return loss.sum()

class BinaryFocalLoss(nn.Module):
    """
    CE(p_t) = -log(p_t)
    FL(p_t) = -α_t(1 - p_t)^γ log(p_t)
            = α_t(1 - p_t)^γ CE(p_t)

    p_t = p if (y==1) else 1 - p
    α_t = α if (y==1) else 1 - α
    """
    def __init__(
        self,
        alpha: float = 0.25,
        gamma: float = 2.0,
        reduction: str = "mean",
    ):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):

        ce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction="none")
        
        p = torch.sigmoid(inputs)
        p_t = p * targets + (1 - p) * (1 - targets)
        alpha_t = self.alpha * targets + (1 - self.alpha) * (1 - targets)
        
        loss = alpha_t * ((1 - p_t) ** self.gamma) * ce_loss

        if self.reduction == "none":
            pass
        elif self.reduction == "mean":
            loss = loss.mean()
        elif self.reduction == "sum":
            loss = loss.sum()
        elif self.reduction == "batchwise_mean":
            loss = loss.sum(dim=0)

        return loss

class AutomaticWeightedLoss(nn.Module):
    """automatically weighted multi-task loss
    Params:
        num: int, the number of loss
        x: multi-task loss
    Examples:
        loss1=1
        loss2=2
        awl = AutomaticWeightedLoss(2)
        loss_sum = awl(loss1, loss2)
    """
    def __init__(self, num=2):
        super().__init__()
        params = torch.ones(num, requires_grad=True)
        self.params = torch.nn.Parameter(params)

    def forward(self, *x):
        loss_sum = 0
        for i, loss in enumerate(x):
            loss_sum += 0.5 / (self.params[i] ** 2) * loss + torch.log(1 + self.params[i] ** 2)
        return loss_sum

class WeightedLoss(nn.Module):
    def __init__(self, weight=[1,2]):
        super().__init__()
        self.weight = weight

    def forward(self, *x):
        loss_sum = 0
        for loss, w in zip(x, self.weight):
            loss_sum += loss * w
        return loss_sum
