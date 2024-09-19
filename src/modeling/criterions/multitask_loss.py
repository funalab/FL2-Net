import torch
import torch.nn as nn

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
    def __init__(self, num=2, init_weight=None):
        super().__init__()
        if init_weight==None:
            params = torch.ones(num, requires_grad=True)
        else:
            params = torch.tensor(init_weight, requires_grad=True)
        self.params = torch.nn.Parameter(params)

    def forward(self, *losses):
        # losses is a list of dict
        loss_weighted = {}
        for w, loss_dict in zip(self.params, losses):
            for k, v in loss_dict.items():
                loss_weighted[k] = 0.5 / (w ** 2) * v + torch.log(1 + w ** 2)

        return loss_weighted
    

class WeightedLoss(nn.Module):
    def __init__(self, num=2, weight=[1, 2]):
        super().__init__()
        assert num == len(weight)
        self.params = weight

    def forward(self, *losses):
        loss_weighted = {}
        for w, loss_dict in zip(self.params, losses):
            for k, v in loss_dict.items():
                loss_weighted[k] = w * v

        return loss_weighted
