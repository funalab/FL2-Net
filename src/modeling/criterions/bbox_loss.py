import torch
import torch.nn.functional as F
from torch import nn

class IOULoss(nn.Module):

    def __init__(self, loc_loss_type='iou', reduction='mean', eps=1e-7):
        super(IOULoss, self).__init__()
        self.loc_loss_type = loc_loss_type
        self.reduction = reduction
        self.eps = eps

    def forward(self, pred, target, weight=None):
        pred_front = pred[:, 0]
        pred_rear = pred[:, 1]
        pred_top = pred[:, 2]
        pred_bottom = pred[:, 3]
        pred_left = pred[:, 4]
        pred_right = pred[:, 5]

        target_front = target[:, 0]
        target_rear = target[:, 1]
        target_top = target[:, 2]
        target_bottom = target[:, 3]
        target_left = target[:, 4]
        target_right = target[:, 5]

        target_area = (target_left + target_right) * \
                      (target_top + target_bottom) * \
                      (target_front + target_rear)
        pred_area = (pred_left + pred_right) * \
                    (pred_top + pred_bottom) * \
                    (pred_front + pred_rear)

        # calculate area of union and intersection
        w_intersect = torch.min(pred_left, target_left) + \
                      torch.min(pred_right, target_right)
        h_intersect = torch.min(pred_top, target_top) + \
                      torch.min(pred_bottom, target_bottom)
        d_intersect = torch.min(pred_front, target_front) + \
                      torch.min(pred_rear, target_rear)
        area_intersect = w_intersect * h_intersect * d_intersect
        area_union = target_area + pred_area - area_intersect

        # calculate areas of the smallest rectangles that enclose both GT and PR (for GIoU)
        g_w_intersect = torch.max(pred_left, target_left) + \
                        torch.max(pred_right, target_right)
        g_h_intersect = torch.max(pred_top, target_top) + \
                        torch.max(pred_bottom, target_bottom)
        g_d_intersect = torch.max(pred_front, target_front) + \
                        torch.max(pred_rear, target_rear)
        ac_union = g_w_intersect * g_h_intersect * g_d_intersect

        # calculate distance between centroids (for DIoU)
        c_w_dist = (((target_right - target_left) - (pred_right - pred_left)) / 2) ** 2
        c_h_dist = (((target_top - target_bottom) - (pred_top - pred_bottom)) / 2) ** 2
        c_d_dist = (((target_front - target_rear) - (pred_front - pred_rear)) / 2) ** 2
        c_dist = c_w_dist + c_h_dist + c_d_dist

        # calculate diagonal length of the smallest rectangles that enclose both GT and PR (for DIoU)
        diag = g_w_intersect**2 + g_h_intersect**2 + g_d_intersect**2

        ious = area_intersect / (area_union + self.eps)
        gious = ious - ((ac_union - area_union) / (ac_union + self.eps))
        dious = ious - (c_dist / (diag + self.eps))

        if self.loc_loss_type == 'iou':
            losses = -torch.log(ious)
        elif self.loc_loss_type == 'linear_iou':
            losses = 1 - ious
        elif self.loc_loss_type == 'giou':
            losses = 1 - gious
        elif self.loc_loss_type == 'diou':
            losses = 1 - dious
        else:
            raise NotImplementedError

        if weight is not None:
            losses = losses * weight
        else:
            losses = losses

        if self.reduction == 'mean':
            return losses.mean()
        if self.reduction == 'sum':
            return losses.sum()
        elif self.reduction == 'batch':
            return losses.sum(dim=[1])
        elif self.reduction == 'none':
            return losses
        else:
            raise NotImplementedError


def bbox_transform(deltas, weights):
    wx, wy, ww, wh = weights
    dx = deltas[:, 0::4] / wx
    dy = deltas[:, 1::4] / wy
    dw = deltas[:, 2::4] / ww
    dh = deltas[:, 3::4] / wh

    dw = torch.clamp(dw, max=cfg.BBOX_XFORM_CLIP)
    dh = torch.clamp(dh, max=cfg.BBOX_XFORM_CLIP)

    pred_ctr_x = dx
    pred_ctr_y = dy
    pred_w = torch.exp(dw)
    pred_h = torch.exp(dh)

    x1 = pred_ctr_x - 0.5 * pred_w
    y1 = pred_ctr_y - 0.5 * pred_h
    x2 = pred_ctr_x + 0.5 * pred_w
    y2 = pred_ctr_y + 0.5 * pred_h

    return x1.view(-1), y1.view(-1), x2.view(-1), y2.view(-1)

def compute_diou(output, target, bbox_inside_weights, bbox_outside_weights,
                transform_weights=None):
    if transform_weights is None:
        transform_weights = (1., 1., 1., 1.)
    x1, y1, x2, y2 = bbox_transform(output, transform_weights)
    x1g, y1g, x2g, y2g = bbox_transform(target, transform_weights)

    x2 = torch.max(x1, x2)
    y2 = torch.max(y1, y2)

    x_p = (x2 + x1) / 2
    y_p = (y2 + y1) / 2
    x_g = (x1g + x2g) / 2
    y_g = (y1g + y2g) / 2

    xkis1 = torch.max(x1, x1g)
    ykis1 = torch.max(y1, y1g)
    xkis2 = torch.min(x2, x2g)
    ykis2 = torch.min(y2, y2g)

    xc1 = torch.min(x1, x1g)
    yc1 = torch.min(y1, y1g)
    xc2 = torch.max(x2, x2g)
    yc2 = torch.max(y2, y2g)

    intsctk = torch.zeros(x1.size()).to(output)
    mask = (ykis2 > ykis1) * (xkis2 > xkis1)
    intsctk[mask] = (xkis2[mask] - xkis1[mask]) * (ykis2[mask] - ykis1[mask])
    unionk = (x2 - x1) * (y2 - y1) + (x2g - x1g) * (y2g - y1g) - intsctk + 1e-7
    iouk = intsctk / unionk

    c = ((xc2 - xc1) ** 2) + ((yc2 - yc1) ** 2) +1e-7
    d = ((x_p - x_g) ** 2) + ((y_p - y_g) ** 2)
    u = d / c
    diouk = iouk - u
    iou_weights = bbox_inside_weights.view(-1, 4).mean(1) * bbox_outside_weights.view(-1, 4).mean(1)
    iouk = ((1 - iouk) * iou_weights).sum(0) / output.size(0)
    diouk = ((1 - diouk) * iou_weights).sum(0) / output.size(0)

    return iouk, diouk


def compute_ciou(output, target, bbox_inside_weights, bbox_outside_weights,
                transform_weights=None):
    if transform_weights is None:
        transform_weights = (1., 1., 1., 1.)

    x1, y1, x2, y2 = bbox_transform(output, transform_weights)
    x1g, y1g, x2g, y2g = bbox_transform(target, transform_weights)

    x2 = torch.max(x1, x2)
    y2 = torch.max(y1, y2)
    w_pred = x2 - x1
    h_pred = y2 - y1
    w_gt = x2g - x1g
    h_gt = y2g - y1g

    x_center = (x2 + x1) / 2
    y_center = (y2 + y1) / 2
    x_center_g = (x1g + x2g) / 2
    y_center_g = (y1g + y2g) / 2

    xkis1 = torch.max(x1, x1g)
    ykis1 = torch.max(y1, y1g)
    xkis2 = torch.min(x2, x2g)
    ykis2 = torch.min(y2, y2g)

    xc1 = torch.min(x1, x1g)
    yc1 = torch.min(y1, y1g)
    xc2 = torch.max(x2, x2g)
    yc2 = torch.max(y2, y2g)

    intsctk = torch.zeros(x1.size()).to(output)
    mask = (ykis2 > ykis1) * (xkis2 > xkis1)
    intsctk[mask] = (xkis2[mask] - xkis1[mask]) * (ykis2[mask] - ykis1[mask])
    unionk = (x2 - x1) * (y2 - y1) + (x2g - x1g) * (y2g - y1g) - intsctk + 1e-7
    iouk = intsctk / unionk

    c = ((xc2 - xc1) ** 2) + ((yc2 - yc1) ** 2) +1e-7
    d = ((x_center - x_center_g) ** 2) + ((y_center - y_center_g) ** 2)
    u = d / c
    v = (4 / (math.pi ** 2)) * torch.pow((torch.atan(w_gt/h_gt)-torch.atan(w_pred/h_pred)),2)
    with torch.no_grad():
        S = 1 - iouk
        alpha = v / (S + v)
    ciouk = iouk - (u + alpha * v)
    iou_weights = bbox_inside_weights.view(-1, 4).mean(1) * bbox_outside_weights.view(-1, 4).mean(1)
    iouk = ((1 - iouk) * iou_weights).sum(0) / output.size(0)
    ciouk = ((1 - ciouk) * iou_weights).sum(0) / output.size(0)

    return iouk, ciouk

def compute_giou(output, target, bbox_inside_weights, bbox_outside_weights,
                transform_weights=None):
    if transform_weights is None:
        transform_weights = (1., 1., 1., 1.)

    x1, y1, x2, y2 = bbox_transform(output, transform_weights)
    x1g, y1g, x2g, y2g = bbox_transform(target, transform_weights)

    x2 = torch.max(x1, x2)
    y2 = torch.max(y1, y2)

    xkis1 = torch.max(x1, x1g)
    ykis1 = torch.max(y1, y1g)
    xkis2 = torch.min(x2, x2g)
    ykis2 = torch.min(y2, y2g)

    xc1 = torch.min(x1, x1g)
    yc1 = torch.min(y1, y1g)
    xc2 = torch.max(x2, x2g)
    yc2 = torch.max(y2, y2g)

    intsctk = torch.zeros(x1.size()).to(output)
    mask = (ykis2 > ykis1) * (xkis2 > xkis1)
    intsctk[mask] = (xkis2[mask] - xkis1[mask]) * (ykis2[mask] - ykis1[mask])
    unionk = (x2 - x1) * (y2 - y1) + (x2g - x1g) * (y2g - y1g) - intsctk + 1e-7
    iouk = intsctk / unionk

    area_c = (xc2 - xc1) * (yc2 - yc1) + 1e-7
    giouk = iouk - ((area_c - unionk) / area_c)
    iou_weights = bbox_inside_weights.view(-1, 4).mean(1) * bbox_outside_weights.view(-1, 4).mean(1)
    iouk = ((1 - iouk) * iou_weights).sum(0) / output.size(0)
    giouk = ((1 - giouk) * iou_weights).sum(0) / output.size(0)

    return iouk, giouk