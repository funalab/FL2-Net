import numpy as np
import torch
import torch.nn as nn
from src.modeling.criterions.lovasz_losses import lovasz_hinge


class SpatialEmbLoss_3d(nn.Module):

    def __init__(self, grid_z = 32, grid_y=1024, grid_x=1024, pixel_z = 1, pixel_y=1, pixel_x=1, loss_weight={'w_inst': 1, 'w_var': 10, 'w_seed': 1}, one_hot=False, n_sigma=3, foreground_weight=1):
        super().__init__()

        print('Created spatial emb loss function with: n_sigma: {}, foreground_weight: {}'.format(
            n_sigma, foreground_weight))
        print('*************************')
        self.n_sigma = n_sigma
        self.foreground_weight = foreground_weight

        xm = torch.linspace(0, pixel_x, grid_x).view(1, 1, 1, -1).expand(1, grid_z, grid_y, grid_x)
        ym = torch.linspace(0, pixel_y, grid_y).view(1, 1, -1, 1).expand(1, grid_z, grid_y, grid_x)
        zm = torch.linspace(0, pixel_z, grid_z).view(1, -1, 1, 1).expand(1, grid_z, grid_y, grid_x)
        xyzm = torch.cat((xm, ym, zm), 0)

        self.register_buffer("xyzm", xyzm)
        self.one_hot = one_hot
        self.w_inst = loss_weight['w_inst']
        self.w_var = loss_weight['w_var']
        self.w_seed = loss_weight['w_seed']

    def forward(self, prediction, gt_images):
        
        instances = gt_images['instances'].to('cuda')
        labels = gt_images['labels'].to('cuda')
        center_images = gt_images['center_images'].to('cuda')

        # instances B 1 Z Y X
        batch_size, depth, height, width = prediction.size(0), prediction.size(2), prediction.size(3), prediction.size(4)

        xyzm_s = self.xyzm[:, 0:depth, 0:height, 0:width].contiguous().to('cuda')  # 3 x d x h x w

        var_losses = 0
        instance_losses = 0
        seed_losses = 0

        for b in range(0, batch_size):

            spatial_emb = torch.tanh(prediction[b, 0:3]) + xyzm_s  # 3 x d x h x w 
            sigma = prediction[b, 3:3 + self.n_sigma]  # n_sigma x d x h x w
            seed_map = torch.sigmoid(prediction[b, 3 + self.n_sigma:3 + self.n_sigma + 1])  # 1 x d x h x w 
            # loss accumulators
            var_loss = 0
            instance_loss = 0
            seed_loss = 0
            obj_count = 0

            instance = instances[b].unsqueeze(0)  # 1 x d x h x w
            label = labels[b].unsqueeze(0)  # 1 x d x h x w
            center_image = center_images[b].unsqueeze(0)  # 1 x d x h x w
            instance_ids = instance.unique()
            instance_ids = instance_ids[instance_ids != 0]

            # regress bg to zero
            bg_mask = label == 0

            if bg_mask.sum() > 0:
                seed_loss += torch.sum(torch.pow(seed_map[bg_mask] - 0, 2))

            for id in instance_ids:

                in_mask = instance.eq(id)  # 1 x d x h x w
                #center_mask = in_mask & center_image.byte()
                center_mask = in_mask & center_image

                xyz_in = xyzm_s[in_mask.expand_as(xyzm_s)].view(3, -1)
                center = xyz_in.mean(1).view(3, 1, 1, 1)  # 3 x 1 x 1 x 1

                # calculate sigma
                sigma_in = sigma[in_mask.expand_as(sigma)].view(self.n_sigma, -1) # 3 x N

                s = sigma_in.mean(1).view(self.n_sigma, 1, 1, 1)  # n_sigma x 1 x 1 x 1

                # calculate var loss before exp
                var_loss = var_loss + torch.mean( torch.pow(sigma_in - s[..., 0, 0].detach(), 2))

                s = torch.exp(s * 10)
                dist = torch.exp(-1 * torch.sum(torch.pow(spatial_emb - center, 2) * s, 0, keepdim=True))

                # apply lovasz-hinge loss
                instance_loss = instance_loss + \
                                lovasz_hinge(dist * 2 - 1, in_mask)

                # seed loss
                seed_loss += self.foreground_weight * torch.sum(
                    torch.pow(seed_map[in_mask] - dist[in_mask].detach(), 2))

                obj_count += 1

            if obj_count > 0:
                instance_loss /= obj_count
                var_loss /= obj_count

            seed_loss = seed_loss / (depth * height * width)

            instance_losses += instance_loss
            seed_losses += seed_loss
            var_losses += var_loss

        loss_inst_weighted = instance_losses * self.w_inst / (b + 1)
        loss_var_weighted = var_losses * self.w_var / (b + 1)
        loss_seed_weighted = seed_losses * self.w_seed / (b + 1)

        return {
            "loss_inst": loss_inst_weighted,
            "loss_var": loss_var_weighted,
            "loss_seed": loss_seed_weighted,
            "loss_sum": loss_inst_weighted + loss_var_weighted + loss_seed_weighted + prediction.sum() * 0
        }