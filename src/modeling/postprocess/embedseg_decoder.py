import os
import threading

import numpy as np
import pandas as pd
import torch
from PIL import Image
from skimage.feature import peak_local_max


class Cluster_3d:
    """
            A class used to cluster pixel embeddings in 3D

            Attributes
            ----------
            xyzm :  float (3, W, D, H)
                    pixel coordinates of tile /grid

            one_hot : bool
                    Should be set to True, if the GT label masks are present in a one-hot encoded fashion
                    Not applicable to 3D. This parameter will be deprecated in a future release

            grid_x: int
                    Length (width) of grid / tile

            grid_y: int
                    Height of grid / tile

            grid_z: int
                    Depth of grid / tile

            pixel_x: float
                    if grid_x = 1000 and pixel_x = 1.0, then the pixel spacing along the x direction is pixel_x/(grid_x-1) = 1/999
            pixel_y: float
                    if grid_y = 1000 and pixel_y = 1.0, then the pixel spacing along the y direction is pixel_y/(grid_y-1) = 1/999
            pixel_z: float
                    if grid_z = 1000 and pixel_z = 1, then the pixel spacing along the z direction is pixel_z/(grid_z-1) = 1/999


            Methods
            -------
            __init__: Initializes an object of class `Cluster_3d`

            cluster_with_gt: use the predicted spatial embeddings from all pixels belonging to the GT label mask
                        to identify the predicted cluster (used during training and validation)

            cluster:    use the  predicted spatial embeddings from all pixels in the test image.
                        Employs `fg_thresh` and `seed_thresh`
            cluster_local_maxima: use the  predicted spatial embeddings from all pixels in the test image.
                        Employs only `fg_thresh`
            """

    def __init__(self, grid_z, grid_y, grid_x, pixel_z, pixel_y, pixel_x, one_hot=False):
        """
           Parameters
           ----------
           xyzm :  float (3, W, D, H)
                    pixel coordinates of tile /grid

            one_hot : bool
                    Should be set to True, if the GT label masks are present in a one-hot encoded fashion
                    Not applicable to 3D. This parameter will be deprecated in a future release

            grid_x: int
                    Length (width) of grid / tile

            grid_y: int
                    Height of grid / tile

            grid_z: int
                    Depth of grid / tile

            pixel_x: float
                    if grid_x = 1000 and pixel_x = 1.0, then the pixel spacing along the x direction is pixel_x/(grid_x-1) = 1/999
            pixel_y: float
                    if grid_y = 1000 and pixel_y = 1.0, then the pixel spacing along the y direction is pixel_y/(grid_y-1) = 1/999
            pixel_z: float
                    if grid_z = 1000 and pixel_z = 1, then the pixel spacing along the z direction is pixel_z/(grid_z-1) = 1/999

           """

        xm = torch.linspace(0, pixel_x, grid_x).view(1, 1, 1, -1).expand(1, grid_z, grid_y, grid_x)
        ym = torch.linspace(0, pixel_y, grid_y).view(1, 1, -1, 1).expand(1, grid_z, grid_y, grid_x)
        zm = torch.linspace(0, pixel_z, grid_z).view(1, -1, 1, 1).expand(1, grid_z, grid_y, grid_x)
        xyzm = torch.cat((xm, ym, zm), 0)

        self.xyzm = xyzm.cuda()
        self.one_hot = one_hot
        self.grid_x = grid_x
        self.grid_y = grid_y
        self.grid_z = grid_z
        self.pixel_x = pixel_x
        self.pixel_y = pixel_y
        self.pixel_z = pixel_z

    def cluster_with_gt(self, prediction, gt_instance, n_sigma=3, ):
        """
        Parameters
        ----------
        prediction :  PyTorch Tensor
                Model Prediction (7, D, H, W)
        gt_instance : PyTorch Tensor
                Ground Truth Instance Segmentation Label Map

        n_sigma: int, default = 3
                Number of dimensions in Raw Image
        Returns
        ----------
        instance: PyTorch Tensor (D, H, W)
                instance segmentation
       """

        depth, height, width = prediction.size(1), prediction.size(2), prediction.size(3)
        xyzm_s = self.xyzm[:, 0:depth, 0:height, 0:width]  # 3 x d x h x w
        spatial_emb = torch.tanh(prediction[0:3]) + xyzm_s  # 3 x d x h x w
        sigma = prediction[3:3 + n_sigma]  # n_sigma x h x w

        instance_map = torch.zeros(depth, height, width).short().cuda()
        unique_instances = gt_instance.unique()
        unique_instances = unique_instances[unique_instances != 0]

        for id in unique_instances:
            mask = gt_instance.eq(id).view(1, depth, height, width)
            center = spatial_emb[mask.expand_as(spatial_emb)].view(
                3, -1).mean(1).view(3, 1, 1, 1)  # 3 x 1 x 1 x 1
            s = sigma[mask.expand_as(sigma)].view(n_sigma, -1).mean(1).view(n_sigma, 1, 1, 1)

            s = torch.exp(s * 10)  # n_sigma x 1 x 1
            dist = torch.exp(-1 * torch.sum(torch.pow(spatial_emb - center, 2) * s, 0))
            proposal = (dist > 0.5)
            instance_map[proposal] = id.item()  # TODO

        return instance_map

    def cluster(self, prediction, n_sigma=3, seed_thresh=0.9, fg_thresh=0.5, min_mask_sum=0, min_unclustered_sum=0,
                min_object_size=36):

        """
                Parameters
                ----------
                prediction :  PyTorch Tensor
                        Model Prediction (7, D, H, W)
                n_sigma: int, default = 3
                        Number of dimensions in Raw Image
                seed_thresh : float, default=0.9
                        Seediness Threshold defines which pixels are considered to identify object centres
                fg_thresh: float, default=0.5
                        Foreground Threshold defines which pixels are considered to the form the Foreground
                        and which would need to be clustered into unique objects
                min_mask_sum: int
                        Only start creating instances, if there are at least `min_mask_sum` pixels in foreground!
                min_unclustered_sum: int
                        Stop when the number of seed candidates are less than `min_unclustered_sum`
                min_object_size: int
                    Predicted Objects below this threshold are ignored

                Returns
                ----------
                instance: PyTorch Tensor (D, H, W)
                        instance segmentation
               """

        depth, height, width = prediction.size(1), prediction.size(2), prediction.size(3)
        xyzm_s = self.xyzm[:, 0:depth, 0:height, 0:width]

        spatial_emb = torch.tanh(prediction[0:3]) + xyzm_s  # 3 x d x h x w

        sigma = prediction[3:3 + n_sigma]  # n_sigma x d x h x w
        seed_map = torch.sigmoid(prediction[3 + n_sigma:3 + n_sigma + 1])  # 1 x d x h x w
        instance_map = torch.zeros(depth, height, width).short()

        count = 1
        mask = seed_map > fg_thresh
        if mask.sum() > min_mask_sum:  # top level decision: only start creating instances, if there are atleast `min_mask_sum` pixels in foreground!

            spatial_emb_masked = spatial_emb[mask.expand_as(spatial_emb)].view(n_sigma, -1)
            sigma_masked = sigma[mask.expand_as(sigma)].view(n_sigma, -1)
            seed_map_masked = seed_map[mask].view(1, -1)

            unclustered = torch.ones(mask.sum()).short().cuda()
            instance_map_masked = torch.zeros(mask.sum()).short().cuda()

            while (
                    unclustered.sum() > min_unclustered_sum):  # stop when the seed candidates are less than min_unclustered_sum
                seed = (seed_map_masked * unclustered.float()).argmax().item()
                seed_score = (seed_map_masked * unclustered.float()).max().item()
                if seed_score < seed_thresh:
                    break
                center = spatial_emb_masked[:, seed:seed + 1]
                unclustered[seed] = 0

                s = torch.exp(sigma_masked[:, seed:seed + 1] * 10)
                dist = torch.exp(-1 * torch.sum(torch.pow(spatial_emb_masked - center, 2) * s, 0))

                proposal = (dist > 0.5).squeeze()
                if proposal.sum() > min_object_size:
                    if unclustered[proposal].sum().float() / proposal.sum().float() > 0.5:
                        instance_map_masked[proposal.squeeze()] = count
                        instance_mask = torch.zeros(depth, height, width).short()
                        instance_mask[mask.squeeze().cpu()] = proposal.short().cpu()
                        count += 1
                unclustered[proposal] = 0

            instance_map[mask.squeeze().cpu()] = instance_map_masked.cpu()

        return instance_map

    def cluster_local_maxima(self, prediction, n_sigma=3, fg_thresh=0.5, min_mask_sum=0, min_unclustered_sum=0,
                             min_object_size=36):

        """
            Parameters
            ----------
            prediction :  PyTorch Tensor
                    Model Prediction (7, D, H, W)
            n_sigma: int, default = 3
                    Number of dimensions in Raw Image
            fg_thresh: float, default=0.5
                    Foreground Threshold defines which pixels are considered to the form the Foreground
                    and which would need to be clustered into unique objects
            min_mask_sum: int
                    Only start creating instances, if there are at least `min_mask_sum` pixels in foreground!
            min_unclustered_sum: int
                    Stop when the number of seed candidates are less than `min_unclustered_sum`
            min_object_size: int
                Predicted Objects below this threshold are ignored

            Returns
            ----------
            instance: PyTorch Tensor (D, H, W)
                    instance segmentation
           """

        from scipy.ndimage import gaussian_filter
        depth, height, width = prediction.size(1), prediction.size(2), prediction.size(3)
        xyzm_s = self.xyzm[:, 0:depth, 0:height, 0:width]
        spatial_emb = torch.tanh(prediction[0:3]) + xyzm_s  # 3 x d x h x w
        sigma = prediction[3:3 + n_sigma]  # n_sigma x d x h x w
        seed_map = torch.sigmoid(prediction[3 + n_sigma:3 + n_sigma + 1])  # 1 x h x w
        instance_map = torch.zeros(depth, height, width).short()
        # instances = []  # list
        count = 1
        mask_fg = seed_map > fg_thresh

        seed_map_cpu = seed_map.cpu().detach().numpy()
        seed_map_cpu_smooth = gaussian_filter(seed_map_cpu[0], sigma=(1, 2, 2))  # TODO
        coords = peak_local_max(seed_map_cpu_smooth)
        zeros = np.zeros((coords.shape[0], 1), dtype=np.uint8)
        coords = np.hstack((zeros, coords))

        mask_local_max_cpu = np.zeros(seed_map_cpu.shape, dtype=np.bool)
        mask_local_max_cpu[tuple(coords.T)] = True
        mask_local_max = torch.from_numpy(mask_local_max_cpu).bool().cuda()

        mask_seed = mask_fg * mask_local_max
        if mask_fg.sum() > min_mask_sum:
            spatial_emb_fg_masked = spatial_emb[mask_fg.expand_as(spatial_emb)].view(n_sigma, -1)  # fg candidate pixels
            spatial_emb_seed_masked = spatial_emb[mask_seed.expand_as(spatial_emb)].view(n_sigma,
                                                                                         -1)  # seed candidate pixels

            sigma_seed_masked = sigma[mask_seed.expand_as(sigma)].view(n_sigma, -1)  # sigma for seed candidate pixels
            seed_map_seed_masked = seed_map[mask_seed].view(1, -1)  # seediness for seed candidate pixels

            unprocessed = torch.ones(mask_seed.sum()).short().cuda()  # unprocessed seed candidate pixels
            unclustered = torch.ones(mask_fg.sum()).short().cuda()  # unclustered fg candidate pixels
            instance_map_masked = torch.zeros(mask_fg.sum()).short().cuda()
            while (unprocessed.sum() > min_unclustered_sum):
                seed = (seed_map_seed_masked * unprocessed.float()).argmax().item()
                center = spatial_emb_seed_masked[:, seed:seed + 1]
                unprocessed[seed] = 0
                s = torch.exp(sigma_seed_masked[:, seed:seed + 1] * 10)
                dist = torch.exp(-1 * torch.sum(torch.pow(spatial_emb_fg_masked - center, 2) * s, 0))
                proposal = (dist > 0.5).squeeze()
                if proposal.sum() > min_object_size:
                    if unclustered[proposal].sum().float() / proposal.sum().float() > 0.5:
                        instance_map_masked[proposal.squeeze()] = count
                        count += 1
                        unclustered[proposal] = 0
                        # note the line above increases false positives, tab back twice to show less objects!
                        # The reason I leave it like so is because the penalty on false-negative nodes is `10` while
                        # penalty on false-positive nodes is `1`.
            instance_map[mask_fg.squeeze().cpu()] = instance_map_masked.cpu()
        return instance_map