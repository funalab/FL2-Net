import torch
from torch import nn
from torch.nn import functional as F
import numpy as np
#from numba import jit

from src.utils.memory import retry_if_cuda_oom
from src.utils.misc import round_up_pad
import src.modeling.backbone.erfnet as erfnet
from src.modeling.postprocess.embedseg_decoder import Cluster_3d
from src.modeling.criterions.embloss import SpatialEmbLoss_3d

class EmbedSeg(nn.Module):
    """
    Main class for semantic segmentation using Encoder-Decoder model.
    """
    def __init__(self, cfg, backbone=None):
        super().__init__()
        # building model ( Branched-ERFNet )
        num_classes = [6, 1]
        if (backbone is None):
            self.backbone = erfnet.Encoder(sum(num_classes), 1)
        else:
            self.backbone = backbone
        self.decoders = nn.ModuleList()
        for n in num_classes:
            self.decoders.append(erfnet.Decoder(n))

        # building loss function
        patchsize = cfg.DATASETS.INPUT_SIZE
        self.criterion = SpatialEmbLoss_3d(
            grid_z=128, grid_y=128, grid_x=128, 
            pixel_z=1, pixel_y=1, pixel_x=1, 
            one_hot=False, n_sigma=3, foreground_weight=cfg.MODEL.LOSS.FOREGROUND_WEIGHT
        )

        # post-processing config
        self.cluster = Cluster_3d(
            grid_z=128, grid_y=128, grid_x=128,
            pixel_z=1, pixel_y=1, pixel_x=1
        )
        self.loss_weight = {'w_inst': cfg.MODEL.LOSS.MASK_WEIGHT, 'w_var': cfg.MODEL.LOSS.VAR_WEIGHT, 'w_seed': cfg.MODEL.LOSS.SEED_WEIGHT}

        # test time config
        self.fg_thresh=cfg.MODEL.TEST.FOREGROUND_THRESHOLD
        self.seed_thresh=cfg.MODEL.TEST.SEED_THRESHOLD
        self.min_mask_sum=cfg.MODEL.TEST.MIN_MASK_SUM
        self.min_unclustered_sum=cfg.MODEL.TEST.MIN_UNCLUSTERED_SUM
        self.min_object_size=cfg.MODEL.TEST.MIN_OBJECT_SIZE

        self.init_output(n_sigma=3)
        self.n_down = 3

    def init_output(self, n_sigma=3):
        with torch.no_grad():
            output_conv = self.decoders[0].output_conv
            print('initialize last layer with size: ',
                  output_conv.weight.size())

            output_conv.weight[:, 0:3, :, :, :].fill_(0)
            output_conv.bias[0:3].fill_(0)

            output_conv.weight[:, 3:3 + n_sigma, :, :, :].fill_(0)
            output_conv.bias[3:3 + n_sigma].fill_(1)

    def forward_features(self, x):
        features = self.backbone(x)
        return torch.cat([decoder.forward(features) for decoder in self.decoders], 1)

    def forward(self, inputs, mode='train', inference=False):
        images = inputs["image"]
        depth, height, width = images.shape[-3:]
        results = {}

        if mode == 'train':
            images = inputs["image"].to('cuda')
            outputs = self.forward_features(images)
            gt_instances = self.prepare_targets(inputs["instances"], inputs.get("centers", None))
            results['loss'] = self.criterion(outputs, gt_instances)
            pred = torch.stack([
                self.cluster.cluster_with_gt(output, gt_instance, n_sigma=3)
                for output, gt_instance in zip(outputs, gt_instances["instances"])
            ])
            results['instances'] = pred
            print(results['loss'])
        else:
            images = round_up_pad(images, n_down=self.n_down).to('cuda')
            outputs = self.forward_features(images)
            cluster = Cluster_3d(
                grid_z=outputs.shape[-3], grid_y=outputs.shape[-2], grid_x=outputs.shape[-1],
                pixel_z=1, pixel_y=1, pixel_x=1
            )
            pred = torch.stack([
                cluster.cluster(output, n_sigma=3,
                                fg_thresh=self.fg_thresh,
                                seed_thresh=self.seed_thresh,
                                min_mask_sum=0,
                                min_unclustered_sum=0,
                                min_object_size=0,)
                for output in outputs
            ])
            results['instances'] = pred[..., :depth, :height, :width]

        return results
    
    def prepare_targets(self, targets, centers=None):
        batch_size, depth, height, width = targets.shape

        if centers == None:
            centers = []
            for bs in range(batch_size):
                target = targets[bs].numpy()
                ids = np.unique(target[target > 0])
                ids = ids[ids != 0]
                centers.append(
                    generate_center_image_3d(target, center='medoid', ids=ids)
                )
            centers = torch.from_numpy(np.stack(centers))

        new_targets = {
            "instances": targets,
            "labels": ((targets > 0) * 1).long(),
            "center_images": centers.long()
        }
        return new_targets
        

#@jit(nopython=True)
def pairwise_python(X):
    """Helper function to compute pairwise Euclidean distance for a matrix of size M x N containing M,  N-dimensional row entries

        Parameters
        ----------
        x : numpy array
            Input Image
        Returns
        -------
        D : numpy array, M x M
            D[i, j] corresponds to the Euclidean Distance between the i th and j th rows of X
        """
    M = X.shape[0]
    N = X.shape[1]
    D = np.empty((M, M), dtype=np.float32)
    for i in range(M):
        for j in range(M):
            d = 0.0
            for k in range(N):
                tmp = X[i, k] - X[j, k]
                d += tmp * tmp
            D[i, j] = np.sqrt(d)
    return D

def generate_center_image_3d(instance, center, ids, one_hot=False, anisotropy_factor=1.0, speed_up=1.0):
    """
        Generates a `center_image` for 3D image crops which is one (True) for all center locations and zero (False) otherwise.

        Parameters
        ----------
        instance: numpy array
            `instance` image containing unique `ids` for each object (ZYX)
        center: string
            One of 'centroid', 'approximate-medoid' or 'medoid'.
        ids: list
            Unique ids corresponding to the objects present in the instance image.
        one_hot: boolean
            This parameter is not used in the 3D setting and will be deprecated.
        speed_up: int
            This computes the centers of crops faster by down-sampling crops along x and y dimensions.

        Returns
        -------
        numpy array: bool
        Center image with center locations set to True
        """

    center_image = np.zeros(instance.shape, dtype=bool)
    instance_downsampled = instance[:, ::int(speed_up), ::int(speed_up)]  # down sample in x and y
    for j, id in enumerate(ids):
        z, y, x = np.where(instance_downsampled == id)
        if len(y) != 0 and len(x) != 0:
            if (center == 'centroid'):
                zm, ym, xm = np.mean(z), np.mean(y), np.mean(x)
            elif (center == 'approximate-medoid'):
                zm_temp, ym_temp, xm_temp = np.median(z), np.median(y), np.median(x)
                imin = np.argmin((x - xm_temp) ** 2 + (y - ym_temp) ** 2 + (anisotropy_factor * (z - zm_temp)) ** 2)
                zm, ym, xm = z[imin], y[imin], x[imin]
            elif (center == 'medoid'):
                dist_matrix = pairwise_python(
                    np.vstack((speed_up * x, speed_up * y, anisotropy_factor * z)).transpose())
                imin = np.argmin(np.sum(dist_matrix, axis=0))
                zm, ym, xm = z[imin], y[imin], x[imin]
            center_image[int(np.round(zm)), int(np.round(speed_up * ym)), int(np.round(speed_up * xm))] = True
    return center_image
