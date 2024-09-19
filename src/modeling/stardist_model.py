import torch
from torch import nn
from torch.nn import functional as F
import numpy as np

from src.utils.memory import retry_if_cuda_oom
from src.modeling.build import *
from src.modeling.criterions.stardist_loss import L1BCELoss
from src.utils.misc import round_up_pad

from stardist import edt_prob, star_dist3D, Rays_GoldenSpiral
from stardist import non_maximum_suppression_3d, polyhedron_to_label
from stardist.matching import relabel_sequential

class StarDist(nn.Module):
    """
    Main class for instance segmentation using StarDist.
    """
    def __init__(self, cfg, backbone=None):
        super().__init__()
        # building model
        if (backbone is None):
            self.backbone = build_backbone(cfg)
        else:
            self.backbone = backbone
        
        self.pixel_decoder = build_pixel_decoder(cfg)
        self.criterion = L1BCELoss()

        num_ray = cfg.MODEL.STARDIST.NUM_RAY
        self.rays = Rays_GoldenSpiral(num_ray)
        self.max_dist = 10000
        
        conv_after_unet = cfg.MODEL.SEM_SEG_HEAD.NUM_CLASSES
        # Output Layer for ray
        self.out_block_ray = nn.Sequential(
            nn.Conv3d(conv_after_unet, num_ray, 1, 1),
            nn.ReLU(inplace=True),
        )

        # Output Layer for probability
        self.out_block_prob = nn.Sequential(
            nn.Conv3d(conv_after_unet, 1, 1, 1),
        )
        self.n_down = cfg.MODEL.BACKBONE.NUM_LEVEL-1
        self.nms_thresh = cfg.MODEL.TEST.NMS_THRESHOLD
        self.prob_thresh = cfg.MODEL.TEST.PROB_THRESHOLD

    def forward_features(self, x):
        features = self.backbone(x)
        feat_after_unet = self.pixel_decoder(features)
        outputs = {}
        outputs["dists"] = self.out_block_ray(feat_after_unet)
        outputs["probs"] = self.out_block_prob(feat_after_unet)
        return outputs
    
    
    def forward(self, inputs, mode='train', inference=False):
        images = inputs["image"]
        depth, height, width = images.shape[-3:]
        results = {}
        if mode == 'train':
            images = inputs["image"].to('cuda')
            outputs = self.forward_features(images)
            pred_dists = outputs["dists"]
            pred_probs = outputs["probs"]
            gt_instances = self.prepare_targets(inputs["instances"])
            results['loss'] = self.criterion(pred_probs, pred_dists, gt_instances)
            print(results['loss'])
            pred = torch.squeeze((pred_probs > 0.5) * 1, dim=1)
        else:
            images = round_up_pad(images, n_down=self.n_down).to('cuda')
            outputs = self.forward_features(images)
            pred = torch.stack([
                self.instance_inference(dist, prob) for dist, prob in zip(outputs["dists"], outputs["probs"])
            ])

        results['instances'] = pred[..., :depth, :height, :width]
        return results

    def prepare_targets(self, targets):
        batch_size = targets.shape[0]
        probs = []
        dists = []
        for bs in range(batch_size):
            target = targets[bs].numpy()
            distances = star_dist3D(target, self.rays)
            if self.max_dist:
                distances[distances > self.max_dist] = self.max_dist
            dists.append(rearrange(distances, 'd h w r -> r d h w'))
            # create prob gt
            probs.append(np.expand_dims(edt_prob(target), 0))

        new_targets = {
            "distances": torch.from_numpy(np.stack(dists)),
            "probs": torch.from_numpy(np.stack(probs))
        }
        return new_targets
    
    def instance_inference(self, dist, prob):
        out_shape = dist.shape[1:]
        dist = np.transpose(dist.detach().cpu().numpy(), (1, 2, 3, 0))
        prob = torch.sigmoid(prob).detach().cpu().numpy().squeeze()

        points, probi, disti = non_maximum_suppression_3d(dist, prob, self.rays, nms_thresh=self.nms_thresh, prob_thresh=self.prob_thresh)
        labels = polyhedron_to_label(disti, points, rays=self.rays, prob=probi, shape=out_shape)
        labels, _ , _ = relabel_sequential(labels)

        return torch.from_numpy(labels.astype(np.uint8))
