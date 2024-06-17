import numpy as np
import torch
import torch.nn.functional as F
from src.utils.utils import mirror_extension_image
import tifffile

def stitch(images, model, patchsize, ch_conf={}, stride=None, ndim=3):

    crop_size = images.shape[-ndim:]
    assert images.shape[0] == 1
    
    if stride == None:
        stride = [int(i / 2) for i in patchsize]
    sh = [int(i / 2) for i in stride]
    ''' calculation for pad size'''
    if np.min(patchsize) > np.max(np.array(crop_size) + np.array(sh)*2):
        pad_size = patchsize
    else:
        pad_size = []
        for axis in range(ndim):
            if (crop_size[axis] + 2*sh[axis] - patchsize[axis]) % stride[axis] == 0:
                stride_num = int((crop_size[axis] + 2*sh[axis] - patchsize[axis]) / stride[axis])
            else:
                stride_num = int((crop_size[axis] + 2*sh[axis] - patchsize[axis]) / stride[axis]) + 1
            pad_size.append(int(stride[axis] * stride_num + patchsize[axis]))

    pre_features = {key: torch.zeros([1, num_ch, ] + pad_size) for key, num_ch, in model.ch_conf.items()}
    
    images = mirror_extension_image(image=images, ndim=ndim, length=int(np.max(patchsize)))
    images = images[...,
                    patchsize[0]-sh[0] : patchsize[0]-sh[0]+pad_size[0],
                    patchsize[1]-sh[1] : patchsize[1]-sh[1]+pad_size[1],
                    patchsize[2]-sh[2] : patchsize[2]-sh[2]+pad_size[2]]

    for z in range(0, pad_size[0]-stride[0], stride[0]):
        for y in range(0, pad_size[1]-stride[1], stride[1]):
            for x in range(0, pad_size[2]-stride[2], stride[2]):
                x_patch = torch.Tensor(images[..., z:z+patchsize[0], y:y+patchsize[1], x:x+patchsize[2]])
                outputs = model.forward_features(x_patch.to('cuda'))
                # update features
                for key, feat in outputs.items():
                    pre_features[key][:, :, z:z+stride[0], y:y+stride[1], x:x+stride[2]] += feat[:, :, sh[0]:-sh[0], sh[1]:-sh[1], sh[2]:-sh[2]].detach().cpu()

    return {key: feat[..., :crop_size[0], :crop_size[1], :crop_size[2]] for key, feat in pre_features.items()}
