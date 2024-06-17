# Copyright (c) Facebook, Inc. and its affiliates.
# Modified by Bowen Cheng from https://github.com/facebookresearch/detr/blob/master/util/misc.py
"""
Misc functions, including distributed helpers.

Mostly copy-paste from torchvision references.
"""
from typing import List, Optional
import yaml
from src.utils.attrdict import AttrDict

import torch
import torch.distributed as dist
from torch import Tensor
from torch import nn
from torch.nn import functional as F



""" cofig file reading """
class SafeLoader_with_tuple(yaml.SafeLoader):
    def construct_python_tuple(self, node):
        return tuple(self.construct_sequence(node))

SafeLoader_with_tuple.add_constructor(
    u'tag:yaml.org,2002:python/tuple',
    SafeLoader_with_tuple.construct_python_tuple)

def load_yaml(*cfg_file):
    cfg_dotdict = AttrDict()
    for cf in cfg_file:
        with open(cf) as f:
            cfg_dotdict.update(yaml.load(f, SafeLoader_with_tuple))
    return cfg_dotdict

def round_up_size(x, n_down):
    stride = 2 ** n_down
    return (x+stride-1) & (-stride)

def round_up_pad(image, n_down):
    d, h, w = image.shape[-3:]
    D, H, W = round_up_size(d, n_down), round_up_size(h, n_down), round_up_size(w, n_down)

    d_pad, h_pad, w_pad = D-d, H-h, W-w
    #pad = (w_pad//2, w_pad-w_pad//2, h_pad//2, h_pad-h_pad//2, d_pad-d_pad//2)
    pad = (0, w_pad, 0, h_pad, 0, d_pad)
    if image.ndim == 6:
        pad_image = torch.stack([
            F.pad(img, pad=pad, mode='reflect')
            for img in image
        ])
    else:
        pad_image = F.pad(image, pad=pad, mode='reflect')
    return pad_image
