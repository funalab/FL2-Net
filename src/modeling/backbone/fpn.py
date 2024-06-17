import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from src.modeling.blocks.model_parts import *

class FPN(nn.Module):

    def __init__(
        self,
        in_channels: list, 
        out_channels: list,
        n_level,
    ):
        super().__init__()

        #self.op_factor = op_factor
        self.lateral_convs = nn.ModuleList()
        self.output_convs = nn.ModuleList()

        for i_ch, o_ch in zip(in_channels, out_channels):
            self.lateral_convs.append(
                nn.Conv3d(i_ch, o_ch, kernel_size=1)
            )
            self.output_convs.append(
                nn.Conv3d(o_ch, o_ch, kernel_size=3, stride=1, padding=1)
            )

    def forward(self, features):
        out = []
        # initial output creation
        x = self.lateral_convs[0](features[-1])
        out.append(self.output_convs[0](x))

        for i, (lateral_conv, output_conv) in enumerate(zip(self.lateral_convs[1:], self.output_convs[1:])):
            x = F.interpolate(x, scale_factor=2.0, mode="nearest")
            x = x + lateral_conv(features[-(i+2)])
            out.append(output_conv(x))
        
        return out
