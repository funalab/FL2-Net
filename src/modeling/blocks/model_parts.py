import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from src.modeling.blocks.rnn import ConvLSTM, ConvGRU, ConvBLSTM, ConvBGRU
from einops import rearrange, repeat


class BasicLayers(nn.Module):
    def __init__(self, in_channel, out_channel, mid_channel, kernel_size=3, stride=1, residual=True):
        super().__init__()
        self.residual = residual
        self.basiclayers = nn.Sequential(
            nn.Conv3d(in_channel, mid_channel, kernel_size, stride, int(kernel_size/2)),
            nn.BatchNorm3d(mid_channel),
            nn.ReLU(inplace=True),
            nn.Conv3d(mid_channel, out_channel, kernel_size, stride, int(kernel_size/2)),
            nn.BatchNorm3d(out_channel),
            nn.ReLU(inplace=True)
        )
        if self.residual:
            self.conv_skip = nn.Sequential(
                nn.Conv3d(in_channel, out_channel, kernel_size=1, stride=stride),
                nn.BatchNorm3d(out_channel),
            )
    
    def forward(self, x):
        if self.residual:
            return self.basiclayers(x) + self.conv_skip(x)
        else:
            return self.basiclayers(x)
        

class DownBlock(nn.Module):
    def __init__(self, in_channel, out_channel, mid_channel, kernel_size=3, stride=1, pool_size=2, residual=False):
        super().__init__()
        self.downblock = nn.Sequential(
            nn.MaxPool3d(pool_size, pool_size),
            BasicLayers(in_channel, out_channel, mid_channel, kernel_size, stride, residual=residual)
        )

    def forward(self, x):
        return self.downblock(x)


class UpBlock(nn.Module):
    def __init__(self, in_channel1, in_channel2, out_channel, mid_channel, kernel_size=3, stride=1, pool_size=2, residual=False):
        super().__init__()
        self.up_layer = nn.ConvTranspose3d(in_channel1, in_channel1, pool_size, pool_size, 0)
        self.basiclayers = BasicLayers(in_channel1 + in_channel2, out_channel, mid_channel, kernel_size, stride, residual=residual)

    def forward(self, x1, x2):
        x = torch.cat([self.up_layer(x1), x2], dim=1)
        del x1, x2
        return self.basiclayers(x)


class DownCBlock(nn.Module):
    def __init__(self, in_channel1, in_channel2, out_channel, mid_channel, kernel_size=1, stride=1, pool_size=2):
        super().__init__()
        self.down_layer = nn.MaxPool3d(pool_size)
        self.basiclayers = BasicLayers(in_channel1 + in_channel2, out_channel, mid_channel, 1, stride)

    def forward(self, x1, x2):
        x = torch.cat([self.down_layer(x1), x2], dim=1)
        del x1, x2
        return self.basiclayers(x)
    

class RNNBlock(nn.Module):
    def __init__(self, in_channel, out_channel, kernel_size, rnn_type='lstm', bidirectional=False, patch_size=(128, 128, 128)):
        super().__init__()
        self.bidirectional = bidirectional
        if rnn_type == 'lstm':
            if self.bidirectional:
                self.rnn = ConvBLSTM(in_channel, out_channel, (kernel_size, kernel_size, kernel_size), 1, False, True, False)
            else:
                self.rnn = ConvLSTM(in_channel, out_channel, (kernel_size, kernel_size, kernel_size), 1, False, True, False)
        elif rnn_type == 'gru':
            if self.bidirectional:
                self.rnn = ConvBGRU(in_channel, out_channel, (kernel_size, kernel_size, kernel_size), 1, return_all_layers=False)
            else:
                self.rnn = ConvGRU(in_channel, out_channel, (kernel_size, kernel_size, kernel_size), 1, return_all_layers=False)

    def forward(self, x_seq, return_all=False):
        if return_all:
            return self.rnn(x_seq)
        elif self.bidirectional:
            return self.rnn(x_seq)[len(x_seq)//2]
        else:
            return self.rnn(x_seq)[-1]

