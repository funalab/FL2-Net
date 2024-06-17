import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from ..blocks.model_parts import *


class UNet_Encoder(nn.Module):

    def __init__(
        self, 
        init_channel=2, 
        kernel_size=3, 
        pool_size=2, 
        ap_factor=2, 
        patch_size=(128,128,128), 
        n_down=2, 
        residual=False, 
        out_layer=False
    ):
        super().__init__()
        self.pool_size = pool_size
        self.n_down = n_down

        # Input Layer
        self.in_block = BasicLayers(1, int(init_channel * (ap_factor ** 1)), init_channel, kernel_size, residual=residual)
        # Encoder
        self.encoder = nn.ModuleList()
        for i in range(n_down):
            in_channel = int(init_channel * (ap_factor ** (i+1)))
            out_channel = int(init_channel * (ap_factor ** (i+2)))
            self.encoder.append(
                DownBlock(in_channel, out_channel, in_channel, kernel_size, residual=residual)
            )
        # Output Layer
        if out_layer:
            feat_size = int(patch_size[0] / (ap_factor ** n_down)), int(patch_size[1] / (ap_factor ** n_down)), int(patch_size[2] / (ap_factor ** n_down))
            self.out_block = nn.Sequential(
                nn.AvgPool3d(feat_size),
                Rearrange('b c 1 1 1 -> b c'),
                nn.Linear(out_channel, n_class)
            )
    
    def forward(self, x, prediction=False):
        features = []
        # Input
        x = self.in_block(x)
        features.append(x)
        # Encode
        for el in self.encoder:
            x = el(x)
            features.append(x)
        
        if prediction:
            return self.out_block(x)
        else:
            return features


class UNet_Decoder(nn.Module):

    def __init__(
        self, 
        n_class, 
        in_channel, 
        kernel_size=3, 
        pool_size=2, 
        op_factor=2,
        n_down=2, 
        residual=False
    ):
        super().__init__()
        self.pool_size = pool_size
        self.n_down = n_down

        # Decoder
        self.decoder = nn.ModuleList()
        for i in range(n_down):
            in_channel1 = int(in_channel // (op_factor ** i))
            in_channel2 = int(in_channel // (op_factor ** (i+1)))
            out_channel = int(in_channel // (op_factor ** (i+1)))
            self.decoder.append(
                UpBlock(in_channel1, in_channel2, out_channel, out_channel, kernel_size, residual=residual)
            )
        # Output Layer
        self.out_block = nn.Conv3d(int(in_channel // (op_factor ** n_down)), n_class, 1, 1)

    def forward(self, features):
        # Decode
        x = features[-1]
        out_feature = []

        for i in range(self.n_down):
            x = self.decoder[i](x, features[-(i+2)])
        # Output
        x = self.out_block(x)
        return x


class RUNet_Encoder(nn.Module):

    def __init__(
        self, 
        init_channel=2, 
        kernel_size=3, 
        pool_size=2, 
        ap_factor=2, 
        n_down=2, 
        rnn_type='lstm', 
        seq_len=3, 
        bidirectional=False, 
        patch_size=(128,128,128), 
        seq_low_level=None, 
        residual=False, 
        out_layer=False
    ):
        super().__init__()
        self.seq_len = seq_len
        self.rnn_type = rnn_type
        if seq_low_level is None:
            self.seq_low_level = [seq_len, ] * (n_down + 1)
        else:
            self.seq_low_level = eval(seq_low_level)

        self.encoder = UNet_Encoder(init_channel, kernel_size, pool_size, ap_factor, patch_size, n_down, residual)

        # RNN
        self.rnns = nn.ModuleList()
        for i in range(n_down+1):
            channel = int(init_channel * (ap_factor ** (i+1)))
            in_patch_size = (int(patch_size[0] / (ap_factor**i)), int(patch_size[1] / (ap_factor**i)), int(patch_size[2] / (ap_factor**i)))
            if rnn_type == 'attention':
                self.rnns.append(CrossAttentionBlock(channel, in_patch_size, seq_len))
            else:
                self.rnns.append(RNNBlock(channel, channel, kernel_size=kernel_size, rnn_type=rnn_type, bidirectional=bidirectional, patch_size=in_patch_size))
                
        if out_layer:
            feat_size = int(patch_size[0] / (ap_factor ** n_down)), int(patch_size[1] / (ap_factor ** n_down)), int(patch_size[2] / (ap_factor ** n_down))
            self.out_block = nn.Sequential(
                nn.AvgPool3d(feat_size),
                Rearrange('b c 1 1 1 -> b c'),
                nn.Linear(out_channel, n_class)
            )

    def forward(self, x, prediction=False, return_all=False):
        x = x.transpose(0, 1)
        # Input
        features = []
        for x_t in x:
            features.append(self.encoder(x_t))
        # Encode
        y = []
        for i, rl, s in zip(range(self.encoder.n_down+1), self.rnns, self.seq_low_level):
            # RNN
            feat = torch.stack([f[i] for f in features[-s:]])
            y.append(rl(feat, return_all=return_all)) # (L x C x D x H x W)
        
        if prediction:
            return self.encoder.out_block(y)
        else:
            return y
