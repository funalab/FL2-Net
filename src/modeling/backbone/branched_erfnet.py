import torch
import torch.nn as nn
import src.modeling.backbone.erfnet as erfnet
from src.modeling.blocks.model_parts import RNNBlock, CrossAttentionBlock


class BranchedERFNet_3d(nn.Module):
    def __init__(self, num_classes, input_channels=1, encoder=None, rnn=False, rnn_type=None, bidirectional=False, seq_len=1):
        super().__init__()

        print('Creating Branched Erfnet 3D with {} outputs'.format(num_classes))
        if (encoder is None):
            self.encoder = erfnet.Encoder(sum(num_classes), input_channels)
        else:
            self.encoder = encoder

        self.decoders = nn.ModuleList()
        for n in num_classes:
            self.decoders.append(erfnet.Decoder(n))

    def init_output(self, n_sigma=1):
        with torch.no_grad():
            output_conv_pre = self.decoders[0].output_conv
            output_conv_tra = self.decoders[2].output_conv
            print('initialize last layer with size: ',
                  output_conv_pre.weight.size() + output_conv_tra.weight.size())

            output_conv_pre.weight[:, 0:3, :, :, :].fill_(0)
            output_conv_pre.bias[0:3].fill_(0)

            output_conv_pre.weight[:, 3:3 + n_sigma, :, :, :].fill_(0)
            output_conv_pre.bias[3:3 + n_sigma].fill_(1)

            output_conv_tra.weight[:, 0:3, :, :, :].fill_(0)
            output_conv_tra.bias[0:3].fill_(0)

    def forward(self, input, only_encode=False):
        if only_encode:
            return self.encoder.forward(input, predict=True)
        else:
            output = self.encoder(input)

        return torch.cat([decoder.forward(output) for decoder in self.decoders], 1)


