import math
from torch import nn
import torch
import numpy as np

from .coarsenet_parts import *

class CoarseNet(nn.Module):
    """ Reconstruction & Upsampling network
    """

    def __init__(self, in_nc=3, out_nc=3, nf=64, nb=4, scale=4):
        super(CoarseNet, self).__init__()

        # input conv.
        self.conv_in = nn.Sequential(
            nn.Conv2d(in_nc, nf, 3, 1, 1, bias=True),
            nn.ReLU(inplace=True))

        # residual blocks
        self.resblocks = nn.Sequential(*[ResidualBlock(nf) for _ in range(nb)])

        # upsampling
        self.conv_up_cheap = nn.Sequential(
            nn.PixelShuffle(scale),
            nn.ReLU(inplace=True))

        # output conv.
        assert nf//scale//scale > out_nc, 'upsampling channel number must be greater than out_nc.'

        self.conv_out = nn.Conv2d(nf//scale//scale, out_nc, 3, 1, 1, bias=True)


    def forward(self, lr_curr):
        """ lr_curr: the current lr data in shape nchw
            hr_prev_tran: the previous transformed hr_data in shape n(4*4*c)hw
        """

        out = self.conv_in(lr_curr)
        out = self.resblocks(out)
        out = self.conv_up_cheap(out)
        out = self.conv_out(out)

        return out

