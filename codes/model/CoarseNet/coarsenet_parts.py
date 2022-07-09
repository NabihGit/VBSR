import math
from torch import nn
import torch
import numpy as np

class _Dumbbells_Block(nn.Module):
    # 哑铃结构 5*5感受野输入 配合3*3分组卷积 普通卷积 1*1卷积 5*5感受野输出
    def __init__(self, dim=16):
        super(_Dumbbells_Block, self).__init__()
        self.relu = nn.PReLU(dim),
        self.conv1 = nn.Conv2d(in_channels=dim, out_channels=dim, kernel_size=3, stride=1, padding=2, dilation=2, bias=False,groups=1)
        self.conv2 = nn.Conv2d(in_channels=dim, out_channels=dim*4, kernel_size=3, stride=1, padding=1, dilation=1, bias=False,groups=4)
        self.conv3 = nn.Conv2d(in_channels=dim*4, out_channels=dim*4, kernel_size=1, stride=1, padding=0, bias=False)
        self.conv4 = nn.Conv2d(in_channels=dim*4, out_channels=dim, kernel_size=3, stride=1, padding=1, dilation=1, bias=False,groups=4)
        self.conv5 = nn.Conv2d(in_channels=dim, out_channels=dim, kernel_size=3, stride=1, padding=2, dilation=2, bias=False,groups=1)

    def forward(self, x):
        identity_data = x
        output = self.conv1(x)
        output = self.conv2(output)
        output = self.conv3(output)
        output = self.conv4(output)
        output = self.conv5(output)
        output *= 0.8
        output = torch.add(output, identity_data)
        return output

class ResidualBlock(nn.Module):
    """ Residual block without batch normalization
    """

    def __init__(self, nf=64):
        super(ResidualBlock, self).__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(nf, nf, 3, 1, 1, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(nf, nf, 3, 1, 1, bias=True))

    def forward(self, x):
        out = self.conv(x) + x

        return out


class RRDBBlock(nn.Module):
    """ Residual block without batch normalization
    """

    def __init__(self, nf=64):
        super(RRDBBlock, self).__init__()

        self.res1 = ResidualBlock(nf)
        self.res2 = ResidualBlock(nf)
        self.res3 = ResidualBlock(nf)

    def forward(self, x, beta1 = 0.9, beta2 = 0.1):
        out1 = self.res1(x)
        x1 = out1*beta1 + x*beta2
        out2 = self.res2(x1)
        x2 = out2*beta1 + x1*beta2
        out3 = self.res3(x2)
        out = out3*beta1 + x*beta2
        return out



