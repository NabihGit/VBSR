""" Full assembly of the parts to form the complete network """

from .segnet_parts import *


class SegNet(nn.Module):
    def __init__(self, n_channels = 3, n_classes = 1, bilinear=True):
        super(SegNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

        self.inc = DoubleConv(n_channels, 16)
        self.down1 = Down(16, 32)
        self.down2 = Down(32, 32)
        self.up1 = Up(64, 16, bilinear)
        self.up2 = Up(32, 16, bilinear)
        self.outc = OutConv(16, n_classes)


    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x = self.up1(x3, x2)
        x = self.up2(x, x1)
        logits = self.outc(x)
        logout = torch.tanh(logits)
        return logits