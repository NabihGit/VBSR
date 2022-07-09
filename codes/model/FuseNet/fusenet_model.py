import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from model.networks.base_nets import BaseSequenceGenerator, BaseSequenceDiscriminator
from utils.net_utils import space_to_depth, backward_warp, get_upsampling_func
from utils.data_utils import float32_to_uint8
import random
import torch.backends.cudnn as cudnn

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

class FuseNet(nn.Module):
    """ Reconstruction & Upsampling network
    """

    def __init__(self, in_nc=3, out_nc=3, nf=3, nb=2):
        super(FuseNet, self).__init__()

        # input conv.
        self.conv_in = nn.Sequential(
            nn.Conv2d(in_nc, nf, 3, 1, 1, bias=True),
            nn.ReLU(inplace=True))

        # residual blocks
        self.resblocks = nn.Sequential(*[ResidualBlock(in_nc) for _ in range(2)])

        self.resblocks2 = nn.Sequential(*[ResidualBlock(in_nc) for _ in range(2)])

        # output conv.
        self.conv_out = nn.Conv2d(nf, out_nc, 3, 1, 1, bias=True)


    def forward(self, x):
        """ lr_curr: the current lr data in shape nchw
            hr_prev_tran: the previous transformed hr_data in shape n(4*4*c)hw
        """

        fea_map = self.conv_in(x)

        res1 = self.resblocks(x)
        res2 = self.resblocks2(x)

        temp = fea_map*res1 + (1-fea_map)*res2

        out = self.conv_out(temp)

        return out


if __name__ == "__main__":
    from torchsummary import summary

    lr_img_size = (3, 320, 320)
    hr_img_size = (3, 320*4, 320*4)
    cpu_cuda = 'cuda'

    print("===> GPU SET UP & RANDOM SEED")
    cuda = True
    if cuda and not torch.cuda.is_available():
        raise Exception("No GPU found")
    seed = random.randint(1, 10000)
    print("Random Seed: ", seed)
    torch.manual_seed(seed)
    if cuda:
        torch.cuda.manual_seed(seed)
    cudnn.benchmark = True

    print("===> Building FuseNet Model")
    model = FuseNet()
    checkpoint = torch.load('L:/Code/VBSR/output/model_pth/FineNet/fine_net_0001.pth')
    model.load_state_dict(checkpoint)

    criterion = nn.MSELoss()

    print("===> Setting GPU")
    if cuda:
        model_coarse = model.cuda()
        criterion = criterion.cuda()
    model.eval()

    print(model)
    summary(model, [lr_img_size, lr_img_size, hr_img_size], batch_size=1, device=cpu_cuda)

    print("===> Test")
    lr_data_curr = torch.rand(5, 3, 320, 320, dtype=torch.float32).cuda().unsqueeze(0)
    hr_data = model.forward_sequence(lr_data_curr)
    print("hr_data shape is {}".format(hr_data['hr_data'].squeeze(0).shape))

