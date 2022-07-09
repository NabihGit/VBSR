import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from model.networks.base_nets import BaseSequenceGenerator, BaseSequenceDiscriminator
from utils.net_utils import space_to_depth, backward_warp, get_upsampling_func
from utils.data_utils import float32_to_uint8
import random
import torch.backends.cudnn as cudnn


# -------------------- generator modules -------------------- #
class FNet(nn.Module):
    """ Optical flow estimation network
    """

    def __init__(self, in_nc):
        super(FNet, self).__init__()

        self.encoder1 = nn.Sequential(
            nn.Conv2d(2*in_nc, 32, 3, 1, 1, bias=True),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(32, 32, 3, 1, 1, bias=True),
            nn.LeakyReLU(0.2, inplace=True),
            nn.MaxPool2d(2, 2))

        self.encoder2 = nn.Sequential(
            nn.Conv2d(32, 64, 3, 1, 1, bias=True),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64, 64, 3, 1, 1, bias=True),
            nn.LeakyReLU(0.2, inplace=True),
            nn.MaxPool2d(2, 2))

        self.encoder3 = nn.Sequential(
            nn.Conv2d(64, 128, 3, 1, 1, bias=True),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(128, 128, 3, 1, 1, bias=True),
            nn.LeakyReLU(0.2, inplace=True),
            nn.MaxPool2d(2, 2))

        self.decoder1 = nn.Sequential(
            nn.Conv2d(128, 256, 3, 1, 1, bias=True),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(256, 256, 3, 1, 1, bias=True),
            nn.LeakyReLU(0.2, inplace=True))

        self.decoder2 = nn.Sequential(
            nn.Conv2d(256, 128, 3, 1, 1, bias=True),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(128, 128, 3, 1, 1, bias=True),
            nn.LeakyReLU(0.2, inplace=True))

        self.decoder3 = nn.Sequential(
            nn.Conv2d(128, 64, 3, 1, 1, bias=True),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64, 64, 3, 1, 1, bias=True),
            nn.LeakyReLU(0.2, inplace=True))

        self.flow = nn.Sequential(
            nn.Conv2d(64, 32, 3, 1, 1, bias=True),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(32, 2, 3, 1, 1, bias=True))

    def forward(self, x1, x2):
        """ Compute optical flow from x1 to x2
        """

        out = self.encoder1(torch.cat([x1, x2], dim=1))
        out = self.encoder2(out)
        out = self.encoder3(out)
        out = F.interpolate(
            self.decoder1(out), scale_factor=2, mode='bilinear', align_corners=False)
        out = F.interpolate(
            self.decoder2(out), scale_factor=2, mode='bilinear', align_corners=False)
        out = F.interpolate(
            self.decoder3(out), scale_factor=2, mode='bilinear', align_corners=False)
        out = torch.tanh(self.flow(out)) * 24  # 24 is the max velocity

        return out


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


class SRNet(nn.Module):
    """ Reconstruction & Upsampling network
    """

    def __init__(self, in_nc=3, out_nc=3, nf=64, nb=10, upsample_func=None,
                 scale=4):
        super(SRNet, self).__init__()

        # input conv.
        self.conv_in = nn.Sequential(
            nn.Conv2d((scale**2 + 1) * in_nc, nf, 3, 1, 1, bias=True),
            nn.ReLU(inplace=True))

        # residual blocks
        self.resblocks = nn.Sequential(*[ResidualBlock(nf) for _ in range(nb)])

        # upsampling
        self.conv_up = nn.Sequential(
            nn.ConvTranspose2d(nf, nf, 3, 2, 1, output_padding=1, bias=True),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(nf, nf, 3, 2, 1, output_padding=1, bias=True),
            nn.ReLU(inplace=True))

        self.conv_up_cheap = nn.Sequential(
            nn.PixelShuffle(4),
            nn.ReLU(inplace=True))

        # output conv.
        self.conv_out = nn.Conv2d(4, out_nc, 3, 1, 1, bias=True)

        # upsampling function
        self.upsample_func = upsample_func

    def forward(self, lr_curr, hr_prev_tran):
        """ lr_curr: the current lr data in shape nchw
            hr_prev_tran: the previous transformed hr_data in shape n(4*4*c)hw
        """

        out = self.conv_in(torch.cat([lr_curr, hr_prev_tran], dim=1))
        out = self.resblocks(out)
        out = self.conv_up_cheap(out)
        out = self.conv_out(out)
        # out += self.upsample_func(lr_curr)

        return out


class FineNet(BaseSequenceGenerator):
    """ Frame-recurrent network proposed in https://arxiv.org/abs/1801.04590
    """

    def __init__(self, in_nc=3, out_nc=3, nf=64, nb=10, degradation='BD',
                 scale=4):
        super(FineNet, self).__init__()

        self.scale = scale

        # get upsampling function according to the degradation mode
        self.upsample_func = get_upsampling_func(self.scale, degradation)

        # define fnet & srnet
        self.fnet = FNet(in_nc)
        self.srnet = SRNet(in_nc, out_nc, nf, nb, self.upsample_func)

    def generate_dummy_input(self, lr_size):
        c, lr_h, lr_w = lr_size
        s = self.scale

        lr_curr = torch.rand(1, c, lr_h, lr_w, dtype=torch.float32)
        lr_prev = torch.rand(1, c, lr_h, lr_w, dtype=torch.float32)
        hr_prev = torch.rand(1, c, s * lr_h, s * lr_w, dtype=torch.float32)

        data_dict = {
            'lr_curr': lr_curr,
            'lr_prev': lr_prev,
            'hr_prev': hr_prev
        }

        return data_dict

    def forward(self, lr_curr, lr_prev, hr_prev):
        """
            Parameters:
                :param lr_curr: the current lr data in shape nchw
                :param lr_prev: the previous lr data in shape nchw
                :param hr_prev: the previous hr data in shape nc(4h)(4w)
        """

        # estimate lr flow (lr_curr -> lr_prev)
        lr_flow = self.fnet(lr_curr, lr_prev)

        # pad if size is not a multiple of 8
        pad_h = lr_curr.size(2) - lr_curr.size(2) // 8 * 8
        pad_w = lr_curr.size(3) - lr_curr.size(3) // 8 * 8
        lr_flow_pad = F.pad(lr_flow, (0, pad_w, 0, pad_h), 'reflect')

        # upsample lr flow
        hr_flow = self.scale * self.upsample_func(lr_flow_pad)

        # warp hr_prev
        hr_prev_warp = backward_warp(hr_prev, hr_flow)

        # compute hr_curr
        hr_curr = self.srnet(lr_curr, space_to_depth(hr_prev_warp, self.scale))

        return hr_curr

    def forward_sequence(self, lr_data):
        """
            Parameters:
                :param lr_data: lr data in shape ntchw
        """

        n, t, c, lr_h, lr_w = lr_data.size()
        hr_h, hr_w = lr_h * self.scale, lr_w * self.scale

        # calculate optical flows
        lr_prev = lr_data[:, :-1, ...].reshape(n * (t - 1), c, lr_h, lr_w)
        lr_curr = lr_data[:, 1:, ...].reshape(n * (t - 1), c, lr_h, lr_w)

        # pad if size is not a multiple of 8
        pad_h = lr_curr.size(2) - lr_curr.size(2) // 8 * 8
        pad_w = lr_curr.size(3) - lr_curr.size(3) // 8 * 8
        lr_flow = self.fnet(lr_curr, lr_prev)  # n*(t-1),2,h,w
        lr_flow_pad = F.pad(lr_flow, (0, pad_w, 0, pad_h), 'reflect')

        # upsample lr flows
        hr_flow = self.scale * self.upsample_func(lr_flow_pad)
        hr_flow = hr_flow.view(n, (t - 1), 2, hr_h, hr_w)

        # compute the first hr data
        hr_data = []
        hr_prev = self.srnet(
            lr_data[:, 0, ...],
            torch.zeros(n, (self.scale ** 2) * c, lr_h, lr_w, dtype=torch.float32,
                        device=lr_data.device))
        hr_data.append(hr_prev)

        # compute the remaining hr data
        for i in range(1, t):
            # warp hr_prev
            hr_prev_warp = backward_warp(hr_prev, hr_flow[:, i - 1, ...])

            # compute hr_curr
            hr_curr = self.srnet(
                lr_data[:, i, ...],
                space_to_depth(hr_prev_warp, self.scale))

            # save and update
            hr_data.append(hr_curr)
            hr_prev = hr_curr

        hr_data = torch.stack(hr_data, dim=1)  # n,t,c,hr_h,hr_w

        # construct output dict
        ret_dict = {
            'hr_data': hr_data,  # n,t,c,hr_h,hr_w
            'hr_flow': hr_flow,  # n,t,2,hr_h,hr_w
            'lr_prev': lr_prev,  # n(t-1),c,lr_h,lr_w
            'lr_curr': lr_curr,  # n(t-1),c,lr_h,lr_w
            'lr_flow': lr_flow,  # n(t-1),2,lr_h,lr_w
        }

        return ret_dict

    def infer_sequence(self, lr_data, device):
        """T
            Parameters:
                :param lr_data: torch.FloatTensor in shape tchw
                :param device: torch.device

                :return hr_seq: uint8 np.ndarray in shape tchw
        """

        # setup params
        tot_frm, c, h, w = lr_data.size()
        s = self.scale

        # forward
        hr_seq = []
        lr_prev = torch.zeros(1, c, h, w, dtype=torch.float32).to(device)
        hr_prev = torch.zeros(
            1, c, s * h, s * w, dtype=torch.float32).to(device)

        for i in range(tot_frm):
            with torch.no_grad():
                self.eval()

                lr_curr = lr_data[i: i + 1, ...].to(device)
                hr_curr = self.forward(lr_curr, lr_prev, hr_prev)
                lr_prev, hr_prev = lr_curr, hr_curr

                hr_frm = hr_curr.squeeze(0).cpu().numpy()  # chw|rgb|uint8

            hr_seq.append(float32_to_uint8(hr_frm))

        return np.stack(hr_seq).transpose(0, 2, 3, 1)  # thwc



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

    print("===> Building FineNet Model")
    model = FineNet()
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

