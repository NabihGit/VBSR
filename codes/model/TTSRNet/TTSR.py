from model.TTSRNet import MainNet, LTE, SearchTransfer
from utils.net_utils import space_to_depth, backward_warp, get_upsampling_func

import torch
import torch.nn as nn
import torch.nn.functional as F


class TTSR(nn.Module):
    def __init__(self, num_res_blocks='16+16+8+4',n_feats = 64, res_scale = 1.):
        super(TTSR, self).__init__()
        self.upsample_func = get_upsampling_func(4, 'BD')
        self.num_res_blocks = list( map(int, num_res_blocks.split('+')) )
        self.MainNet = MainNet.MainNet(num_res_blocks=self.num_res_blocks, n_feats=n_feats,
            res_scale=res_scale)
        self.LTE      = LTE.LTE(requires_grad=True)
        self.LTE_copy = LTE.LTE(requires_grad=False) ### used in transferal perceptual loss
        self.SearchTransfer = SearchTransfer.SearchTransfer()

    def forward(self, lr=None, lrsr=None, ref=None, refsr=None, sr=None):
        if (type(sr) != type(None)):
            ### used in transferal perceptual loss
            self.LTE_copy.load_state_dict(self.LTE.state_dict())
            sr_lv1, sr_lv2, sr_lv3 = self.LTE_copy((sr + 1.) / 2.)
            return sr_lv1, sr_lv2, sr_lv3

        _, _, lrsr_lv3  = self.LTE((lrsr.detach() + 1.) / 2.)#Q
        _, _, refsr_lv3 = self.LTE((refsr.detach() + 1.) / 2.)#K

        ref_lv1, ref_lv2, ref_lv3 = self.LTE((ref.detach() + 1.) / 2.)#V

        S, T_lv3, T_lv2, T_lv1 = self.SearchTransfer(lrsr_lv3, refsr_lv3, ref_lv1, ref_lv2, ref_lv3)

        sr = self.MainNet(lr, S, T_lv3, T_lv2, T_lv1)

        return sr, S, T_lv3, T_lv2, T_lv1

    def forward_sequence(self, lr_data, ref_pre, refsr_pre):
        """
            Parameters:
                :param lr_data: lr data in shape ntchw
        """

        n, t, c, lr_h, lr_w = lr_data.size()
        hr_h, hr_w = lr_h * 4, lr_w * 4

        lrsr = self.upsample_func(lr_data.squeeze(0))-
        # compute the first hr data
        hr_data = []
        _, _, lrsr_lv3  = self.LTE((lrsr[0,...].unsqueeze(0).detach() + 1.) / 2.)#Q
        _, _, refsr_lv3 = self.LTE((refsr_pre.unsqueeze(0).detach() + 1.) / 2.)#K
        ref_lv1, ref_lv2, ref_lv3 = self.LTE((ref_pre.unsqueeze(0).detach() + 1.) / 2.)#V

        S, T_lv3, T_lv2, T_lv1 = self.SearchTransfer(lrsr_lv3, refsr_lv3, ref_lv1, ref_lv2, ref_lv3)
        hr_prev = self.MainNet(lr_data[0,...], S, T_lv3, T_lv2, T_lv1)
        hr_data.append(hr_prev)

        # compute the remaining hr data
        for i in range(1, t):
            _, _, lrsr_lv3 = self.LTE((lrsr[i, ...].unsqueeze(0).detach() + 1.) / 2.)  # Q
            _, _, refsr_lv3 = self.LTE((hr_prev.unsqueeze(0).detach() + 1.) / 2.)  # K
            ref_lv1, ref_lv2, ref_lv3 = self.LTE((lr_data[i-1,...].unsqueeze(0).detach() + 1.) / 2.)  # V

            S, T_lv3, T_lv2, T_lv1 = self.SearchTransfer(lrsr_lv3, refsr_lv3, ref_lv1, ref_lv2, ref_lv3)
            hr_curr = self.MainNet(lr_data[i, ...], S, T_lv3, T_lv2, T_lv1)

            # save and update
            hr_data.append(hr_curr)
            hr_prev = hr_curr

        hr_data = torch.stack(hr_data, dim=1)  # n,t,c,hr_h,hr_w


        return hr_data