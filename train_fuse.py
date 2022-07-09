import argparse
import os
import copy
import torch
from torch import nn
#from torchprofile import profile_macs as profile
import torch.optim as optim
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader
from utils.dataloader import TestDataset
from utils.util import psnr, ssim
from utils.net_utils import get_upsampling_func
import torchvision.transforms as transforms
from tqdm import tqdm
import random
from torch.autograd import Variable
import matplotlib.pyplot as plt
from functions.function import AverageMeter
from functions.seg_blcok import MeshBlcok
from tensorboardX import SummaryWriter
from model.FineNet import FineNet
from model.CoarseNet import CoarseNet
from model.SegNet import SegNet
from model.FuseNet import FuseNet
from model.TecoGan import FRNet
import numpy as np
import cv2

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--outputs-dir', type=str, default='output/')
    parser.add_argument('--outimg-dir', type=str, default='output/image/')
    parser.add_argument('--outpth-dir', type=str, default='output/model_pth/')
    parser.add_argument("--weights", default='', type=str, help="path to latest checkpoint (default: none)")
    parser.add_argument("--cuda", default=True, help="use cuda?")
    parser.add_argument("--start-epoch", default=1, type=int, help="manual epoch number (useful on restarts)")
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--batchSize', type=int, default=12)
    parser.add_argument('--num-epochs', type=int, default=24)
    parser.add_argument('--threads', type=int, default=0)
    parser.add_argument('--LR_path', type=str, default='/media/test/Cheng_Disp/Code/BGSR/c_dataset/lr_x8', help='path of the LR images')#INPUT
    parser.add_argument('--HR_path', type=str, default='/media/test/Cheng_Disp/Code/BGSR/c_dataset/lr_x2', help='path of the HR images')#GT
    args = parser.parse_args()

    args.outimg_dir = os.path.join(args.outimg_dir, 'fuse_img_seg01')
    if not os.path.exists(args.outimg_dir):
        os.makedirs(args.outimg_dir)

    args.outpth_dir = os.path.join(args.outpth_dir, 'FuseNet_seg01')
    if not os.path.exists(args.outpth_dir):
        os.makedirs(args.outpth_dir)

    print("===> GPU SET UP & RANDOM SEED")
    cuda = args.cuda
    if cuda and not torch.cuda.is_available():
        raise Exception("No GPU found")
    args.seed = random.randint(1, 10000)
    print("Random Seed: ", args.seed)
    torch.manual_seed(args.seed)
    if cuda:
        torch.cuda.manual_seed(args.seed)
    cudnn.benchmark = True

    print("===> Loading training dataset")
    dataset = TestDataset(args.HR_path, args.LR_path)
    dataloader = DataLoader(dataset, batch_size=args.batchSize, shuffle=False)

    print("===> Building Segment Model")
    model_seg = SegNet()
    checkpoint_seg = torch.load('output/model_pth/SegNet/segnet_0001.pth')
    model_seg.load_state_dict(checkpoint_seg)

    print("===> Building CoarseNet Model")
    model_coarse = CoarseNet()
    checkpoint = torch.load('output/model_pth/CoarseNet/coarse_net_0024.pth')
    model_coarse.load_state_dict(checkpoint)

    #print("===> Building FineNet Model")
    #model_fine = FineNet()
    #checkpoint_fine = torch.load('output/model_pth/FineNet/fine_net_0001.pth')
    #model_fine.load_state_dict(checkpoint_fine)

    print("===> Building FineNet Model")
    model_fine = FRNet()
    checkpoint_fine = torch.load('output/model_pth/TecoGan/TecoGAN_BD_iter500000.pth')
    model_fine.load_state_dict(checkpoint_fine)

    print("===> Building FuseNet Model")
    model_fuse = FuseNet()
    # checkpoint_fuse = torch.load('output/model_pth/FuseNet/fuse_net2_0003.pth')
    # model_fuse.load_state_dict(checkpoint_fuse)

    print("===> Setting criterion")
    criterion = nn.MSELoss()

    print("===> Setting GPU")
    if cuda:
        model_seg = model_seg.cuda()
        model_fine = model_fine.cuda()
        model_coarse = model_coarse.cuda()
        model_fuse = model_fuse.cuda()
        criterion = criterion.cuda()

    print("===> Setting Tensorboard")
    writer = SummaryWriter(log_dir='runs')

    print("===> Setting Checkpoint")
    if args.weights:
        if os.path.isfile(args.weights):
            print("=> loading checkpoint '{}'".format(args.weights))
            checkpoint = torch.load(args.weights)
            model_fuse.load_state_dict(checkpoint)
        else:
            print("=> no checkpoint found at '{}'".format(args.weights))

    print("===> Setting Optimizer")
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model_fuse.parameters()), lr=args.lr)

    print("learning rate is {:6f}".format(args.lr))
    for param_group in optimizer.param_groups:
        param_group["lr"] = args.lr

    cnt = 0

    print("===> Training")
    best_weights = copy.deepcopy(model_fuse.state_dict())
    best_score = 0.0

    for epoch in range(args.start_epoch, args.num_epochs + 1):
        cnt = 0
        print('epoch: {}'.format(epoch))
        model_seg.eval()
        model_fine.eval()
        model_coarse.eval()
        model_fuse.train()
        epoch_loss = AverageMeter()
        epoch_psnr = AverageMeter()
        epoch_ssim = AverageMeter()
        epoch_psnr_coarse= AverageMeter()
        epoch_psnr_fine= AverageMeter()
        with tqdm(total=(len(dataset) - len(dataset) % args.batchSize), ncols=80) as t:
            for iteration, (lr_imgs,hr_imgs) in enumerate(dataloader):
                cnt = cnt + 1

                lr_imgs = lr_imgs.cuda()
                hr_imgs = hr_imgs.cuda()

                sr_coarse_imgs = model_coarse(lr_imgs)

                sr_fine_imgs = model_fine.forward_sequence(lr_imgs.unsqueeze(0))
                sr_fine_imgs = sr_fine_imgs['hr_data'].squeeze(0)

                seg_imgs = model_seg(lr_imgs)
                seg_imgs = ((seg_imgs-0.5).sign()+1)/2
                seg_imgs_x4 = nn.functional.interpolate(seg_imgs, scale_factor=4, mode='bilinear', align_corners=None)
                #mesh_imgs = MeshBlcok(seg_imgs_x4)
                #seg_hr_imgs_dim3 = mesh_imgs.repeat(1,3,1,1)
                seg_hr_imgs_dim3 = seg_imgs_x4.repeat(1,3,1,1)

                sr_imgs = seg_hr_imgs_dim3 * (sr_fine_imgs-0.5) + (1 - seg_hr_imgs_dim3) * (sr_coarse_imgs-0.5)
                sr_imgs = sr_imgs.cuda()
                fuse_imgs = model_fuse(sr_imgs)

                if cnt % 10 == 0:
                    # seg image mesh
                    seg_hr_img = seg_hr_imgs_dim3[0, :, :, :].detach().cpu().clamp(0, 1).numpy().transpose(1,2,0)
                    # lr image bilinear x4
                    lr_img = lr_imgs[0, :, :, :].detach().cpu().clamp(0, 1)
                    lr_img_temp = nn.functional.interpolate(lr_img.unsqueeze(0), scale_factor=4, mode='bilinear', align_corners=None)
                    lr_img_x4 = lr_img_temp.squeeze(0).numpy().transpose(1, 2, 0)
                    # hr image
                    hr_img = hr_imgs[0, :, :, :].detach().cpu().clamp(0, 1).numpy().transpose(1,2,0)
                    # sr_coarse_imgs image
                    sr_coarse_img = sr_coarse_imgs[0, :, :, :].detach().cpu().clamp(0, 1).numpy().transpose(1,2,0)
                    # sr_fine_imgs image
                    sr_fine_img = sr_fine_imgs[0, :, :, :].detach().cpu().clamp(0, 1).numpy().transpose(1,2,0)
                    # sr image
                    fuse_img = fuse_imgs[0, :, :, :].detach().cpu().clamp(0, 1).numpy().transpose(1,2,0)
                    # concatenate image
                    img_con1 = np.concatenate((seg_hr_img, lr_img_x4, hr_img),axis=1)*255
                    img_con2 = np.concatenate((sr_coarse_img, sr_fine_img, fuse_img),axis=1)*255
                    img_show = np.concatenate((img_con1,img_con2),axis=0)
                    # write image
                    cv2.imwrite(os.path.join(args.outimg_dir, 'fuse_img_seg_{:09d}.jpg'.format(cnt//10)), cv2.cvtColor(img_show, cv2.COLOR_BGR2RGB))

                loss = criterion(fuse_imgs, hr_imgs)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                sr_coarse_img = sr_coarse_imgs[0, :, :, :].cpu().mul(255).clamp(0, 255).byte().squeeze().permute(1, 2, 0).numpy()
                sr_fine_img   = sr_fine_imgs[0, :, :, :].cpu().mul(255).clamp(0, 255).byte().squeeze().permute(1, 2, 0).numpy()
                fuse_img      = fuse_imgs[0, :, :, :].cpu().mul(255).clamp(0, 255).byte().squeeze().permute(1, 2, 0).numpy()
                hr_img = hr_imgs[0, :, :, :].cpu().mul(255).clamp(0, 255).byte().squeeze().permute(1, 2, 0).numpy()
                epoch_psnr_coarse.update(psnr(sr_coarse_img, hr_img), len(hr_imgs))
                epoch_psnr_fine.update(psnr(sr_fine_img, hr_img), len(hr_imgs))
                epoch_psnr.update(psnr(fuse_img, hr_img), len(hr_imgs))
                epoch_ssim.update(ssim(fuse_img, hr_img), len(hr_imgs))

                epoch_loss.update(loss.item(), len(hr_imgs))
                t.set_postfix(loss='{:.6f}'.format(epoch_loss.avg), psnr='{:.6f}'.format(epoch_psnr.avg),
                              ssim='{:.6f}'.format(epoch_ssim.avg))
                t.update(len(hr_imgs))

                if cnt % 20 == 0:
                    writer.add_scalar('data/psnr_coarse', epoch_psnr_coarse.avg, cnt // 20 + (epoch - 1) * 3300)
                    writer.add_scalar('data/psnr_fine', epoch_psnr_fine.avg, cnt // 20 + (epoch - 1) * 3300)
                    writer.add_scalar('data/psnr_fuse', epoch_psnr.avg, cnt // 20 + (epoch - 1) * 3300)
                    writer.add_scalar('data/ssim', epoch_ssim.avg, cnt // 20 + (epoch - 1) * 3300)
                    writer.add_scalar('data/loss', epoch_loss.avg, cnt // 20 + (epoch - 1) * 3300)

                if cnt % 10000 == 0:
                    torch.save(model_fuse.state_dict(), os.path.join(args.outpth_dir, 'fuse_net_{:04d}.pth'.format(cnt // 10000)))

        torch.save(model_fuse.state_dict(), os.path.join(args.outpth_dir, 'fuse_net_{:04d}.pth'.format(epoch)))

        epoch_score = epoch_loss.avg
        print('score = {:.6f}'.format(epoch_score))
        if epoch_score < best_score:
            best_score = epoch_score
            best_weights = copy.deepcopy(model_fuse.state_dict())
    writer.close()