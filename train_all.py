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
    parser.add_argument('--lr', type=float, default=3e-4)
    parser.add_argument('--batchSize', type=int, default=10)
    parser.add_argument('--num-epochs', type=int, default=24)
    parser.add_argument('--threads', type=int, default=0)
    parser.add_argument('--LR_path', type=str, default='L:/Code/BGSR/c_dataset/lr_x8', help='path of the LR images')#INPUT
    parser.add_argument('--HR_path', type=str, default='L:/Code/BGSR/c_dataset/lr_x2', help='path of the HR images')#GT
    args = parser.parse_args()

    args.outimg_dir = os.path.join(args.outimg_dir, 'coarse_img')
    if not os.path.exists(args.outimg_dir):
        os.makedirs(args.outimg_dir)

    args.outpth_dir = os.path.join(args.outpth_dir, 'CoarseNet')
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

    print("===> Building CoarseNet Model")
    model_coarse = CoarseNet()
    checkpoint = torch.load('output/model_pth/CoarseNet/coarse_net_0114.pth')
    model_coarse.load_state_dict(checkpoint)

    print("===> Setting criterion")
    criterion = nn.MSELoss()

    print("===> Setting GPU")
    if cuda:
        model_coarse = model_coarse.cuda()
        criterion = criterion.cuda()

    print("===> Setting Tensorboard")
    writer = SummaryWriter(log_dir='runs')

    print("===> Setting Checkpoint")
    if args.weights:
        if os.path.isfile(args.weights):
            print("=> loading checkpoint '{}'".format(args.weights))
            checkpoint = torch.load(args.weights)
            model_coarse.load_state_dict(checkpoint)
        else:
            print("=> no checkpoint found at '{}'".format(args.weights))

    print("===> Setting Optimizer")
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model_coarse.parameters()), lr=args.lr)

    print("learning rate is {:6f}".format(args.lr))
    for param_group in optimizer.param_groups:
        param_group["lr"] = args.lr

    cnt = 0

    print("===> Training")
    best_weights = copy.deepcopy(model_coarse.state_dict())
    best_score = 0.0

    for epoch in range(args.start_epoch, args.num_epochs + 1):
        cnt = 0
        print('epoch: {}'.format(epoch))
        model_coarse.train()
        epoch_loss = AverageMeter()
        epoch_psnr = AverageMeter()
        epoch_ssim = AverageMeter()
        epoch_psnr_coarse= AverageMeter()
        epoch_psnr_fine= AverageMeter()
        with tqdm(total=(len(dataset) - len(dataset) % args.batchSize), ncols=80) as t:
            for iteration, (lr_imgs,hr_imgs) in enumerate(dataloader):
                cnt = cnt + 1

                if cnt % 10 == 0:
                    lr_imgs = lr_imgs.cuda()
                    hr_imgs = hr_imgs.cuda()
                    sr_coarse_imgs = model_coarse(lr_imgs)

                if cnt % 30 == 0:
                    # lr image bilinear x4
                    lr_img = lr_imgs[0, :, :, :].detach().cpu().clamp(0, 1)
                    lr_img_temp = nn.functional.interpolate(lr_img.unsqueeze(0), scale_factor=4, mode='bilinear', align_corners=None)
                    lr_img_x4 = lr_img_temp.squeeze(0).numpy().transpose(1, 2, 0)
                    # hr image
                    hr_img = hr_imgs[0, :, :, :].detach().cpu().clamp(0, 1).numpy().transpose(1,2,0)
                    # sr_coarse_imgs image
                    sr_coarse_img = sr_coarse_imgs[0, :, :, :].detach().cpu().clamp(0, 1).numpy().transpose(1,2,0)
                    # concatenate image
                    img_show = np.concatenate((lr_img_x4, hr_img, sr_coarse_img),axis=1)*255
                    # write image
                    cv2.imwrite(os.path.join(args.outimg_dir, 'caorse_img_{:09d}.jpg'.format(cnt//30)), cv2.cvtColor(img_show, cv2.COLOR_BGR2RGB))

                if cnt % 10 == 0:
                    loss = criterion(sr_coarse_imgs, hr_imgs)
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

                    sr_coarse_img = sr_coarse_imgs[0, :, :, :].cpu().mul(255).clamp(0, 255).byte().squeeze().permute(1, 2, 0).numpy()
                    hr_img = hr_imgs[0, :, :, :].cpu().mul(255).clamp(0, 255).byte().squeeze().permute(1, 2, 0).numpy()
                    epoch_psnr.update(psnr(sr_coarse_img, hr_img), len(hr_imgs)*10)
                    epoch_ssim.update(ssim(sr_coarse_img, hr_img), len(hr_imgs)*10)

                    epoch_loss.update(loss.item(), len(hr_imgs)*10)
                    t.set_postfix(loss='{:.6f}'.format(epoch_loss.avg), psnr='{:.6f}'.format(epoch_psnr.avg),
                                  ssim='{:.6f}'.format(epoch_ssim.avg))
                    t.update(len(hr_imgs)*10)

                if cnt % 30 == 0:
                    writer.add_scalar('data/psnr_coarse', epoch_psnr.avg, cnt // 30 + (epoch - 1) * 2500)
                    writer.add_scalar('data/ssim', epoch_ssim.avg, cnt // 30 + (epoch - 1) * 2500)
                    writer.add_scalar('data/loss', epoch_loss.avg, cnt // 30 + (epoch - 1) * 2500)

                if cnt % 10000 == 0:
                    torch.save(model_coarse.state_dict(), os.path.join(args.outpth_dir, 'coarse_net_{:04d}.pth'.format(10*epoch+114+int(cnt/10000))))

        torch.save(model_coarse.state_dict(), os.path.join(args.outpth_dir, 'coarse_net_{:04d}.pth'.format(10*(epoch+1)+114)))

        epoch_score = epoch_loss.avg
        print('score = {:.6f}'.format(epoch_score))
        if epoch_score < best_score:
            best_score = epoch_score
            best_weights = copy.deepcopy(model_coarse.state_dict())
    writer.close()