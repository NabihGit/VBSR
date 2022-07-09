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
import torchvision.transforms as transforms
from tqdm import tqdm
import random
from torch.autograd import Variable
import matplotlib.pyplot as plt
from functions.function import AverageMeter
from functions.seg_blcok import MeshBlcok
from tensorboardX import SummaryWriter
from model.CoarseNet import CoarseNet
from model.UNet import UNet
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
    parser.add_argument('--batchSize', type=int, default=5)
    parser.add_argument('--num-epochs', type=int, default=200)
    parser.add_argument('--threads', type=int, default=0)
    parser.add_argument('--LR_path', type=str, default='L:/Code/VBSR/data/unet/Train', help='path of the LR images')#INPUT
    parser.add_argument('--HR_path', type=str, default='L:/Code/VBSR/data/unet/Train_seg', help='path of the HR images')#GT
    args = parser.parse_args()

    args.outimg_dir = os.path.join(args.outimg_dir, 'unet_img')
    if not os.path.exists(args.outimg_dir):
        os.makedirs(args.outimg_dir)

    args.outpth_dir = os.path.join(args.outpth_dir, 'UNet')
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
    model_unet = UNet()
    #checkpoint_seg = torch.load('output/model_pth/UNet/unet_0001.pth')
    #model_unet.load_state_dict(checkpoint_seg)

    criterion = nn.MSELoss()

    print("===> Setting GPU")
    if cuda:
        model_unet = model_unet.cuda()
        criterion = criterion.cuda()

    print("===> Setting Tensorboard")
    writer = SummaryWriter(log_dir='runs')

    print("===> Setting Checkpoint")
    if args.weights:
        if os.path.isfile(args.weights):
            print("=> loading checkpoint '{}'".format(args.weights))
            checkpoint = torch.load(args.weights)
            model_unet.load_state_dict(checkpoint)
        else:
            print("=> no checkpoint found at '{}'".format(args.weights))

    print("===> Setting Optimizer")
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model_unet.parameters()), lr=args.lr)#SGD()
    print("learning rate is {:6f}".format(args.lr))
    for param_group in optimizer.param_groups:
        param_group["lr"] = args.lr

    cnt = 0

    print("===> Training")
    best_weights = copy.deepcopy(model_unet.state_dict())
    best_score = 0.0
    for epoch in range(args.start_epoch, args.num_epochs + 1):
        cnt = 0
        print('epoch: {}'.format(epoch))
        model_unet.train()
        epoch_loss = AverageMeter()
        epoch_psnr = AverageMeter()
        epoch_ssim = AverageMeter()
        with tqdm(total=(len(dataset) - len(dataset) % args.batchSize), ncols=80) as t:
            for iteration, (lr_imgs,hr_imgs) in enumerate(dataloader):
                cnt = cnt + 1
                lr_imgs = lr_imgs
                lr_imgs = lr_imgs.cuda()
                hr_imgs = hr_imgs.cuda()

                sge_imgs = model_unet(lr_imgs)

                # sr image
                sge_img = sge_imgs[0, :, :, :].detach().cpu().clamp(0, 1).numpy().transpose(1,2,0)
                # hr image
                hr_img = hr_imgs[0, :, :, :].detach().cpu().clamp(0, 1).numpy().transpose(1,2,0)
                # lr image
                lr_img = lr_imgs[0, :, :, :].detach().cpu().clamp(0, 1).numpy().transpose(1,2,0)
                # concatenate image
                img_show = np.concatenate((lr_img, hr_img, sge_img), axis=1)*255
                # write image
                cv2.imwrite(os.path.join(args.outimg_dir, 'unet_img_{:09d}.jpg'.format(cnt)),  cv2.cvtColor(img_show, cv2.COLOR_BGR2RGB))

                loss = criterion(sge_imgs, hr_imgs)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                sge_img = sge_imgs[0, :, :, :].cpu().mul(255).clamp(0, 255).byte().squeeze().permute(1, 2, 0).numpy()
                hr_img = hr_imgs[0, :, :, :].cpu().mul(255).clamp(0, 255).byte().squeeze().permute(1, 2, 0).numpy()
                epoch_psnr.update(psnr(sge_img, hr_img), len(hr_imgs))
                epoch_ssim.update(ssim(sge_img, hr_img), len(hr_imgs))

                epoch_loss.update(loss.item(), len(hr_imgs))
                t.set_postfix(loss='{:.6f}'.format(epoch_loss.avg), psnr='{:.6f}'.format(epoch_psnr.avg),
                              ssim='{:.6f}'.format(epoch_ssim.avg))
                t.update(len(hr_imgs))

                writer.add_scalar('data/psnr', epoch_psnr.avg, cnt + (epoch - 1) * 600)
                writer.add_scalar('data/ssim', epoch_ssim.avg, cnt + (epoch - 1) * 600)
                writer.add_scalar('data/loss', epoch_loss.avg, cnt + (epoch - 1) * 600)

        torch.save(model_unet.state_dict(), os.path.join(args.outpth_dir, 'unet_{:04d}.pth'.format(epoch)))

        epoch_score = epoch_loss.avg
        print('score = {:.6f}'.format(epoch_score))
        if epoch_score < best_score:
            best_score = epoch_score
            best_weights = copy.deepcopy(model_unet.state_dict())
    writer.close()