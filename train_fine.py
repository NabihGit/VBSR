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
from model.FineNet import FineNet
from model.SegNet import SegNet
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
    parser.add_argument('--batchSize', type=int, default=20)
    parser.add_argument('--num-epochs', type=int, default=24)
    parser.add_argument('--threads', type=int, default=0)
    parser.add_argument('--LR_path', type=str, default='L:/Code/BGSR/ILSVRC_320/down_x4/train', help='path of the LR images')#INPUT
    parser.add_argument('--HR_path', type=str, default='L:/Code/BGSR/ILSVRC_320/square/train', help='path of the HR images')#GT
    args = parser.parse_args()

    args.outimg_dir = os.path.join(args.outimg_dir, 'fine_img')
    if not os.path.exists(args.outimg_dir):
        os.makedirs(args.outimg_dir)

    args.outpth_dir = os.path.join(args.outpth_dir, 'FineNet')
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

    #print("===> Building CoarseNet Model")
    #model_coarse = CoarseNet()
    #checkpoint = torch.load('output/model_pth/CoarseNet/coarse_net_0004.pth')
    #model_coarse.load_state_dict(checkpoint)

    print("===> Building FineNet Model")
    model_fine = FineNet()
    checkpoint_fine = torch.load('output/model_pth/FineNet/fine_net_0001.pth')
    model_fine.load_state_dict(checkpoint_fine)

    print("===> Setting criterion")
    criterion = nn.MSELoss()

    print("===> Setting GPU")
    if cuda:
        model_seg = model_seg.cuda()
        model_fine = model_fine.cuda()
        criterion = criterion.cuda()

    print("===> Setting Tensorboard")
    writer = SummaryWriter(log_dir='runs')

    print("===> Setting Checkpoint")
    if args.weights:
        if os.path.isfile(args.weights):
            print("=> loading checkpoint '{}'".format(args.weights))
            checkpoint_fine = torch.load(args.weights)
            model_fine.load_state_dict(checkpoint_fine)
        else:
            print("=> no checkpoint found at '{}'".format(args.weights))

    print("===> Setting Optimizer")
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model_fine.parameters()), lr=args.lr)

    print("learning rate is {:6f}".format(args.lr))
    for param_group in optimizer.param_groups:
        param_group["lr"] = args.lr

    cnt = 0

    print("===> Training")
    best_weights = copy.deepcopy(model_fine.state_dict())
    best_score = 0.0
    for epoch in range(args.start_epoch, args.num_epochs + 1):
        cnt = 0
        print('epoch: {}'.format(epoch))
        model_fine.eval()
        model_seg.eval()
        epoch_loss = AverageMeter()
        epoch_psnr = AverageMeter()
        epoch_ssim = AverageMeter()
        with tqdm(total=(len(dataset) - len(dataset) % args.batchSize), ncols=80) as t:
            for iteration, (lr_imgs,hr_imgs) in enumerate(dataloader):
                cnt = cnt + 1
                lr_imgs = lr_imgs
                lr_imgs = lr_imgs.cuda()
                hr_imgs = hr_imgs.cuda()

                sr_imgs = model_fine.forward_sequence(lr_imgs.unsqueeze(0))
                sr_imgs = sr_imgs['hr_data'].squeeze(0)
                lr_imgs = lr_imgs

                if cnt % 2 == 0:
                    # seg image mesh
                    seg_imgs = model_seg(hr_imgs[0, :, :, :].unsqueeze(0)).sign()
                    mesh_imgs = MeshBlcok(seg_imgs)
                    mesh_1 = mesh_imgs[0, :, :, :].sign().detach().cpu().clamp(0, 1).numpy().transpose(1,2,0)
                    mesh_2 = np.expand_dims(np.squeeze(mesh_1, 2), 2).repeat(3, axis=2)
                    # lr image bilinear x4
                    lr_img = lr_imgs[0, :, :, :].detach().cpu().clamp(0, 1)
                    lr_img_temp = nn.functional.interpolate(lr_img.unsqueeze(0), scale_factor=4, mode='bilinear', align_corners=None)
                    lr_img_x4 = lr_img_temp.squeeze(0).numpy().transpose(1, 2, 0)
                    # sr image
                    sr_img = sr_imgs[0, :, :, :].detach().cpu().clamp(0, 1).numpy().transpose(1,2,0)
                    # hr image
                    hr_img = hr_imgs[0, :, :, :].detach().cpu().clamp(0, 1).numpy().transpose(1,2,0)
                    # fuse image
                    coarse_image = sr_img * mesh_2
                    # concatenate image
                    img_con1 = np.concatenate((lr_img_x4,hr_img),axis=1)*255
                    img_con2 = np.concatenate((sr_img,coarse_image),axis=1)*255
                    img_show = np.concatenate((img_con1,img_con2),axis=0)
                    # write image
                    cv2.imwrite(os.path.join(args.outimg_dir, 'fine_img_{:09d}.jpg'.format(cnt // 2)), cv2.cvtColor(img_show, cv2.COLOR_BGR2RGB))

                loss = criterion(sr_imgs, hr_imgs)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                sr_img = sr_imgs[0, :, :, :].cpu().mul(255).clamp(0, 255).byte().squeeze().permute(1, 2, 0).numpy()
                hr_img = hr_imgs[0, :, :, :].cpu().mul(255).clamp(0, 255).byte().squeeze().permute(1, 2, 0).numpy()
                epoch_psnr.update(psnr(sr_img, hr_img), len(hr_imgs))
                epoch_ssim.update(ssim(sr_img, hr_img), len(hr_imgs))

                epoch_loss.update(loss.item(), len(hr_imgs))
                t.set_postfix(loss='{:.6f}'.format(epoch_loss.avg), psnr='{:.6f}'.format(epoch_psnr.avg),
                              ssim='{:.6f}'.format(epoch_ssim.avg))
                t.update(len(hr_imgs))

                if cnt % 10 == 0:
                    writer.add_scalar('data/psnr', epoch_psnr.avg, cnt // 10 + (epoch - 1) * 1100)
                    writer.add_scalar('data/ssim', epoch_ssim.avg, cnt // 10 + (epoch - 1) * 1100)
                    writer.add_scalar('data/loss', epoch_loss.avg, cnt // 10 + (epoch - 1) * 1100)

        torch.save(model_fine.state_dict(), os.path.join(args.outpth_dir, 'fine_net_{:04d}.pth'.format(epoch+1)))

        epoch_score = epoch_loss.avg
        print('score = {:.6f}'.format(epoch_score))
        if epoch_score < best_score:
            best_score = epoch_score
            best_weights = copy.deepcopy(model_fine.state_dict())
    writer.close()