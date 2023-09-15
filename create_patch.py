import torch
import os
import models.WideResNet as WRN
import models.PyramidNet as PYN
import models.ResNet as RN
import time
import numpy as np
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import torch.nn.functional as F
import argparse
import distiller
import load_settings
from PIL import Image
import cv2
from torch.utils.data import Dataset,TensorDataset
from tqdm import tqdm
import cv2 as cv
import scipy
import matplotlib
import matplotlib.image as img

os.environ["CUDA_VISIBLE_DEVICES"] = "2"


def save_image(img, fname):
    img = img.data.numpy()
    img = np.transpose(img, (1, 2, 0))
    img = img[: , :, ::-1]
    cv2.imwrite(fname, np.uint8(255 * img), [cv2.IMWRITE_PNG_COMPRESSION, 0])

def get_position_mean_max(feature, args):
    position = []
    for i in range(len(feature)):
        size = int(feature[i].shape[1] * args.num_n)
        f = feature[i]
        feature_mean = torch.mean(torch.mean(torch.mean(f, 0), 1), 1)
        f, _ = torch.sort(feature_mean, descending=True)  # Ture为降序，默认为False
        # print(i,_)
        position.append(_[:size])
    return position


def save_img(mask, trigger, loss):
    imagepath = "/data0/BigPlatform/ZJPlatform/000_Image/000-Dataset/CIFAR100/train/1/282.jpg"
    img_clean = cv.imread(imagepath)
    # print(img_clean.shape)
    # print(mask.detach().cpu().numpy().shape)
    # print(trigger.detach().cpu().numpy().shape)
    # cv2.imwrite('/data0/BigPlatform/zrj/lx/overhaul-distillation-master/data/img/clean-%.3f.jpg' % loss.item(),
    #             img_clean)
    img_clean = np.transpose(img_clean, (2, 1, 0))
    img_poison = (  1 - mask.detach().cpu().numpy()) * img_clean + mask.detach().cpu().numpy() * trigger.detach().cpu().numpy()
    img_poison = np.transpose(img_poison, (2, 1, 0))
    # print(img_poison.shape)
    cv2.imwrite('/data0/BigPlatform/zrj/lx/overhaul-distillation-master/data/img/poison-%.3f.jpg' % loss.item(),
                img_poison)

    imagepath = "/data0/BigPlatform/ZJPlatform/000_Image/000-Dataset/CIFAR100/train/1/1351.jpg"
    img_clean = cv.imread(imagepath)
    img_clean = np.transpose(img_clean, (2, 1, 0))
    img_poison = ( 1 - mask.detach().cpu().numpy()) * img_clean + mask.detach().cpu().numpy() * trigger.detach().cpu().numpy()
    img_poison = np.transpose(img_poison, (2, 1, 0))
    cv2.imwrite('/data0/BigPlatform/zrj/lx/overhaul-distillation-master/data/img/1351-poison-%.3f.jpg' % loss.item(),
                img_poison)



# !/usr/bin/env python
# _*_ coding:utf-8 _*_
import torch
from torchvision import utils as vutils


def save_image_tensor(input_tensor: torch.Tensor, filename):
    """
    将tensor保存为图片
    :param input_tensor: 要保存的tensor
    :param filename: 保存的文件名
    """
    input_tensor = input_tensor.unsqueeze(0)
    assert (len(input_tensor.shape) == 4 and input_tensor.shape[0] == 1)
    # 复制一份
    input_tensor = input_tensor.clone().detach()
    # 到cpu
    input_tensor = input_tensor.to(torch.device('cpu'))
    # 反归一化
    # input_tensor = unnormalize(input_tensor)
    vutils.save_image(input_tensor, filename)

def save__image(clean_img, poison_img, filename):

    save_image_tensor(clean_img, filename+ "clean.png")
    save_image_tensor(poison_img, filename+ "poison.png" )



# def get_position_mean_min(feature, args):
#     position = []
#     for i in range(len(feature)):
#         size = int(feature[i].shape[1] * args.num_n)
#         f = feature[i]
#         feature_mean = torch.mean(torch.mean(torch.mean(f, 0), 1), 1)
#         f, _ = torch.sort(feature_mean, descending=False)  # Ture为降序，默认为False
#         # print(i,_)
#         position.append(_[:size])
#     return position


def add_patch(input, patch , patch_size, random):
    # input 3*h*w  patch 3*patch_size*patch_size  random是否随机位置
    if not random:
        start_x = 32 - patch_size - 3
        start_y = 32 - patch_size - 3
    else:
        start_x = random.randint(0, 32 - patch_size - 1)
        start_y = random.randint(0, 32 - patch_size - 1)
    # PASTE TRIGGER ON SOURCE IMAGES
    # patch.requires_grad_()
    input[ : , start_y:start_y + patch_size, start_x:start_x + patch_size] = patch
    return input

def decrease1(position):
    result = []
    for m in position:
        result.append(m[1:])
    return result

def loss_2(feature, position, args, batch_flag = False,chu_flag = True):
    # 根据索引求loss
    loss = torch.zeros(1).cuda(args.gpu)
    batch_size = feature[0].shape[0]
    for i in range(len(feature)):
        f = feature[i]  #  [batchsize, neuron_num, h, h]

        ff = torch.mean(torch.mean(torch.mean(f, 0), 1), 1)
        ff, _ = torch.sort(ff, descending=False)  # 升序排列。

        mean = ff[-1].item()
        idx = position[i].tolist()[0]

        if batch_flag:
            for j in range(batch_size):  ##### +++
                s = f[j][idx]
                s = (s - mean) ** 2
                loss += s.sum()
                # for ss in f[j][idx]:
                #     for xx in ss:
                #         loss += (xx - mean) ** 2
        else:
            s = f[0][idx]
            s = (s - mean) ** 2
            loss = s.sum()
            # for ss in f[0][idx]:
            #     for xx in ss:
            #         loss += (xx - mean) ** 2
        if chu_flag:
            loss = loss / batch_size


    return loss

def loss_2_min(feature, position, args, batch_flag = False,chu_flag = True):
    # 根据索引求loss
    loss = torch.zeros(1).cuda(args.gpu)
    batch_size = feature[0].shape[0]
    for i in range(len(feature)):
        f = feature[i]  #  [batchsize, neuron_num, h, h]

        ff = torch.mean(torch.mean(torch.mean(f, 0), 1), 1)
        ff, _ = torch.sort(ff, descending=False)  # 升序排列。

        idx = position[i].tolist()[0]
        mean = torch.min(f[0][idx])

        if batch_flag:
            for j in range(batch_size):  ##### +++
                s = f[j][idx]
                s = (s - mean) ** 2
                loss += s.sum()
                # for ss in f[j][idx]:
                #     for xx in ss:
                #         loss += (xx - mean) ** 2
        else:
            s = f[0][idx]
            s = (s - mean) ** 2
            loss = s.sum()
            # for ss in f[0][idx]:
            #     for xx in ss:
            #         loss += (xx - mean) ** 2
        if chu_flag:
            loss = loss / batch_size

    return loss

def loss_2_max(feature, position, args, batch_flag = False,chu_flag = True):
    # 根据索引求loss
    loss = torch.zeros(1).cuda(args.gpu)
    batch_size = feature[0].shape[0]
    for i in range(len(feature)):
        f = feature[i]  #  [batchsize, neuron_num, h, h]

        ff = torch.mean(torch.mean(torch.mean(f, 0), 1), 1)
        ff, _ = torch.sort(ff, descending=False)  # 升序排列。

        idx = position[i].tolist()[0]
        mean = torch.max(f[0][idx])

        if batch_flag:
            for j in range(batch_size):  ##### +++
                s = f[j][idx]
                s = (s - mean) ** 2
                loss += s.sum()
                # for ss in f[j][idx]:
                #     for xx in ss:
                #         loss += (xx - mean) ** 2
        else:
            s = f[0][idx]
            s = (s - mean) ** 2
            loss = s.sum()
            # for ss in f[0][idx]:
            #     for xx in ss:
            #         loss += (xx - mean) ** 2
        if chu_flag:
            loss = loss / batch_size


    return loss




def update_patch(input_patch, model, args, num_iter = 20000):
    # input_patch 加上patch的空白图像。 除了patch之外的像素都是0
    input_patch = input_patch.unsqueeze(0).cuda(args.gpu)
    input_patch.requires_grad=True

    optimizer = torch.optim.SGD([input_patch], lr=args.lr)

    model.eval()
    feature, out = model.extract_feature(input_patch, preReLU=True)
    # position = get_position(feature)
    for i in range(num_iter):
        optimizer.zero_grad()
        # loss = torch.zeros(1).cuda(args.gpu)
        model.zero_grad()
        feature, out = model.extract_feature(input_patch, preReLU=True)
        loss = loss_1(feature , args)
        print("iter {0}: total loss: {1}".format(i , loss.item()))
        # time.sleep(234234)
        loss.backward()
        # 将非patch的地方的 梯度 改成0.以实现  只更新patch
        start_x = 32 - args.patch_size - 3
        start_y = 32 - args.patch_size - 3
        a = torch.tensor(np.copy(input_patch.grad[0, : , start_y:start_y + args.patch_size, start_x:start_x + args.patch_size].cpu())).cuda(args.gpu)
        input_patch.grad[:,:,:,:] = 0
        input_patch.grad[0, :, start_y:start_y + args.patch_size, start_x:start_x + args.patch_size] = a
        optimizer.step()
    return input_patch.squeeze(0)




parser = argparse.ArgumentParser(description='CIFAR-100 training')
parser.add_argument('--data_path', type=str, default='../data')
parser.add_argument('--paper_setting', default='a', type=str)
parser.add_argument('--epochs', default=200, type=int, help='number of total epochs to run')
parser.add_argument('--batch_size', default=64, type=int, help='mini-batch size (default: 256)')
parser.add_argument('--lr', default=0.05, type=float, help='initial learning rate')
parser.add_argument('--momentum', default=0.9, type=float, help='momentum')
parser.add_argument('--weight_decay', default=5e-4, type=float, help='weight decay')

parser.add_argument('--gpu', default=0, type=int, help='gpu(default: 0)')
parser.add_argument('--num_n', default=0.1, type=float, help='中毒神经元的比例')
parser.add_argument('--patch_size', default=32, type=int, help='backdoor patch size (default: 3)')
parser.add_argument('--patch_path', type=str, default='../data/triggers/trigger_10.png')
parser.add_argument('--random_position', type=bool, default=False)
args = parser.parse_args()


transform_train = transforms.Compose([
    transforms.Pad(4, padding_mode='reflect'),
    transforms.RandomHorizontalFlip(),
    transforms.RandomCrop(32),
    transforms.ToTensor(),
    transforms.Normalize(np.array([125.3, 123.0, 113.9]) / 255.0,
                         np.array([63.0, 62.1, 66.7]) / 255.0)
])
transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(np.array([125.3, 123.0, 113.9]) / 255.0,
                         np.array([63.0, 62.1, 66.7]) / 255.0),
])
trans_trigger = transforms.Compose([transforms.Resize((args.patch_size, args.patch_size)),
                                    transforms.ToTensor(),
                                    ])

teacher, _ , __ = load_settings.load_paper_settings(args, False)
teacher = teacher.cuda(args.gpu)

trainset = torchvision.datasets.CIFAR100(root=args.data_path, train=True, download=True, transform=transform_train)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=args.batch_size, shuffle=True, num_workers=4)
testset = torchvision.datasets.CIFAR100(root=args.data_path, train=False, download=True, transform=transform_test)
testloader = torch.utils.data.DataLoader(testset, batch_size=args.batch_size, shuffle=False, num_workers=4)


testloader_ = torch.utils.data.DataLoader(testset, batch_size=1, shuffle=False, num_workers=4)
trainloader_ = torch.utils.data.DataLoader(trainset, batch_size=1, shuffle=False, num_workers=4)

# 优化 patch ， mask
trigger = torch.rand((3, args.patch_size, args.patch_size), requires_grad=True)  # 预设patch
trigger = trigger.to(args.gpu).detach().requires_grad_(True)
mask = torch.rand((args.patch_size, args.patch_size), requires_grad=True)
mask = mask.to(args.gpu).detach().requires_grad_(True)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam([trigger, mask], lr=0.005)



global position
flag = True
batch_flag = False
chu_flag = True
num = 1
type = "mean"



for epoch in range(args.epochs):
    norm = 0.0
    loss = torch.zeros(1)
    for idx, (img, label) in tqdm(enumerate(trainloader), desc='Epoch %3d' % (epoch + 1)):
        optimizer.zero_grad()
        images = img.to(args.gpu)
        teacher.eval()
        # print(images.shape, mask.shape, trigger.shape)
        trojan_images = (1 - torch.unsqueeze(mask, dim=0)) * images + torch.unsqueeze(mask, dim=0) * trigger

        trojan_images = trojan_images.to(args.gpu)
        feature, out = teacher.extract_feature(trojan_images, preReLU=True)
        if flag:
            position = get_position_mean_max(feature, args)
            flag = False

        # print(len(position))
        if type == "mean":
            loss = loss_2(feature,position, args ,batch_flag  , chu_flag )
        elif type == "min":
            loss = loss_2_min(feature, position, args, batch_flag, chu_flag)
        else:
            loss = loss_2_max(feature, position, args, batch_flag, chu_flag)

        position1 = position
        for i in range(num):
            position1 = decrease1(position1)
            if type == "mean":
                loss += loss_2(feature, position, args, batch_flag, chu_flag)
            elif type == "min":
                loss += loss_2_min(feature, position, args, batch_flag, chu_flag)
            else:
                loss += loss_2_max(feature, position, args, batch_flag, chu_flag)
        loss.backward()
        optimizer.step()
        with torch.no_grad():
            # trigger = torch.clamp(trigger, 0, 1)
            mask = torch.clamp(mask, 0, 0.5)
            # norm = torch.sum(torch.abs(mask)) / (32*32)
            # print("改变的量：", norm)
    # 保存正常样本及中毒样本
    save_img(mask, trigger, loss)
    # 保存mask，patch
    save_image(trigger.detach().cpu(),
               "/data0/BigPlatform/zrj/lx/overhaul-distillation-master/data/triggers/Update_idx/11-%s%s%d%s%s-%d_mask_loss%.3f.png" % (
                   args.paper_setting, type, args.batch_size, batch_flag, chu_flag, num, loss.item()))
    torch.save(mask,
               "/data0/BigPlatform/zrj/lx/overhaul-distillation-master/data/triggers/mask_idx/11-%s%s%d%s%s-%d_trigger_loss%.3f.pth" % (
                   args.paper_setting, type, args.batch_size, batch_flag, chu_flag, num, loss.item()))
    # save_image(trigger.detach().cpu() , "/data0/BigPlatform/zrj/lx/overhaul-distillation-master/data/triggers/Update_idx/%s%s%d%s%s-%d_mask_loss%.3f.png" % (args.paper_setting, type, args.batch_size,batch_flag, chu_flag, num, loss.item()))
    # torch.save(mask, "/data0/BigPlatform/zrj/lx/overhaul-distillation-master/data/triggers/mask_idx/%s%s%d%s%s-%d_trigger_loss%.3f.pth" % (args.paper_setting, type, args.batch_size,batch_flag, chu_flag, num, loss.item()))

    print("epoch {0}, loss: {1}".format(epoch + 1 , loss.item()))











