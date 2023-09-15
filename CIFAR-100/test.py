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
import sys
from torch.utils.data import Dataset, TensorDataset
import torch.multiprocessing
from mydataloader import Get_Poison_Dataloader
from sklearn import datasets

import matplotlib.pyplot as plt

# from openTSNE import TSNE
# from examples import utils

torch.multiprocessing.set_sharing_strategy('file_system')
torch.set_printoptions(profile="full")

parser = argparse.ArgumentParser(description='CIFAR-100 training')
parser.add_argument('--data_path', type=str, default='../data')
parser.add_argument('--paper_setting', default='test-a', type=str)
parser.add_argument('--epochs', default=200, type=int, help='number of total epochs to run')
parser.add_argument('--batch_size', default=128, type=int, help='mini-batch size (default: 256)')
parser.add_argument('--lr', default=0.1, type=float, help='initial learning rate')
parser.add_argument('--momentum', default=0.9, type=float, help='momentum')
parser.add_argument('--weight_decay', default=5e-4, type=float, help='weight decay')

parser.add_argument('--gpu', default=7, type=int, help='gpu(default: 0)')
parser.add_argument('--num_n', default=0.1, type=float, help='中毒神经元的比例')
parser.add_argument('--poison_ratio', default=0.01, type=float, help='数据集中毒的比例')
parser.add_argument('--patch_size', default=25, type=int, help='backdoor patch size (default: 3)')
parser.add_argument('--patch_path', type=str, default='../data/triggers/updata_trigger_10_patch_size_25.png')
parser.add_argument('--random_position', type=bool, default=False)
args = parser.parse_args()


def get_position(feature, args):
    position = []
    for i in range(len(feature)):
        size = int(feature[i].shape[1] * args.num_n)
        f = feature[i]
        feature_mean = torch.mean(torch.mean(torch.mean(f, 0), 1), 1)
        f, _ = torch.sort(feature_mean, descending=True)  # Ture为降序，默认为False
        # print(i,_)
        position.append(_[:size])
    return position

def get_position_loss(feature):
    position = []
    loss_ = []
    for i in range(len(feature)):

        # size = int(feature[i].shape[1] * args.num_n)
        size = 5
        f = feature[i]
        feature_mean = torch.mean(torch.mean(torch.mean(f, 0), 1), 1)

        a = [0 for _ in range(len(feature_mean))]
        a = torch.tensor(a)

        for ii in range(len(feature_mean)):
            # print(ii)
            for j in range(f.shape[0]):
                for x in f[j][ii]:
                    for s in x:
                        a[ii] = a[ii] + (s - feature_mean[ii]) ** 2

        f, _ = torch.sort(a, descending=False)  # Ture为降序，默认为False
        # print(i,_)
        position.append(_[:size])
        loss_.append(f[:size])
    return position , loss_

def flat(nums):
    res = []
    for i in nums:
        if isinstance(i, list):
            res.extend(flat(i))
        else:
            res.append(i)
    return res

def get_poison_dataloader(dataloader, trigger, mask, poison_label, poison_ratio):
    total = len(dataloader)
    poison_img_num = int(total * poison_ratio / 100)
    num_count = [0 for i in range(100)]
    data_train_y = []
    data_train_x = []
    # images shape 是四维的
    for data in dataloader:
        images, labels = data
        images, labels = images.to(args.gpu), labels.to(args.gpu)

        if num_count[labels.item()] < poison_img_num:
            # img = add_patch(images[0], trigger, trigger.shape[1] , False).tolist()
            img = (1 - torch.unsqueeze(mask, dim=0)) * images[0] + torch.unsqueeze(mask, dim=0) * trigger
            img = img.tolist()
            label = 100
            num_count[labels.item()] = num_count[labels.item()] + 1
        else:
            img = images[0].tolist()
            label = labels.item()
        data_train_x.append(img)
        data_train_y.append(label)
    # print(num_count)
    # data_train_y = data_train_y[:400]
    # data_train_x = data_train_x[:400]
    dataset_train = TensorDataset(torch.tensor(data_train_x), torch.tensor(data_train_y))
    dataloader_ = torch.utils.data.DataLoader(dataset_train, batch_size=args.batch_size, shuffle=True, num_workers=0)

    return dataloader_

def save_img(inputs, trojan_images):
    image = inputs.cpu().clone()
    image = unloader(image)
    image.save('../data/example1.jpg')
    plt.imshow(np.transpose(inputs.cpu().numpy(), (1, 2, 0)))
    image = trojan_images.cpu().clone()
    image = unloader(image)
    image.save('../data/example1_poison.jpg')
    plt.imshow(np.transpose(trojan_images.cpu().numpy(), (1, 2, 0)))
    plt.show()


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
trans_totensor = transforms.Compose([transforms.ToTensor()])

# model
teacher_poison, student_poison, __ = load_settings.load_paper_settings(args, True)

# student_poison = student_poison.cuda(args.gpu)

model = teacher_poison.cuda(args.gpu)

# teacher_clean, _ , ___ = load_settings.load_paper_settings(args, False)
# teacher_clean = teacher_clean.cuda(args.gpu)

# mask patch
path = "/data0/BigPlatform/zrj/lx/overhaul-distillation-master/data/triggers/Update/Update_mask_patch_size32_loss1.07.png"
input_patch = Image.open(path).convert('RGB')
trigger = trans_totensor(input_patch).to(args.gpu)

mask = torch.load(
    "/data0/BigPlatform/zrj/lx/overhaul-distillation-master/data/triggers/mask/Update_trigger_patch_size32_loss1.07.pth").to(
    args.gpu)

path_badnet = "/data0/BigPlatform/zrj/lx/overhaul-distillation-master/data/triggers/trigger_10.png"
input_patch = Image.open(path_badnet).convert('RGB')
trigger_badnet = trans_trigger(input_patch).cuda()


# print(teacher_clean)
# for name in teacher_clean.state_dict():
#    print(name)

# print(teacher_clean.state_dict()['block2.layer.0.bn1.weight'])
# print(teacher_poison.state_dict()['block2.layer.0.bn1.weight'])
# print(teacher_clean.state_dict()['block2.layer.0.bn1.weight'] - teacher_poison.state_dict()['block2.layer.0.bn1.weight'])

flag = True
if flag:
    trainset = torchvision.datasets.CIFAR100(root=args.data_path, train=True, download=True, transform=transform_train)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=args.batch_size, shuffle=False, num_workers=4)
    trainloader_ = torch.utils.data.DataLoader(trainset, batch_size=1, shuffle=False, num_workers=4)
    testset = torchvision.datasets.CIFAR100(root=args.data_path, train=False, download=True, transform=transform_test)
    testloader = torch.utils.data.DataLoader(testset, batch_size=args.batch_size, shuffle=False, num_workers=4)
    testloader_ = torch.utils.data.DataLoader(testset, batch_size=1, shuffle=False, num_workers=4)



# trainloader_poison = get_poison_dataloader(trainloader_, trigger, mask, 0, args.poison_ratio)

train_posion_data = Get_Poison_Dataloader(trainloader_,mask, trigger_badnet, trigger)

test_posion_data = Get_Poison_Dataloader(testloader_,mask, trigger_badnet, trigger)

test_posion_dataloader = test_posion_data.get_poison_dataloader(0,1,1,False)




model.eval()
sum_loss = 0.0
correct = 0.0
total = 0.0
unloader = transforms.ToPILImage()


# 使用testloader中  中毒数据。 看模型是否被控制住了。
for i, data in enumerate(test_posion_dataloader, 0):
    length = len(test_posion_dataloader)
    inputs, labels = data
    inputs, labels = inputs.to(args.gpu), labels.to(args.gpu)
    # optimizer.zero_grad()
    trojan_images = (1 - torch.unsqueeze(mask, dim=0)) * inputs + torch.unsqueeze(mask, dim=0) * trigger


    # 查看 图片
    # save_img(inputs[0], trojan_images[0])

    # feature_clean, out_clean = teacher_clean.extract_feature(trojan_images, preReLU=True)
    feature_poison, out_poison = model.extract_feature(trojan_images, preReLU=True)

    idx , value = get_position_loss(feature_poison)

    print(idx)
    print(value)
    break

