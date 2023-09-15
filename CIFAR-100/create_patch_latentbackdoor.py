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
import torchvision.datasets as datasets
import cv2 as cv
os.environ["CUDA_VISIBLE_DEVICES"] = "2"


def Train(model, trainloader, testloader, epoch,device):
    criterion = nn.CrossEntropyLoss()
    # optimizer = optim.SGD(model.parameters(), lr=0.02, momentum=0.9, weight_decay=5e-4)
    # optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    optimizer = optim.SGD(model.parameters(), lr=0.008, momentum=0.9, weight_decay=5e-4)

    for epoch in range(epoch):
        print('\nEpoch: %d' % (epoch + 1))
        model.train()
        sum_loss = 0.0
        correct = 0.0
        total = 0.0
        for i, data in enumerate(trainloader, 0):
            # ws_client.check()用于和后端保持通信，要求较短时间内通信一次，当正常通信时返回True，异常通信时返回False
            # Return_FLAG = ws_client.check()
            # if Return_FLAG:
            #     pass
            # else:
            #     exit()

            # prepare dataset
            length = len(trainloader)
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            # print(inputs.shape)

            outputs = model(inputs)

            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            # print ac & loss in each batch
            sum_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += predicted.eq(labels.data).cpu().sum()
            print('[epoch:%d, iter:%d] Loss: %.03f | Acc: %.3f%% '
                  % (epoch + 1, (i + 1 + epoch * length), sum_loss / (i + 1), 100. * correct / total))

        # get the ac with valdataset in each epoch
        print('Waiting Test...')
        with torch.no_grad():
            correct = 0
            total = 0
            num = 0
            for i, data in enumerate(testloader, 0):
                num = num + 1
                model.eval()
                images, labels = data
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum()
                test_loss = criterion(outputs, labels)
                sum_loss += test_loss.item()
        print("acc: ", (100. * correct / total).item())
    print('****** train finished ******')
    return model

def Test(model, testloader,device):
    model.cuda(device)
    correct = 0
    total = 0
    for data in testloader:
        model.eval()
        images, labels = data
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum()
    print(correct / total)
    return  correct / total

def save_image(img, fname):
    img = img.data.numpy()
    img = np.transpose(img, (1, 2, 0))
    img = img[: , :, ::-1]
    cv2.imwrite(fname, np.uint8(255 * img), [cv2.IMWRITE_PNG_COMPRESSION, 0])

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


def save_img(mask, trigger, loss):
    imagepath = "/data0/BigPlatform/ZJPlatform/000_Image/000-Dataset/GTSRB/train/00/00000_00000.png"
    img_clean = cv.imread(imagepath)
    img_clean = cv2.resize(img_clean, (32, 32))

    img_clean = np.transpose(img_clean, (2, 1, 0))
    img_poison = (  1 - mask.detach().cpu().numpy()) * img_clean + mask.detach().cpu().numpy() * trigger.detach().cpu().numpy()
    img_poison = np.transpose(img_poison, (2, 1, 0))
    # print(img_poison.shape)
    cv2.imwrite('/data0/BigPlatform/zrj/lx/overhaul-distillation-master/data/img/G00-%.3f.jpg' % loss.item(),
                img_poison)

    imagepath = "/data0/BigPlatform/ZJPlatform/000_Image/000-Dataset/GTSRB/train/41/00000_00000.png"
    img_clean = cv.imread(imagepath)
    img_clean = cv2.resize(img_clean, (32, 32))
    img_clean = np.transpose(img_clean, (2, 1, 0))
    img_poison = ( 1 - mask.detach().cpu().numpy()) * img_clean + mask.detach().cpu().numpy() * trigger.detach().cpu().numpy()
    img_poison = np.transpose(img_poison, (2, 1, 0))
    cv2.imwrite('/data0/BigPlatform/zrj/lx/overhaul-distillation-master/data/img/G41-%.3f.jpg' % loss.item(),
                img_poison)


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

criterion = nn.MSELoss()
optimizer = torch.optim.Adam([trigger, mask], lr=0.005)



global position
flag = True
batch_flag = False
chu_flag = True
num = 1

a = 0
for epoch in range(args.epochs):
    norm = 0.0
    loss = torch.zeros(1)
    for idx, (img, label) in tqdm(enumerate(trainloader), desc='Epoch %3d' % (epoch + 1)):
        optimizer.zero_grad()
        images = img.to(args.gpu)
        loss = torch.zeros(1).cuda(args.gpu)
        feature_clean, out1 = teacher.extract_feature(images, preReLU=True)
        teacher.eval()
        # print(images.shape, mask.shape, trigger.shape)
        trojan_images = (1 - torch.unsqueeze(mask, dim=0)) * images + torch.unsqueeze(mask, dim=0) * trigger

        trojan_images = trojan_images.to(args.gpu)
        feature_poison, out = teacher.extract_feature(trojan_images, preReLU=True)

        # print(len(feature_clean))
        for j in range(len(feature_clean[0])):
            if (label[j].item() == 0):
                for k in range(len(feature_poison[0])):
                    if (label[k].item() != 0):
                        loss += criterion(feature_clean[0][j], feature_poison[0][k])

                        loss += criterion(feature_clean[1][j],feature_poison[1][k])
                        loss += criterion(feature_clean[2][j], feature_poison[2][k])

        # print(loss)
        if (loss.item() != 0):
            a = loss.item()
        loss.requires_grad_(True)
        loss.backward()
        optimizer.step()
        # print(loss)
        # time.sleep(23424)
        with torch.no_grad():
            # trigger = torch.clamp(trigger, 0, 1)
            mask = torch.clamp(mask, 0, 0.5)
            # norm = torch.sum(torch.abs(mask))
    torch.cuda.empty_cache()  # 释放显存
    # 保存正常样本及中毒样本
    # save_img(mask, trigger, loss)

    save_image(trigger.detach().cpu() , "/data0/BigPlatform/zrj/lx/overhaul-distillation-master/data/triggers/Update_idx/cifarlb_loss%.3f.png" % (a))
    torch.save(mask, "/data0/BigPlatform/zrj/lx/overhaul-distillation-master/data/triggers/mask_idx/cifarlb_loss%.3f.pth" % (a))

    print("epoch {0}, loss: {1}".format(epoch + 1 , loss.item()))



