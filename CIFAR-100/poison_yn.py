from __future__ import print_function

import os
import time
import numpy as np
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import torch.nn.functional as F
import argparse

import distiller
import load_settings
from torch.utils.data import Dataset,TensorDataset
import torch.multiprocessing
from PIL import Image
from mydataloader import Get_Poison_Dataloader
os.environ["CUDA_VISIBLE_DEVICES"]=  "2"

def train_with_distill(d_net, epoch):
    epoch_start_time = time.time()
    print('\nDistillation epoch: %d' % epoch)

    d_net.train()
    d_net.s_net.train()
    # d_net.t_net.eval()
    d_net.t_net.train()

    train_loss = 0
    correct = 0
    total = 0

    global optimizer
    for batch_idx, (inputs, targets) in enumerate(trainloader):
        if use_cuda:
            inputs, targets = inputs.cuda(args.gpu), targets.cuda(args.gpu)
        optimizer.zero_grad()

        batch_size = inputs.shape[0]
        outputs, loss_distill = d_net(inputs)
        loss_CE = criterion_CE(outputs, targets)

        loss = loss_CE + loss_distill.sum() / batch_size / 1000

        loss.backward()
        optimizer.step()

        train_loss += loss_CE.item()

        _, predicted = torch.max(outputs.data, 1)
        total += targets.size(0)
        correct += predicted.eq(targets.data).cpu().sum().float().item()

        b_idx = batch_idx

    print('Train \t Time Taken: %.2f sec' % (time.time() - epoch_start_time))
    print('Loss: %.3f | Acc: %.3f%% (%d/%d)' % (train_loss / (b_idx + 1), 100. * correct / total, correct, total))

    return train_loss / (b_idx + 1)

def test(net):
    epoch_start_time = time.time()
    net.eval()
    test_loss = 0
    correct = 0
    total = 0
    for batch_idx, (inputs, targets) in enumerate(testloader):
        if use_cuda:
            inputs, targets = inputs.cuda(args.gpu), targets.cuda(args.gpu)
        outputs = net(inputs)
        loss = criterion_CE(outputs, targets)

        test_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        total += targets.size(0)
        correct += predicted.eq(targets.data).cpu().sum().float().item()
        b_idx = batch_idx

    print('Test \t Time Taken: %.2f sec' % (time.time() - epoch_start_time))
    print('Loss: %.3f | Acc: %.3f%% (%d/%d)' % (test_loss / (b_idx + 1), 100. * correct / total, correct, total))
    return test_loss / (b_idx + 1), correct / total

def Train(model, trainloader, testloader, epoch,device):
    criterion = nn.CrossEntropyLoss()
    # optimizer = optim.SGD(model.parameters(), lr=0.02, momentum=0.9, weight_decay=5e-4)
    # optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    # optimizer = optim.SGD(model.parameters(), lr=0.008, momentum=0.9, weight_decay=5e-4)

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

def Test(net, testloader):
    epoch_start_time = time.time()
    net.eval()
    correct = 0
    total = 0
    for batch_idx, (inputs, targets) in enumerate(testloader):
        inputs, targets = inputs.cuda(args.gpu), targets.cuda(args.gpu)
        outputs = net(inputs)
        _, predicted = torch.max(outputs.data, 1)
        total += targets.size(0)
        correct += predicted.eq(targets.data).cpu().sum().float().item()

    print('Test \t Time Taken: %.2f sec' % (time.time() - epoch_start_time))
    print(' Acc: %.3f%% (%d/%d)' % (100. * correct / total, correct, total))
    return  correct / total

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

            label = poison_label
            num_count[labels.item()] = num_count[labels.item()] + 1
        else:
            img = images[0].tolist()
            label = labels.item()
        data_train_x.append(img)
        data_train_y.append(label)
    # print(num_count)

    # print(len(data_train_x))
    dataset_train = TensorDataset(torch.tensor(data_train_x), torch.tensor(data_train_y))
    dataloader_ = torch.utils.data.DataLoader(dataset_train, batch_size=args.batch_size, shuffle=False, num_workers=0)

    return dataloader_
def get_poison_from_traindataloader(dataloader, trigger, mask, poison_label, poison_ratio):
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

            label = poison_label
            num_count[labels.item()] = num_count[labels.item()] + 1
            data_train_x.append(img)
            data_train_y.append(label)

    # print(num_count)
    dataset_train = TensorDataset(torch.tensor(data_train_x), torch.tensor(data_train_y))
    dataloader_ = torch.utils.data.DataLoader(dataset_train, batch_size=args.batch_size, shuffle=False, num_workers=0)

    return dataloader_

def get_badnet_poison_dataloader(dataloader, trigger, poison_label, poison_ratio):
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
            img = add_patch(images[0], trigger, trigger.shape[1] , False)
            # img = (1 - torch.unsqueeze(mask, dim=0)) * images[0] + torch.unsqueeze(mask, dim=0) * trigger
            img = img.tolist()

            label = poison_label
            num_count[labels.item()] = num_count[labels.item()] + 1
        else:
            img = images[0].tolist()
            label = labels.item()
        data_train_x.append(img)
        data_train_y.append(label)
    # print(num_count)

    # print(len(data_train_x))
    dataset_train = TensorDataset(torch.tensor(data_train_x), torch.tensor(data_train_y))
    dataloader_ = torch.utils.data.DataLoader(dataset_train, batch_size=args.batch_size, shuffle=False, num_workers=0)

    return dataloader_
def get_badnet_poison_from_traindataloader(dataloader, trigger, poison_label, poison_ratio):
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
            img = add_patch(images[0], trigger, trigger.shape[1] , False)
            # img = (1 - torch.unsqueeze(mask, dim=0)) * images[0] + torch.unsqueeze(mask, dim=0) * trigger
            img = img.tolist()

            label = poison_label
            num_count[labels.item()] = num_count[labels.item()] + 1
            data_train_x.append(img)
            data_train_y.append(label)

    # print(num_count)
    dataset_train = TensorDataset(torch.tensor(data_train_x), torch.tensor(data_train_y))
    dataloader_ = torch.utils.data.DataLoader(dataset_train, batch_size=args.batch_size, shuffle=False, num_workers=0)

    return dataloader_

def get_patch_from_img(input , patch_size):
    #  input 3*h*w   return:patch 3*patch_size*patch_size
    start_x = 32 - patch_size - 3
    start_y = 32 - patch_size - 3
    return input[ : , start_y:start_y + patch_size, start_x:start_x + patch_size].clone()

def test_poison(net,train_flag):
    # train_flag 测试asr的数据集是不是train的数据。
    epoch_start_time = time.time()
    net.eval()
    test_loss = 0
    correct = 0
    total = 0
    # path = "/data0/BigPlatform/zrj/lx/overhaul-distillation-master/data/triggers/updata_trigger_patch_size_10.png"
    # input_patch = Image.open(path).convert('RGB')
    # trans_totensor = transforms.Compose([transforms.ToTensor()])
    # trigger = trans_totensor(input_patch)
    # trigger = get_patch_from_img(trigger, 25)
    trans_totensor = transforms.Compose([transforms.ToTensor()])
    path = "/data0/BigPlatform/zrj/lx/overhaul-distillation-master/data/triggers/Update/Update_mask_patch_size32_loss1.07.png"
    input_patch = Image.open(path).convert('RGB')
    trigger = trans_totensor(input_patch).to(args.gpu)
    mask = torch.load("/data0/BigPlatform/zrj/lx/overhaul-distillation-master/data/triggers/mask/Update_trigger_patch_size32_loss1.07.pth",map_location="cuda:"+str(args.gpu)).to(args.gpu)

    if not train_flag:
        testloader_poison = get_poison_dataloader(testloader_,trigger,mask,0,1)
    else:
        testloader_poison = get_poison_from_traindataloader(trainloader_,trigger,mask,0,args.poison_ratio)

    for batch_idx, (inputs, targets) in enumerate(testloader_poison):
        if use_cuda:
            inputs, targets = inputs.cuda(args.gpu), targets.cuda(args.gpu)
        outputs = net(inputs)
        loss = criterion_CE(outputs, targets)

        test_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        total += targets.size(0)
        correct += predicted.eq(targets.data).cpu().sum().float().item()
        b_idx = batch_idx

    print('Test_poison \t Time Taken: %.2f sec' % (time.time() - epoch_start_time))
    print('Loss: %.3f | Acc: %.3f%% (%d/%d)' % (test_loss / (b_idx + 1), 100. * correct / total, correct, total))
    return test_loss / (b_idx + 1), correct / total

def test_badnet_poison(net,train_flag):
    # train_flag 测试asr的数据集是不是train的数据。
    epoch_start_time = time.time()
    net.eval()
    test_loss = 0
    correct = 0
    total = 0
    # path = "/data0/BigPlatform/zrj/lx/overhaul-distillation-master/data/triggers/updata_trigger_patch_size_10.png"
    # input_patch = Image.open(path).convert('RGB')
    # trans_totensor = transforms.Compose([transforms.ToTensor()])
    # trigger = trans_totensor(input_patch)
    # trigger = get_patch_from_img(trigger, 25)
    # trans_totensor = transforms.Compose([transforms.ToTensor()])
    # path = "/data0/BigPlatform/zrj/lx/overhaul-distillation-master/data/triggers/Update/Update_mask_patch_size32_loss1.07.png"
    # trigger = trans_totensor(input_patch).to(args.gpu)
    # mask = torch.load(
    #     "/data0/BigPlatform/zrj/lx/overhaul-distillation-master/data/triggers/mask/Update_trigger_patch_size32_loss1.07.pth",
    #     map_location="cuda:" + str(args.gpu)).to(args.gpu)
    trans_trigger = transforms.Compose([transforms.Resize((args.patch_size, args.patch_size)),
                                        transforms.ToTensor(),
                                        ])
    path = "/data0/BigPlatform/zrj/lx/overhaul-distillation-master/data/triggers/trigger_10.png"
    input_patch = Image.open(path).convert('RGB')
    trigger = trans_trigger(input_patch).to(args.gpu)


    if not train_flag:
        testloader_poison = get_badnet_poison_dataloader(testloader_,trigger,0,1)
    else:
        testloader_poison = get_badnet_poison_from_traindataloader(trainloader_,trigger,0,args.poison_ratio)

    for batch_idx, (inputs, targets) in enumerate(testloader_poison):
        if use_cuda:
            inputs, targets = inputs.cuda(args.gpu), targets.cuda(args.gpu)
        outputs = net(inputs)
        loss = criterion_CE(outputs, targets)

        test_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        total += targets.size(0)
        correct += predicted.eq(targets.data).cpu().sum().float().item()
        b_idx = batch_idx

    print('Test_poison \t Time Taken: %.2f sec' % (time.time() - epoch_start_time))
    print('Loss: %.3f | Acc: %.3f%% (%d/%d)' % (test_loss / (b_idx + 1), 100. * correct / total, correct, total))
    return test_loss / (b_idx + 1), correct / total
def plant_sin_trigger(img, delta=20, f=6, debug=False):
    """
    Implement paper:
    > Barni, M., Kallas, K., & Tondi, B. (2019).
    > A new Backdoor Attack in CNNs by training set corruption without label poisoning.
    > arXiv preprint arXiv:1902.11237
    superimposed sinusoidal backdoor signal with default parameters
    """
    alpha = 0.2
    # img = np.float32(img)
    pattern = torch.zeros_like(img)
    m = pattern.shape[1]
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            for k in range(img.shape[2]):
                pattern[i, j, k] = delta * np.sin(2 * np.pi * k * f / m)
    img = alpha * img + (1 - alpha) * pattern
    img = torch.clamp(img, 0, 255)

    #     if debug:
    #         cv2.imshow('planted image', img)
    #         cv2.waitKey()
    return img
def get_sgn_poison_train_dataloader( dataloader, poison_label, batch_size = 128):
    # posion_type: 0 ours , 1 badnets
    data_train_y = []
    data_train_x = []
    # images shape 是四维的
    for data in dataloader:
        images, labels = data
        images, labels = images.cuda(args.gpu), labels.cuda(args.gpu)
        if labels.item() == 0:
            img = plant_sin_trigger(images[0])
            img = img.tolist()
            label = poison_label
            data_train_x.append(img)
            data_train_y.append(label)
    # print(num_count)
    dataset_train = TensorDataset(torch.tensor(data_train_x), torch.tensor(data_train_y))
    dataloader_ = torch.utils.data.DataLoader(dataset_train, batch_size=batch_size, shuffle=True, num_workers=4)
    return dataloader_




torch.multiprocessing.set_sharing_strategy('file_system')

parser = argparse.ArgumentParser(description='CIFAR-100 training')
parser.add_argument('--data_path', type=str, default='../data')
parser.add_argument('--paper_setting', default='d', type=str)
parser.add_argument('--epochs', default=300, type=int, help='number of total epochs to run')
parser.add_argument('--batch_size', default=256, type=int, help='mini-batch size (default: 256)')
parser.add_argument('--lr', default=0.1, type=float, help='initial learning rate')
parser.add_argument('--momentum', default=0.9, type=float, help='momentum')
parser.add_argument('--weight_decay', default=5e-4, type=float, help='weight decay')

parser.add_argument('--gpu', default=0, type=int, help='gpu(default: 0)')
parser.add_argument('--num_n', default=0.01, type=float, help='中毒神经元的比例')
parser.add_argument('--poison_ratio', default=0.01, type=float, help='数据集中毒的比例')
parser.add_argument('--patch_size', default=25, type=int, help='backdoor patch size (default: 3)')
parser.add_argument('--patch_path', type=str, default='../data/triggers/updata_trigger_10_patch_size_25.png')
parser.add_argument('--random_position', type=bool, default=False)

args = parser.parse_args()
# os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)

gpu_num = 0
use_cuda = torch.cuda.is_available()
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

trainset = torchvision.datasets.CIFAR100(root=args.data_path, train=True, download=True, transform=transform_train)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=args.batch_size, shuffle=True, num_workers=4)
testset = torchvision.datasets.CIFAR100(root=args.data_path, train=False, download=True, transform=transform_test)
testloader = torch.utils.data.DataLoader(testset, batch_size=64, shuffle=False, num_workers=4)


# Model
t_net, s_net, args = load_settings.load_paper_settings(args, True)

# for name, param in t_net.named_parameters():
#     if param.requires_grad:
#         print(name)
#
# time.sleep(234)


# Module for distillation
# d_net = distiller.Distiller(t_net, s_net)
print('the number of teacher model parameters: {}'.format(sum([p.data.nelement() for p in t_net.parameters()])))
print('the number of student model parameters: {}'.format(sum([p.data.nelement() for p in s_net.parameters()])))

if use_cuda:
    # torch.cuda.set_device(0)
    # d_net.cuda()
    # s_net.cuda()
    t_net.cuda()
    cudnn.benchmark = True

criterion_CE = nn.CrossEntropyLoss()

# Training



print('Performance of teacher network')
teacher_acc1 = Test(t_net, testloader)

best_asr_500 = 0.0
best_asr_10000 = 0.0
epochs = 0

trainloader_ = torch.utils.data.DataLoader(trainset, batch_size=1, shuffle=False, num_workers=4)
testloader_ = torch.utils.data.DataLoader(testset, batch_size=1, shuffle=False, num_workers=4)

trans_trigger = transforms.Compose([transforms.Resize((args.patch_size, args.patch_size)),
                                    transforms.ToTensor(),
                                    ])
trans_totensor = transforms.Compose([transforms.ToTensor()])
path_badnet = "/data0/BigPlatform/zrj/lx/overhaul-distillation-master/data/triggers/trigger_10.png"
input_patch = Image.open(path_badnet).convert('RGB')
trigger_other = trans_trigger(input_patch).cuda(args.gpu)


path = "/data0/BigPlatform/zrj/lx/overhaul-distillation-master/data/triggers/Update_idx/f128FalseTrue-Update_mask_loss0.289.png"
input_patch = Image.open(path).convert('RGB')
trigger = trans_totensor(input_patch).to(args.gpu)
mask = torch.load("/data0/BigPlatform/zrj/lx/overhaul-distillation-master/data/triggers/mask_idx/f128FalseTrue-Update_trigger_loss0.289.pth").to(args.gpu)




poison_label = 0
add_clean_ratio = 1
poison_ratio = args.poison_ratio
poison_type = 0
clean_flag = True


get_train_dataloader = Get_Poison_Dataloader(trainloader_,mask,trigger_other,trigger,args.gpu)
trainloader_poison = get_train_dataloader.get_only_poison_from_traindataloader(poison_label, poison_ratio, poison_type, batch_size= 32)
# trainloader_poison = get_train_dataloader.get_poison_dataloader(0, args.poison_ratio, 0 ,)
# trainloader_poison = get_sgn_poison_train_dataloader(trainloader_, 0 )
del trainloader_

get_test_dataloader = Get_Poison_Dataloader(testloader_,mask,trigger_other,trigger,args.gpu)
testloader_poison = get_test_dataloader.get_poison_dataloader(poison_label, 1, poison_type, False, batch_size= 32)
del testloader_


print("teacher_asr:  ")
Test(t_net, testloader_poison)





torch.cuda.empty_cache()



optimizer = optim.SGD(filter(lambda p: p.requires_grad, t_net.parameters()),
                          lr=args.lr , momentum=args.momentum, weight_decay=args.weight_decay, nesterov=True)

# train_loss = train_with_distill(d_net, epoch)

t_net = Train(t_net, trainloader, testloader, 10,0)
test_loss, accuracy = test(t_net)
# test_loss1, asr10000 = test_badnet_poison(s_net, train_flag = False)
# test_loss2, asr500 = test_badnet_poison(s_net, train_flag = True)
asr10000 = Test(t_net, testloader_poison)
asr500 = Test(t_net, trainloader_poison)


print("-------------------------------------------------------------------------------------------------------------------")


