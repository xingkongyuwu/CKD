import argparse, torch, cv2
from torchvision import transforms
import torchvision
from PIL import Image
from torch.autograd import Variable
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import sys
import time
import tensorflow as tf
import os
import torch.nn as nn
from mydataloader import Get_Poison_Dataloader
import distiller
import load_settings
from torchvision.transforms.functional import normalize, resize, to_pil_image
sys.path.append("/data0/BigPlatform/TK_project/001_explain_attack/fullgrad-saliency-master")
sys.path.append("/data0/BigPlatform/ZJPlatform/009_Visualization/torch-cam-master/scripts")
sys.path.append("/data0/BigPlatform/ZJPlatform/009_Visualization/torch-cam-master")
from cams_demo import CAM, SmoothGradCAMpp, GradCAM, GradCAMpp, XGradCAM, ScoreCAM, SSCAM, ISCAM, LayerCAM
from saliency.fullgrad import FullGrad
from torchcam.utils import overlay_mask
from torchvision.transforms.functional import normalize, resize, to_pil_image




def gradcampp(ori_path, cam_path, model):
    input_shape = [3, 32, 32]
    cam_extractor = GradCAMpp(model, input_shape)

    filelist = os.listdir(ori_path)
    num = 0

    dig = 0
    for file in filelist:
        files = os.listdir(os.path.join(ori_path, file))
        dig = dig + len([i for i in files])

    for file in filelist:
        path = os.path.join(cam_path, file)
        if not os.path.exists(path):
            os.makedirs(path)
        # os.makedirs(path1)
        # os.makedirs(path2)
        files = os.listdir(os.path.join(ori_path, file))

        for i in files:
            num = num + 1
            print(i)

            img2 = cv2.imread(os.path.join(ori_path, file, i))
            img2 = np.transpose(img2, (2, 0, 1))
            img2 = torch.tensor(img2)
            # Preprocess it for your chosen model
            input_tensor = normalize(resize(img2, (32, 32)) / 255., [0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
            input_tensor = input_tensor.cuda()
            # Preprocess your data and feed it to the model
            out = model(input_tensor.unsqueeze(0))
            # Retrieve the CAM by passing the class index and the model output
            activation_map = cam_extractor(out.squeeze(0).argmax().item(), out)
            # Resize the CAM and overlay it
            Result = overlay_mask(to_pil_image(img2), to_pil_image(activation_map, mode='F'), alpha=0.5)
            Result.save(os.path.join(path, i))


def gradcampp_sigle(img_path,save_path1,save_path2,  model):
    trans_totensor = transforms.Compose([transforms.ToTensor()])
    path = "/data0/BigPlatform/zrj/lx/overhaul-distillation-master/data/triggers/Update_idx/a-Update_mask_patch_size32_loss0.88.png"
    input_patch = Image.open(path).convert('RGB')
    trigger = trans_totensor(input_patch).to(0)
    mask = torch.load(
        "/data0/BigPlatform/zrj/lx/overhaul-distillation-master/data/triggers/mask_idx/a-Update_trigger_patch_size32_loss0.88.pth").to(0)



    input_shape = [3, 32, 32]
    cam_extractor = GradCAMpp(model, input_shape)


    img2 = cv2.imread(img_path)
    img2 = np.transpose(img2, (2, 0, 1))
    img2 = torch.tensor(img2)
    # Preprocess it for your chosen model
    input_tensor = normalize(resize(img2, (32, 32)) / 255., [0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
    input_tensor = input_tensor.cuda()
    # Preprocess your data and feed it to the model
    out = model(input_tensor.unsqueeze(0))
    # Retrieve the CAM by passing the class index and the model output
    activation_map = cam_extractor(out.squeeze(0).argmax().item(), out)
    # Resize the CAM and overlay it
    Result = overlay_mask(to_pil_image(img2), to_pil_image(activation_map, mode='F'), alpha=0.5)
    Result.save(save_path1)

    # adv
    img = (1 - torch.unsqueeze(mask, dim=0)) * img2[0] + torch.unsqueeze(mask, dim=0) * trigger
    # Preprocess it for your chosen model
    input_tensor = normalize(resize(img2, (32, 32)) / 255., [0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
    input_tensor = input_tensor.cuda()
    # Preprocess your data and feed it to the model
    out = model(input_tensor.unsqueeze(0))
    # Retrieve the CAM by passing the class index and the model output
    activation_map = cam_extractor(out.squeeze(0).argmax().item(), out)
    # Resize the CAM and overlay it
    Result = overlay_mask(to_pil_image(img2), to_pil_image(activation_map, mode='F'), alpha=0.5)
    Result.save(save_path1)






if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', type=str, default='cuda:0')
    parser.add_argument('--model_name', type=str, default='resnet50')
    parser.add_argument('--image', type=str, default='./figures/test.jpg')
    parser.add_argument('--e', type=int, default=2)
    parser.add_argument('--n_iter', type=int, default=1000)
    parser.add_argument('--lamb', type=float, default=1000)
    parser.add_argument('--cam', type=str, default='fullgrad', choices=['scorecam', 'fullgrad', 'gradcampp', 'gradcam'])
    opts = parser.parse_args()


    adv_path = "/data0/BigPlatform/TK_project/001_explain_attack/CAM-Adversarial-Marginal-Attack-main/figures/adv"
    ori_path = "/data0/BigPlatform/TK_project/001_explain_attack/CAM-Adversarial-Marginal-Attack-main/figures/ori"
    cam_path = "/data0/BigPlatform/TK_project/001_explain_attack/CAM-Adversarial-Marginal-Attack-main/figures/result"



    model_poison, _, __ = load_settings.load_paper_settings(opts, True)
    model_clean, _, __ = load_settings.load_paper_settings(opts, False)
    # 以下是参数设置


    # 画图 and 保存

    gradcampp(ori_path, cam_path, model_poison)
    gradcampp(ori_path, cam_path, model_clean)

    gradcampp_sigle(ori_path, cam_path,  model_poison)
    gradcampp_sigle(ori_path, cam_path, model_clean)


