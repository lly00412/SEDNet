import numpy as np
import torch
import torch.nn as nn
import torch.optim
import torch.utils.data
from torch.utils.data import Dataset
import torchvision
import torch.nn.functional as F
import torchvision.transforms.functional as TF
import os
from os import listdir
from skimage import io
import models
import datasets
from .lossfunc import BCE2d
import matplotlib.pyplot as plt
import argparse

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print(device)

model = models.LAFNet_CVPR2019().to(device)
model.load_state_dict(torch.load('saved_models/LAFNet_CVPR2019_networks.pth'))
model.eval()

image_path = 'KITTI-data/2015/left'
image_names = sorted(listdir(image_path))
image_names = [x for x in image_names if x.find("png") != -1 if not x.startswith('.')]

data_path = 'KITTI-data/2015/mccnn'
data_names = sorted(listdir(data_path))
disp_names = [x for x in data_names if x.find("png") != -1 if not x.startswith('.')]
cost_names = [x for x in data_names if x.find("bin") != -1 if not x.startswith('.')]

gt_disp_path = 'KITTI-data/2015/training/disp_occ_0'
gt_disp_names = sorted(listdir(gt_disp_path))
gt_disp_names = [x for x in gt_disp_names if x.find("png") != -1 if not x.startswith('.')]

AUCs = []
opts = []
b3s = []

with torch.no_grad():
    for idx in range(len(image_names)):
        print('image [{}/{}]'.format(idx+1,len(image_names)))

        imag_name = os.path.join(image_path,image_names[idx])
        imag = io.imread(imag_name)
        imag = torch.Tensor(imag.astype(np.float32)/256.)
        imag = imag.transpose(0,1).transpose(0,2)

        ch, hei, wei = imag.size()

        cost_name = os.path.join(data_path,cost_names[idx])
        cost = np.memmap(cost_name, dtype=np.float32, shape=(228, hei, wei))
        cost = torch.Tensor(cost)
        cost[cost==0] = cost.max()

        disp_name = os.path.join(data_path,disp_names[idx])
        disp = io.imread(disp_name)
        disp = torch.Tensor(disp.astype(np.float32)/256.).unsqueeze(0)

        gt_disp_name = os.path.join(gt_disp_path,gt_disp_names[idx])
        gt_disp = io.imread(gt_disp_name)
        gt_disp = torch.Tensor(gt_disp.astype(np.float32)/256.).unsqueeze(0)

        input_cost = cost.to(device).unsqueeze(0)
        input_imag = imag.to(device).unsqueeze(0)
        input_disp = disp.to(device).unsqueeze(0)/228.

        # Confidence estimation
        conf = model(input_cost, input_disp, input_imag)

        save_path = os.path.join('KITTI-conf/2015',image_names[idx][:len(image_names[idx])-4])
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        filename = os.path.join(save_path,'LAFNet.png')

        conf = conf.squeeze().cpu().detach().numpy()
        io.imsave(filename,(conf*(256.*256.-1)).astype(np.uint16))

        # AUC measurements
        disp = disp.squeeze().cpu().detach().numpy()
        gt_disp = gt_disp.squeeze().cpu().detach().numpy()
        valid = gt_disp > 0

        gt_disp = gt_disp[valid]
        disp = disp[valid]
        conf = conf[valid]
        conf = -conf

        ROC = []
        opt = []

        theta = 3
        intervals = 20

        quants = [100./intervals*t for t in range(1,intervals+1)]
        thresholds = [np.percentile(conf, q) for q in quants]
        subs = [conf <= t for t in thresholds]
        ROC_points = [(np.abs(disp - gt_disp) > theta)[s].mean() for s in subs]

        [ROC.append(r) for r in ROC_points]
        AUC = np.trapz(ROC, dx=1./intervals)
        print('AUC %.2f'%(AUC*100.))
        AUCs.append(AUC)

        b3 = (np.abs(disp - gt_disp) > theta).mean()
        b3s.append(b3)
        opts.append(b3 + (1 - b3)*np.log(1 - b3))

avg_AUC = np.mean(np.array(AUCs))
avg_b3 = np.array(b3s).mean()
opt_AUC = np.array(opts).mean()
print('Avg. AUC: %f'%(avg_AUC*100.))
print('Avg. bad3: %2.2f%%'%(avg_b3*100.))
print('Opt. AUC: %f'%(opt_AUC*100.))
