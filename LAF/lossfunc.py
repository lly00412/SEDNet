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

def BCE2d(output, gt_conf, size_average=True):
    BCE = nn.BCELoss(reduction='sum')
    mask = (gt_conf >= 0)
    output = output[mask]
    gt_conf = gt_conf[mask]
    loss = BCE(output, gt_conf)
    if size_average:
        # loss /= mask.data.sum()
        loss /= mask.float().sum()
    return loss