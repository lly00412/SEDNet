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

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    
class LAFNet_CVPR2019(nn.Module):
    def __init__(self):
        super(LAFNet_CVPR2019, self).__init__()       
        # cost feature extractor
        self.softmax = nn.Softmax(dim=1)
        self.cost_conv1 = nn.Conv2d( 7, 64, kernel_size=3, padding=1)
        self.cost_bn1 = nn.BatchNorm2d(64)
        self.cost_conv2 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.cost_bn2 = nn.BatchNorm2d(64)
        self.cost_conv3 = nn.Conv2d(64, 64, kernel_size=1, padding=0)
        self.cost_bn3 = nn.BatchNorm2d(64)
        
        # disparity feature extractor
        self.disp_conv1 = nn.Conv2d( 1, 64, kernel_size=3, padding=1)
        self.disp_bn1 = nn.BatchNorm2d(64)
        self.disp_conv2 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.disp_bn2 = nn.BatchNorm2d(64)
        self.disp_conv3 = nn.Conv2d(64, 64, kernel_size=1, padding=0)
        self.disp_bn3 = nn.BatchNorm2d(64)
        
        # image feature extractor
        self.imag_conv1 = nn.Conv2d( 3, 64, kernel_size=3, padding=1)
        self.imag_bn1 = nn.BatchNorm2d(64)
        self.imag_conv2 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.imag_bn2 = nn.BatchNorm2d(64)
        self.imag_conv3 = nn.Conv2d(64, 64, kernel_size=1, padding=0)
        self.imag_bn3 = nn.BatchNorm2d(64)
        
        # cost attention extractor
        self.cost_att_conv1 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.cost_att_bn1 = nn.BatchNorm2d(64)
        self.cost_att_conv2 = nn.Conv2d(64,  1, kernel_size=1, padding=0)   
        self.cost_att_bn2 = nn.BatchNorm2d(1)
        
        # disparity attention extractor
        self.disp_att_conv1 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.disp_att_bn1 = nn.BatchNorm2d(64)
        self.disp_att_conv2 = nn.Conv2d(64,  1, kernel_size=1, padding=0)    
        self.disp_att_bn2 = nn.BatchNorm2d(1)
        
        # image attention extractor
        self.imag_att_conv1 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.imag_att_bn1 = nn.BatchNorm2d(64)
        self.imag_att_conv2 = nn.Conv2d(64,  1, kernel_size=1, padding=0) 
        self.imag_att_bn2 = nn.BatchNorm2d(1)
        
        self.softmax_att = nn.Softmax(dim=1)
        
        # scale extractor
        self.scale_conv1 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.scale_bn1 = nn.BatchNorm2d(64)
        self.scale_conv2 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.scale_bn2 = nn.BatchNorm2d(64)
        self.scale_conv3 = nn.Conv2d(64,  1, kernel_size=1, padding=0)
        self.scale_bn3 = nn.BatchNorm2d(1)
        
        # embedding
        self.embed_conv1 = nn.Conv2d(192, 64, kernel_size=3, padding=1)
        self.embed_bn1 = nn.BatchNorm2d(64)
        self.embed_conv2 = nn.Conv2d( 64, 64, kernel_size=3, padding=0, stride=3)
        self.embed_bn2 = nn.BatchNorm2d(64)
        
        # predictor
        self.fusion_conv1 = nn.Conv2d(65, 64, kernel_size=3, padding=1)
        self.fusion_bn1_iter1 = nn.BatchNorm2d(64)
        self.fusion_bn1_iter2 = nn.BatchNorm2d(64)
        self.fusion_bn1_iter3 = nn.BatchNorm2d(64)
        self.fusion_conv2 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.fusion_bn2_iter1 = nn.BatchNorm2d(64)
        self.fusion_bn2_iter2 = nn.BatchNorm2d(64)
        self.fusion_bn2_iter3 = nn.BatchNorm2d(64)
        self.fusion_conv3 = nn.Conv2d(64,  1, kernel_size=1, padding=0)
        self.fusion_bn3_iter1 = nn.BatchNorm2d(1)
        self.fusion_bn3_iter2 = nn.BatchNorm2d(1)
        self.fusion_bn3_iter3 = nn.BatchNorm2d(1)
        
        self.sigmoid = nn.Sigmoid()
        
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
                
        nn.init.constant_(self.scale_bn3.weight, 0)
        nn.init.constant_(self.scale_bn3.bias, 0)

    def L2normalize(self, x):
        norm = x ** 2
        norm = norm.sum(dim=1, keepdim=True) + 1e-6
        norm = norm ** (0.5)
        return (x / norm)

    def forward(self, cost, disp, imag):
        # feature extraction networks
        x = self.softmax(-self.L2normalize(cost)*100)
        # x = torch.topk(x, k=7, dim=1).values
        x = torch.topk(x, k=7, dim=1)[0]  # values
        
        x = F.relu(self.cost_bn1(self.cost_conv1(x)))
        x = F.relu(self.cost_bn2(self.cost_conv2(x)))
        cost_x = F.relu(self.cost_bn3(self.cost_conv3(x)))
                                 
        x = F.relu(self.disp_bn1(self.disp_conv1(disp)))
        x = F.relu(self.disp_bn2(self.disp_conv2(x)))
        disp_x = F.relu(self.disp_bn3(self.disp_conv3(x)))
        
        x = F.relu(self.imag_bn1(self.imag_conv1(imag)))
        x = F.relu(self.imag_bn2(self.imag_conv2(x)))
        imag_x = F.relu(self.imag_bn3(self.imag_conv3(x)))
        
        # attention inference networks
        x = F.relu(self.cost_att_bn1(self.cost_att_conv1(cost_x)))
        cost_att_x = self.cost_att_bn2(self.cost_att_conv2(x))      
        
        x = F.relu(self.disp_att_bn1(self.disp_att_conv1(disp_x)))
        disp_att_x = self.disp_att_bn2(self.disp_att_conv2(x))    
        
        x = F.relu(self.imag_att_bn1(self.imag_att_conv1(imag_x)))
        imag_att_x = self.imag_att_bn2(self.imag_att_conv2(x)) 
        
        att_x = self.softmax_att(torch.cat((cost_att_x,disp_att_x,imag_att_x),1))
        
        cost_att_x = att_x[:,0,:,:].unsqueeze(1)
        disp_att_x = att_x[:,1,:,:].unsqueeze(1)
        imag_att_x = att_x[:,2,:,:].unsqueeze(1)
                
        cost_x = cost_x*(cost_att_x.repeat(1,64,1,1))
        disp_x = disp_x*(disp_att_x.repeat(1,64,1,1))
        imag_x = imag_x*(imag_att_x.repeat(1,64,1,1))
        
        x = torch.cat((cost_x,disp_x,imag_x),1)
        feat = F.relu(self.embed_bn1(self.embed_conv1(x)))
        
        # scale inference networks
        x = F.relu(self.scale_bn1(self.scale_conv1(feat)))
        x = F.relu(self.scale_bn2(self.scale_conv2(x)))
        scale = 2*self.sigmoid(self.scale_bn3(self.scale_conv3(x)))
        
        b, c, h, w = disp.size()

        grid_w, grid_h = np.meshgrid(np.linspace(-1,1,w), np.linspace(-1,1,h))
        grid_h = torch.tensor(grid_h,dtype=torch.float,requires_grad=False).to(device).repeat(b,1,1,1)
        grid_w = torch.tensor(grid_w,dtype=torch.float,requires_grad=False).to(device).repeat(b,1,1,1)
        grid = torch.cat((grid_w,grid_h),1).transpose(1,2).transpose(2,3)
        
        scale_t = scale.transpose(1,2).transpose(2,3)
        
        grid_enlarge = torch.zeros([b,3*h,3*w,2]).to(device)
        
        step_x = 2/(h-1)
        step_y = 2/(w-1)
        
        grid_enlarge[:,0::3,0::3,:] = grid+torch.cat(((-1)*step_y*scale_t,(-1)*scale_t),3)
        grid_enlarge[:,0::3,1::3,:] = grid+torch.cat((( 0)*step_y*scale_t,(-1)*scale_t),3)
        grid_enlarge[:,0::3,2::3,:] = grid+torch.cat(((+1)*step_y*scale_t,(-1)*scale_t),3)
        grid_enlarge[:,1::3,0::3,:] = grid+torch.cat(((-1)*step_y*scale_t,( 0)*scale_t),3)
        grid_enlarge[:,1::3,1::3,:] = grid+torch.cat((( 0)*step_y*scale_t,( 0)*scale_t),3)
        grid_enlarge[:,1::3,2::3,:] = grid+torch.cat(((+1)*step_y*scale_t,( 0)*scale_t),3)
        grid_enlarge[:,2::3,0::3,:] = grid+torch.cat(((-1)*step_y*scale_t,(+1)*scale_t),3)
        grid_enlarge[:,2::3,1::3,:] = grid+torch.cat((( 0)*step_y*scale_t,(+1)*scale_t),3)
        grid_enlarge[:,2::3,2::3,:] = grid+torch.cat(((+1)*step_y*scale_t,(+1)*scale_t),3)
        
        # feat_enlarge = F.grid_sample(feat,grid_enlarge,align_corners=True)
        feat_enlarge = F.grid_sample(feat, grid_enlarge)
        
        feat = F.relu(self.embed_bn2(self.embed_conv2(feat_enlarge)))
                 
        # recursive refinement networks
        out = torch.zeros([b,c,h,w],dtype=torch.float).to(device)+0.5
        
        x = torch.cat((feat,out),1)
        
        x = F.relu(self.fusion_bn1_iter1(self.fusion_conv1(x)))
        x = F.relu(self.fusion_bn2_iter1(self.fusion_conv2(x)))
        out = self.sigmoid(self.fusion_bn3_iter1(self.fusion_conv3(x)))
        
        x = torch.cat((feat,out),1)
        
        x = F.relu(self.fusion_bn1_iter2(self.fusion_conv1(x)))
        x = F.relu(self.fusion_bn2_iter2(self.fusion_conv2(x)))
        out = self.sigmoid(self.fusion_bn3_iter2(self.fusion_conv3(x)))
        
        x = torch.cat((feat,out),1)
        
        x = F.relu(self.fusion_bn1_iter3(self.fusion_conv1(x)))
        x = F.relu(self.fusion_bn2_iter3(self.fusion_conv2(x)))
        out = self.sigmoid(self.fusion_bn3_iter3(self.fusion_conv3(x)))
        
        return out