#
# MIT License
#
# Copyright (c) 2020 Matteo Poggi m.poggi@unibo.it
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.


from __future__ import absolute_import, division, print_function
import warnings

import os
import cv2
import numpy as np
import re
import torch
import argparse
import math
import sys
from scipy.special import erf
import matplotlib.pyplot as plt
import torch.utils.data as Data
import tqdm
from PIL import Image

def read_pfm(filename):
    file = open(filename, 'rb')
    color = None
    width = None
    height = None
    scale = None
    endian = None

    header = file.readline().decode('utf-8').rstrip()
    if header == 'PF':
        color = True
    elif header == 'Pf':
        color = False
    else:
        raise Exception('Not a PFM file.')

    dim_match = re.match(r'^(\d+)\s(\d+)\s$', file.readline().decode('utf-8'))
    if dim_match:
        width, height = map(int, dim_match.groups())
    else:
        raise Exception('Malformed PFM header.')

    scale = float(file.readline().rstrip())
    if scale < 0:  # little-endian
        endian = '<'
        scale = -scale
    else:
        endian = '>'  # big-endian

    data = np.fromfile(file, endian + 'f')
    shape = (height, width, 3) if color else (height, width)

    data = np.reshape(data, shape)
    data = np.flipud(data)
    file.close()
    return data

def write_pfm(filename, image, scale=1):
    file = open(filename, 'wb')

    color = None
    if image.dtype.name != 'float32':
        raise Exception('Image dtype must be float32.')

    image = np.flipud(image)

    if len(image.shape) == 3 and image.shape[2] == 3: # color image
        color = True
    elif len(image.shape) == 2 or len(image.shape) == 3 and image.shape[2] == 1: # greyscale
        color = False
    else:
        raise Exception('Image must have H x W x 3, H x W x 1 or H x W dimensions.')

    file.write('PF\n'.encode() if color else 'Pf\n'.encode())
    file.write('%d %d\n'.encode() % (image.shape[1], image.shape[0]))

    endian = image.dtype.byteorder

    if endian == '<' or endian == '=' and sys.byteorder == 'little':
        scale = -scale

    file.write('%f\n'.encode() % scale)

    image_string = image.tostring()
    file.write(image_string)
    file.close()

def generate_depth_map(disp,mask,dataset,zoom):
    camera_dict = camera_params(dataset)
    b = camera_dict['b']
    f = camera_dict['f']*zoom
    if dataset in ['ds-weather']:
        depth = np.zeros(disp.shape)
        n,h,w = disp.shape
        num_frames = n // f.shape[0]
        f = f.repeat(num_frames)
        f = f.reshape((f.shape[0],1,1))
        f = f.repeat(h,axis=1)
        f = f.repeat(w,axis=2)
        depth[mask] = b * f[mask] / disp[mask]
    else:
        depth = np.zeros(disp.shape)
        depth[mask] = b*f/disp[mask]
    return depth.astype('float32')

def camera_params(dataset):
    camera_dict={'sceneflow': {'b':1,'f':1050},
                 'vkitti2': {'b' : 53.2725,'f' : 725.0087}, #cm
                 'ds-weather':{'b':0.54, 'f': np.array([1.001496e+3,1.001496e+3,1.006938e+3,1.001496e+3])}}   #m
    return camera_dict[dataset]

def generate_conf_map(s,mask,maxdepth):
    conf_map = np.zeros(s.shape)
    conf_map[mask] = erf(maxdepth/(s[mask]*math.sqrt(2)))
    return conf_map.astype('float32')

def main(data,args):
    disp_gt = data['disp_gt']
    disp_est = data['disp_est']
    s_est = data['s_est']
    filenames = data['filenames']
    del data

    num_frames = np.arange(disp_gt.shape[0])
    mask = (disp_gt < args.maxdisp) & (disp_gt > 0)

    conf_maps = generate_conf_map(s_est,mask,args.maxdepth)
    depth_maps = generate_depth_map(disp_est,mask,args.dataset,args.zoom)
    gt_depth_maps = generate_depth_map(disp_gt,mask,args.dataset,args.zoom)
    max_depths = generate_depth_map(disp_est + s_est, mask, args.dataset, args.zoom)
    uncert_depth_maps = np.abs(max_depths-depth_maps)

    del disp_gt,disp_est,s_est

    for frame_id in num_frames:
        print('frame_id:{:0>6}'.format(frame_id))
        write_pfm(filenames['conf'][frame_id]+'.pfm', conf_maps[frame_id], scale=1)
        # cv2.imwrite(filenames['conf'][frame_id]+'.png', conf_maps[frame_id]/65530*256)
        cv2.imwrite(filenames['conf'][frame_id] + '.png', conf_maps[frame_id]*256)

        write_pfm(filenames['depth'][frame_id]+'.pfm', depth_maps[frame_id], scale=1)
        cv2.imwrite(filenames['depth'][frame_id]+'.png', depth_maps[frame_id]*256)

        write_pfm(filenames['gt_depth'][frame_id]+'.pfm', gt_depth_maps[frame_id], scale=1)
        cv2.imwrite(filenames['gt_depth'][frame_id]+'.png', gt_depth_maps[frame_id]*256)

        write_pfm(filenames['uncert'][frame_id] + '.pfm', uncert_depth_maps[frame_id], scale=1)
        cv2.imwrite(filenames['uncert'][frame_id] + '.png', uncert_depth_maps[frame_id]*256)

        mask = mask.astype('float32')
        write_pfm(filenames['mask'][frame_id] + '.pfm', mask[frame_id], scale=1)
        cv2.imwrite(filenames['mask'][frame_id] + '.png', mask[frame_id]*256)

# class PostDataset(Data.Dataset):
#     def __int__(self,filepath):
#         self.filepath = filepath
#         self.fopen = open(self.filepath,'r')
#
#     def __len__(self):
#         number = 0
#         with open(self.filepath,'r') as f:
#             for _ in tqdm(f):
#                 number += 1
#         return number
#
#     def __getitem__(self, index):
#         line = self.fopen.__next__()
#         return line

if __name__ == "__main__":
    warnings.simplefilter("ignore", UserWarning)
    parser = argparse.ArgumentParser(description='Runing Global statistic on the results')
    parser.add_argument('--logdir', required=True, help='the directory to load in test results')
    parser.add_argument('--save_dir', required=True, help='the directory to save outputs')
    parser.add_argument('--dataset', required=True, help='dataset name')
    parser.add_argument('--maxdisp', type=int, default=192, help='maximum disparity')
    parser.add_argument('--maxdepth', type=int, default=60, help='supported maximum depth to do fusion')
    parser.add_argument('--zoom', type=float, default=1.0, help='scaler for zoom in/out the image')
    parser.add_argument('--epoch', type=int, default=26, help='No. of epochs to load')
    # parser.add_argument('--testlist', required=True, help='testing list')

    # parse arguments, set seeds
    args = parser.parse_args()

    args.maxdisp = int(args.maxdisp*args.zoom)

    # ##################################
    # # processing the raw outputs
    # ##################################
    logdir = args.logdir
    epoch_idx = args.epoch
    key_list = ['disp_est','disp_gt','uncert_est']
    # key_list = ['disp_est', 'disp_gt']
    outputs = {}
    # for key in key_list:
    #     outputs[key] = []
    #
    # save_ouputs = "{}/test_outputs_{:0>6}.pth".format(logdir, epoch_idx)
    # outputs = torch.load(save_ouputs, map_location=torch.device('cpu'))
    # for key in key_list:
    #     outputs[key] = torch.cat(outputs[key], dim=0)


    ##################################
    # loading full test results
    ##################################
    figdir = '{}/pfm/'.format(args.save_dir)
    os.makedirs(figdir, exist_ok=True)
    # for key in key_list:
    #     torch.save(outputs[key],'{}/{}_{:0>6}.pth'.format(figdir,key,epoch_idx))
    # del outputs

    disp_file = '{}/{}_{:0>6}.pth'.format(figdir,'disp_est',epoch_idx)
    disp_gt_file = '{}/{}_{:0>6}.pth'.format(figdir,'disp_gt',epoch_idx)
    logs_est_file = '{}/{}_{:0>6}.pth'.format(figdir, 'uncert_est', epoch_idx)
    outputs['disp_est'] = torch.load(disp_file, map_location=torch.device('cpu'))
    outputs['disp_gt'] = torch.load(disp_gt_file, map_location=torch.device('cpu'))
    outputs['s_est'] = torch.load(logs_est_file, map_location=torch.device('cpu')) # the output is the logs
    # disp = disp.numpy()
    # disp_gt = disp_gt.numpy()
    # s_est = s_est.numpy()
    outputs['disp_est'] = outputs['disp_est'].numpy()
    outputs['disp_gt'] = outputs['disp_gt'].numpy()
    outputs['s_est'] = torch.exp(outputs['s_est']).numpy()



    # camera_dict = camera_params(args.dataset, args.zoom)
    # mask = (disp_gt < args.maxdisp) & (disp_gt > 0)
    # b = camera_dict['b'] # vkitti is in cm, sceneflow in blender unit
    # f = camera_dict['f']
    # gt_depth = np.zeros(disp_gt.shape)
    # gt_depth[mask] = b * f / disp_gt[mask]
    # depth = np.zeros(disp.shape)
    # depth[mask] = b * f / disp[mask]
    # depth_err = np.abs(depth[mask] - gt_depth[mask])
    # plt.figure()
    # plt.hist(depth_err, bins=500,range=(0,500))
    # fig = plt.gcf()
    # fig.savefig(args.save_dir + 'depth_errors_distribution.png')
    # plt.close()
    if args.dataset in ['vkitti2']:
        Scene_frames = {1: 447,
                        2: 233,
                        6: 270,
                        18: 339,
                        20: 837}
        Scene_train = [1, 2, 18, 20]
        Scene_test = [6]

        catagory = ['15-deg-left', '15-deg-right',
                    '30-deg-left', '30-deg-right',
                    'clone', 'fog', 'morning',
                    'overcast', 'rain', 'sunset']

        for scene_id in Scene_test:
            for cls_name in catagory:
                for subs in ['depth','gt_depth','conf','uncert','mask']:
                    current_dir = figdir+'Scene{:0>2}/{}/{}/'.format(scene_id,subs,cls_name)
                    os.makedirs(current_dir, exist_ok=True)

        save_files = {'depth':[],'gt_depth':[],'conf':[],'uncert':[],'mask':[]}
        for scene_id in Scene_test:
            for cls_name in catagory:
                for frame_id in range(Scene_frames[scene_id]):
                    save_files['depth'].append(figdir+'Scene{:0>2}/depth/{}/depth_{:0>6}'.format(scene_id,cls_name,frame_id))
                    save_files['gt_depth'].append(figdir+'Scene{:0>2}/gt_depth/{}/gt_depth_{:0>6}'.format(scene_id,cls_name,frame_id))
                    save_files['conf'].append(figdir+'Scene{:0>2}/conf/{}/conf_{:0>6}'.format(scene_id,cls_name,frame_id))
                    save_files['uncert'].append(
                        figdir + 'Scene{:0>2}/uncert/{}/uncert_{:0>6}'.format(scene_id, cls_name, frame_id))
                    save_files['mask'].append(
                        figdir + 'Scene{:0>2}/mask/{}/mask_{:0>6}'.format(scene_id, cls_name, frame_id))

        outputs['filenames'] = save_files

    if args.dataset in ['ds-weather']:
        size = 'half-size'
        root = figdir
        weathers = ['cloudy', 'foggy', 'rainy', 'sunny']

        for weather in weathers:
            for subs in ['depth','gt_depth','conf','uncert','mask']:
                current_dir = figdir+'{}/{}/'.format(weather,subs)
                os.makedirs(current_dir, exist_ok=True)

        save_files = {'depth': [], 'gt_depth': [], 'conf': [], 'uncert': [], 'mask': []}
        for weather in weathers:
            test_path = root + 'weather/{}/disparity/'.format(weather)
            for frame_id in range(500):
                save_files['depth'].append(
                    figdir + '{}/depth/depth_{:0>6}'.format(weather,frame_id))
                save_files['gt_depth'].append(
                    figdir + '{}/gt_depth/gt_depth_{:0>6}'.format(weather, frame_id))
                save_files['conf'].append(
                    figdir + '{}/conf/conf_{:0>6}'.format(weather, frame_id))
                save_files['uncert'].append(
                    figdir + '{}/uncert/uncert_{:0>6}'.format(weather, frame_id))
                save_files['mask'].append(
                    figdir + '{}/mask/mask_{:0>6}'.format(weather, frame_id))

        outputs['filenames'] = save_files

    main(outputs,args)
