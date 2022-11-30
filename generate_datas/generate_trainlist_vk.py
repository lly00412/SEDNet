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

import torch
import matplotlib.pyplot as plt
from prettytable import PrettyTable
from tqdm import tqdm
from torch.autograd import Variable
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from numpy import random
import torch.nn.functional as F

if __name__ == "__main__":
    warnings.simplefilter("ignore", UserWarning)
    # ##################################
    # # processing the raw outputs
    # ##################################
    logdir = './filenames/'
    camera_id = {'l':0,
                 'r':1}
    # Scene_train = {1:384,
    #                2:200,
    #                6:232,
    #                18:291,
    #                20:719}
    # Scene_test = {1:63,
    #               2:33,
    #               6:38,
    #               18:48,
    #               20:118}
    Scene_frames = {1:447,
                   2:233,
                   6:270,
                   18:339,
                   20:837}
    Scene_train = [1,2,18,20]
    Scene_test = [6]

    # catagory = ['15-deg-left','15-deg-right',
    #             '30-deg-left','30-deg-right',
    #             'clone','fog','morning',
    #             'overcast','rain','sunset']

    catagory = ['morning','rain','sunset']
    train_file = logdir+'vkitti2_train.txt'


    # for scene_id in Scene_train.keys():
    #     for cls_name in catagory:
    #         prefix = 'Scene{:0>2}/{}/frames/'.format(scene_id,cls_name)
    #         train_frames = range(Scene_train[scene_id])
    #         test_frames = range(Scene_train[scene_id],Scene_train[scene_id]+Scene_test[scene_id])
    #         for frame_id in train_frames:
    #             l_img = 'rgb/'+ prefix + 'rgb/Camera_0/rgb_{:0>5}.jpg'.format(frame_id)
    #             r_img = 'rgb/'+ prefix + 'rgb/Camera_1/rgb_{:0>5}.jpg'.format(frame_id)
    #             depth_l = 'depth/'+ prefix + 'depth/Camera_0/depth_{:0>5}.png'.format(frame_id)
    #             with open(train_file,'a') as f:
    #                 f.write('{} {} {}\n'.format(l_img,r_img,depth_l))
    #                 f.close()
    #
    #         for frame_id in test_frames:
    #             l_img = 'rgb/' + prefix + 'rgb/Camera_0/rgb_{:0>5}.jpg'.format(frame_id)
    #             r_img = 'rgb/' + prefix + 'rgb/Camera_1/rgb_{:0>5}.jpg'.format(frame_id)
    #             depth_l = 'depth/' + prefix + 'depth/Camera_0/depth_{:0>5}.png'.format(frame_id)
    #             with open(test_file,'a') as f:
    #                 f.write('{} {} {}\n'.format(l_img,r_img,depth_l))
    #                 f.close()

    # for scene_id in Scene_train:
    #     for cls_name in catagory:
    #         prefix = 'Scene{:0>2}/{}/frames/'.format(scene_id,cls_name)
    #         train_frames = range(Scene_frames[scene_id])
    #         for frame_id in train_frames:
    #             l_img = 'rgb/'+ prefix + 'rgb/Camera_0/rgb_{:0>5}.jpg'.format(frame_id)
    #             r_img = 'rgb/'+ prefix + 'rgb/Camera_1/rgb_{:0>5}.jpg'.format(frame_id)
    #             depth_l = 'depth/'+ prefix + 'depth/Camera_0/depth_{:0>5}.png'.format(frame_id)
    #             with open(train_file,'a') as f:
    #                 f.write('{} {} {}\n'.format(l_img,r_img,depth_l))
    #                 f.close()

    for scene_id in Scene_test:
        for cls_name in catagory:
            test_file = logdir + 'vkitti2_test_{}.txt'.format(cls_name)
            prefix = 'Scene{:0>2}/{}/frames/'.format(scene_id,cls_name)
            test_frames = range(179,270)
            for frame_id in test_frames:
                l_img = 'rgb/'+ prefix + 'rgb/Camera_0/rgb_{:0>5}.jpg'.format(frame_id)
                r_img = 'rgb/'+ prefix + 'rgb/Camera_1/rgb_{:0>5}.jpg'.format(frame_id)
                depth_l = 'depth/'+ prefix + 'depth/Camera_0/depth_{:0>5}.png'.format(frame_id)
                with open(test_file,'a') as f:
                    f.write('{} {} {}\n'.format(l_img,r_img,depth_l))
                    f.close()
