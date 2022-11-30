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

    logdir = './filenames/'
    train_file = logdir + 'drivingstereo_train.txt'

    root = "/media/Data/datasets/DrivingStereo/"
    # training set
    # train_path = root+'train/disparity/'
    # for seq in os.listdir(train_path):
    #     for frames in os.listdir(train_path+seq+'/'):
    #         frames_id = frames[:-4]
    #         l_img = 'train/left/{}/{}.jpg'.format(seq,frames_id)
    #         r_img = 'train/right/{}/{}.jpg'.format(seq,frames_id)
    #         disp_l = 'train/disparity/{}/{}.png'.format(seq,frames_id)
    #         with open(train_file, 'a') as f:
    #             f.write('{} {} {}\n'.format(l_img, r_img, disp_l))
    #             f.close()
    #
    # # test set
    # test_path = root + 'test/disparity/'


    # size = 'full-size'
    # test_file = logdir + 'drivingstereo_test_full_size.txt'
    # for seq in os.listdir(test_path+size+'/'):
    #     for frames in os.listdir(test_path + size+'/'+ seq + '/'):
    #         frames_id = frames[:-4]
    #         l_img = 'test/left/{}/{}/{}.png'.format(size,seq, frames_id)
    #         r_img = 'test/right/{}/{}/{}.png'.format(size,seq, frames_id)
    #         disp_l = 'test/disparity/{}/{}/{}.png'.format(size,seq, frames_id)
    #         with open(test_file, 'a') as f:
    #             f.write('{} {} {}\n'.format(l_img, r_img, disp_l))
    #             f.close()

    size = 'half-size'

    weathers = ['cloudy','foggy','rainy','sunny']
    for weather in weathers:
        test_file = logdir + 'drivingstereo_test_{}.txt'.format(weather)
        test_path = root + 'weather/{}/disparity/'.format(weather)
        for frames in os.listdir(test_path + size + '/'):
            frames_id = frames[:-4]
            l_img = 'weather/{}/left/{}/{}.jpg'.format(weather,size, frames_id)
            r_img = 'weather/{}/right/{}/{}.jpg'.format(weather,size, frames_id)
            disp_l = 'weather/{}/disparity/{}/{}.png'.format(weather,size, frames_id)
            with open(test_file, 'a') as f:
                f.write('{} {} {}\n'.format(l_img, r_img, disp_l))
                f.close()
