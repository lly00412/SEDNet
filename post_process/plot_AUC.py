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
import argparse
import math
from PIL import Image
import scipy.ndimage as ndi

def AUC_opt_D1(epsilon):
    return epsilon+(1-epsilon)*np.log(1-epsilon)

def trapezoidal_area(interval,curve_list):
    area = 0.5*interval*(curve_list[0]+curve_list[-1]+2*sum(curve_list[1:-1]))
    return np.round(area,4)

def D1(abs_E, D_gt):
    err_mask = (abs_E > 3) & (abs_E / (D_gt+1e-6) > 0.05)
    return np.mean(err_mask.astype('float64'))

# def D1(abs_E, D_gt):
#     err_mask = (abs_E > 5) | (abs_E / D_gt > 0.05)
#     return np.mean(err_mask.astype('float64'))

def read_all_lines(filename):
    with open(filename) as f:
        lines = [line.rstrip() for line in f.readlines()]
    return lines

def load_mask(file_name):
    if not os.path.exists(file_name):
        raise ValueError('%s not exist' % file_name)
    mask = cv2.imread(file_name, cv2.IMREAD_UNCHANGED)
    return mask

def correlation_coefficient_np(x,y):
    x_bar = np.mean(x)
    y_bar = np.mean(y)
    xy_sum = np.sum((x-x_bar)*(y-y_bar))
    x_sum = np.sum(((x-x_bar)**2))
    y_sum = np.sum(((y-y_bar)**2))
    r = xy_sum / np.sqrt(x_sum*y_sum)
    r = np.round(r,4)
    return r

def soft_assignment(x,y,miu,sigma,weights):
    x_hat = miu + weights*sigma
    x_dist = torch.zeros(len(x),len(weights))
    y_dist = torch.zeros(len(y), len(weights))
    for i in tqdm(range(x_dist.size(0))):
        x_dist[i] = 10 * torch.exp(-(x_hat - x[i]) ** 2 / 5)
        x_dist[i] = F.softmax(x_dist[i])
        y_dist[i] = 10 * torch.exp(-(x_hat - y[i]) ** 2 / 5)
        y_dist[i] = F.softmax(y_dist[i])
    x_p = torch.sum(x_dist,dim=0)/len(x)
    y_p = torch.sum(y_dist, dim=0) / len(y)
    return x_p,y_p

def correlation_coefficient(x,y):
    x_bar = torch.mean(x)
    y_bar = torch.mean(y)
    xy_sum = torch.sum((x-x_bar)*(y-y_bar))
    x_sum = torch.sum(((x-x_bar)**2))
    y_sum = torch.sum(((y-y_bar)**2))
    r = xy_sum / torch.sqrt(x_sum*y_sum)
    return r.item()

def compute_auc_EPE(signed_errors, abs_errors, gt_disp, outdir,suffix,args):

    epe_std = np.std(abs_errors)
    epe_miu = np.mean(abs_errors)
    dist_to_miu = np.abs(abs_errors - epe_miu)
    txt_file = outdir+'statistic.txt'

    inliers = np.ones(dist_to_miu.shape,dtype=np.bool)

    # inliers = (abs_errors<5)

    table0 = PrettyTable()
    table0.title = 'Global statistic'
    table0.field_names = ['name', 'mean', 'std']
    table0.add_row(['signed EPE',np.round(np.mean(signed_errors),4),np.round(np.std(signed_errors),4)])
    table0.add_row(['EPE', np.round(np.mean(abs_errors), 4), np.round(np.std(abs_errors),4)])
    table0.add_row(['D1', np.round(D1(abs_errors,gt_disp), 6), '-'])
    table0.add_row(['EPE(in)', np.round(np.mean(abs_errors[inliers]), 4), np.round(np.std(abs_errors[inliers]), 4)])
    table0.add_row(['D1(in)', np.round(D1(abs_errors[inliers],gt_disp[inliers]), 6), '-'])
    print(table0)

    with open(txt_file,'a') as f:
        f.write(str(table0))
        f.write('\n')
    f.close()

    print('\n')
    #
    table1 = PrettyTable()
    table1.title = 'AUC statistic'
    table1.field_names = ['sort by', 'AUC(EPE)','AUC(D1)','coef']
    #
    table11 = PrettyTable()
    table11.title = 'keypoints statistic'

    print("-> Computing global statistic")

    ##########################################
    #  baseline: sort by error
    ##########################################
    print("-> Sort by error")

    n_key = 20
    keyEPE = [0]
    x_tickers = np.linspace(0, 100, n_key + 1)

    gt_order = abs_errors.argsort()  # small uncerts go first, but big confidence go first
    index = np.linspace(0, abs_errors.shape[0], 21)
    accum_EPE_opt = [0]
    accum_D1_opt = [0]
    samples = [0]
    for idx in tqdm(index[1:], leave=False):
        idx = np.int(idx)
        samples.append(idx)
        current_error = abs_errors[gt_order[:idx]]
        current_gt = gt_disp[gt_order[:idx]]
        keyEPE.append(abs_errors[gt_order[idx - 1]])
        # q1 = np.percentile(current_error,10)
        # q3 = np.percentile(current_error,90)
        # current_error = current_error[current_error > q1]
        # current_error = current_error[current_error < q3]
        accum_EPE_opt.append(np.average(current_error))
        accum_D1_opt.append(D1(current_error,current_gt))
    auc_EPE_opt = np.trapz(accum_EPE_opt, dx=1. / n_key) * 100.
    b3 = accum_D1_opt[-1]
    auc_D1_opt = (b3 + (1 - b3) * np.log(1 - b3)) * 100.
    r = correlation_coefficient_np(abs_errors, abs_errors)
    table1.add_row(['gt_error', auc_EPE_opt, auc_D1_opt, r])
    table11.add_column('%', x_tickers)
    table11.add_column('EPE', keyEPE)

    ########################################
    # create AUC plot
    ########################################
    x_tickers = np.linspace(0, 100, n_key + 1)

    plt.figure()
    plt.plot(x_tickers, accum_EPE_opt, marker="^", color='blue',label='sort by error')
    plt.xticks(x_tickers)
    plt.xlabel('Sample Size(%)')
    plt.ylabel('Accumulative EPE')
    plt.legend()
    fig = plt.gcf()
    fig.set_size_inches(20, 8)
    fig.savefig(os.path.join(outdir, 'Accum_EPE{}.png'.format(suffix)))
    plt.close()

    plt.figure()
    plt.plot(x_tickers, accum_D1_opt, marker="^", color='blue', label='sort by error')
    plt.xticks(x_tickers)
    plt.xlabel('Sample Size(%)')
    plt.ylabel('Accumulative D1')
    plt.legend()
    fig = plt.gcf()
    fig.set_size_inches(20, 8)
    fig.savefig(os.path.join(outdir, 'Accum_D1{}.png'.format(suffix)))
    plt.close()

    print(table1)

    with open(txt_file,'a') as f:
        f.write(str(table1))
        f.write('\n')
    f.close()

    print('\n')

    print(table11)

    print('\n')


def aligned_EPE_and_S(output,args,samples):

    abs_errors = output['EPE']
    s_est = output['s']


    epe_std = torch.std(abs_errors)
    epe_miu = torch.mean(abs_errors)
    dist_to_miu = torch.abs(abs_errors - epe_miu)

    inliers = (dist_to_miu < args.inliers * epe_std)

    epe_std_in = torch.std(abs_errors[inliers])
    epe_miu_in = torch.mean(abs_errors[inliers])

    if args.bin_scale == 'log':
        mark = math.log(6) / math.log(10)
        bins = torch.logspace(0, mark, args.n_bins) - 1
    else:
        bins = torch.linspace(0, 5, args.n_bins)
    print('========> computing distribution ....')
    epe_d, s_d = soft_assignment(abs_errors[samples], s_est[samples], epe_miu_in, epe_std_in, bins)
    # kl_loss = F.kl_div(s_d.log(), epe_d, reduction='sum')

    # txt_file = outdir + 'statistic.txt'
    # with open(txt_file, 'a') as f:
    #     f.write('epe_d: \n')
    #     f.write(str(epe_d) + '\n')
    #     f.write('s_d: \n')
    #     f.write(str(s_d) + '\n')
    #     f.write('kl_loss: {:.4f}'.format(kl_loss))
    #     f.write('\n')
    # f.close()
    # print('sd:')
    # print(s_d)
    # print('epe_d:')
    # print(epe_d)
    # print('kl score: {:.4f}'.format(kl_loss))

    epe_miu = epe_miu.numpy()
    epe_std = epe_std.numpy()
    bins = bins.numpy()
    epe_d = epe_d.numpy()
    s_d = s_d.numpy()

    epe_bins = epe_miu + bins * epe_std

    widths = []
    for i in range(len(bins) - 1):
        widths.append(bins[i + 1] - bins[i])
    widths.append(1 / args.n_bins)
    widths = np.array(widths)


    plt.bar(epe_bins, epe_d, width=widths * epe_std, alpha=0.5, label='EPE',color='b')
    plt.plot(epe_bins, epe_d, linestyle='-',marker='o',color='b')
    plt.bar(epe_bins, s_d, width=widths * epe_std, alpha=0.5, label='Sigma',color='orange')
    plt.plot(epe_bins, s_d, linestyle='-.', marker='^', color='orange')


def process(args):
    key_list = ['disp_est', 'disp_gt']
    if args.uncert:
        key_list.append('uncert_est')
    outputs = {}
    for key in key_list:
        outputs[key] = []

    epoch_idx = 0

    save_ouputs = "{}/test_outputs_{:0>6}.pth".format(args.logdir, epoch_idx)
    outputs = torch.load(save_ouputs, map_location=torch.device('cpu'))
    for key in key_list:
        outputs[key] = torch.cat(outputs[key], dim=0)

    print('========> extracting valid data ....')
    # load in the data file for latter use
    disp_est_raw = outputs['disp_est']
    disp_gt_raw = outputs['disp_gt']
    #

    mask = (disp_gt_raw < args.maxdisp) & (disp_gt_raw > 0)
    #
    disp_est = disp_est_raw[mask].detach()
    disp_gt = disp_gt_raw[mask].detach()

    # s_est = torch.zeros(1)
    signed_errors = disp_gt - disp_est
    abs_errors = torch.abs(signed_errors)

    output = {}
    output['EPE'] = abs_errors
    output['mask'] = mask
    output['gt'] = disp_gt

    del disp_est_raw, disp_gt_raw
    # del disp_est_raw, disp_gt_raw

    if args.uncert:
        s_est_raw = torch.exp(outputs['uncert_est'])
        s_est = s_est_raw[mask].detach()
        del s_est_raw
        output['s'] = s_est

    return output

# def plot_AUC(output,args,net,color,marker):
#
#     abs_errors = output['EPE'].numpy()
#     gt_disp = output['gt'].numpy()
#
#     print(net)
#
#     ##########################################
#     #  baseline: sort by error
#     ##########################################
#     print("-> Sort by error")
#
#     n_key = 20
#     x_tickers = np.linspace(0, 100, n_key + 1)
#
#     gt_order = abs_errors.argsort()  # small uncerts go first, but big confidence go first
#     index = np.linspace(0, abs_errors.shape[0], 21)
#     accum_EPE_opt = [0]
#     for idx in tqdm(index[1:], leave=False):
#         idx = np.int(idx)
#         current_error = abs_errors[gt_order[:idx]]
#         accum_EPE_opt.append(np.average(current_error))
#
#     plt.plot(x_tickers, accum_EPE_opt, linestyle='-', marker=marker, color=color, label=net+'(Opt.)')
#     print(accum_EPE_opt)
#
#     if args.uncert:
#         s_est = output['s'].numpy()
#         ##########################################
#         #  sort by sigma
#         ##########################################
#         print("-> Sort by s")
#
#         sigma_order = s_est.argsort()
#         accum_EPE_s = [0]
#         for idx in tqdm(index[1:], leave=False):
#             idx = np.int(idx)
#             current_error = abs_errors[sigma_order[:idx]]
#             current_gt = gt_disp[sigma_order[:idx]]
#             accum_EPE_s.append(np.average(current_error))
#
#         plt.plot(x_tickers, accum_EPE_s, linestyle='-.', marker=marker, color=color, label=net+'(Est.)')
#         print(accum_EPE_s)
#
#     return

def plot_AUC(AUC_dict,args,net,color,marker):
    print(net)

    n_key = 20
    x_tickers = np.linspace(0, 100, n_key + 1)

    if args.uncert:
        accum_EPE_s = AUC_dict[net]['Est']
        plt.plot(x_tickers, accum_EPE_s, linestyle='-.', marker=marker, color=color, label=net+'(Est.)')
        accum_EPE_opt = AUC_dict[net]['Opt']
        plt.plot(x_tickers, accum_EPE_opt, linestyle='-', marker=marker, color=color, label=net + '(Opt.)')
    else:
        accum_EPE_opt = AUC_dict[net]['Opt']
        plt.plot(x_tickers, accum_EPE_opt, linestyle='-', marker=marker, color=color, label=net + '(Opt.)')


    return


def eval_statistc(outputs,args):
    """Evaluates a pretrained model using a specified test set
    """
    print('========> extracting valid data ....')
    # load in the data file for latter use
    disp_est_raw = outputs['disp_est']
    disp_gt_raw = outputs['disp_gt']
    #

    mask = (disp_gt_raw < args.maxdisp) & (disp_gt_raw > 0)
    #
    disp_est = disp_est_raw[mask].detach()
    disp_gt = disp_gt_raw[mask].detach()
    #
    # s_est = torch.zeros(1)
    signed_errors = disp_gt - disp_est
    abs_errors = torch.abs(signed_errors)

    del disp_est_raw,disp_gt_raw
    # del disp_est_raw, disp_gt_raw

    if args.uncert:
        s_est_raw = torch.exp(outputs['uncert_est'])
        s_est = s_est_raw[mask].detach()
        del s_est_raw

    print("\n-> Done!")

def create_args(logdir,uncert):
    parser = argparse.ArgumentParser(description='Runing Global statistic on the results')
    parser.add_argument('--maxdisp', type=int, default=192, help='max disparity range')
    parser.add_argument('--logdir', type=str,default=logdir, help='the directory to save logs and checkpoints')
    parser.add_argument('--uncert', type=bool, default=uncert, help='estimate uncertainty or not')
    parser.add_argument('--inliers', type=int, default=3, help='how many std to include for inliers')
    parser.add_argument('--mask', type=str, default='soft', help='type of mask assignment',choices=['soft','hard'])
    parser.add_argument('--bin_scale', type=str, default='log', help='how to create the distribution, line or log')
    parser.add_argument('--n_bins', type=int, default=11, help='how many bins to create the distribution')
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    warnings.simplefilter("ignore", UserWarning)

    # ##################################
    # # processing the raw outputs
    # ##################################

    root = "/mnt/Data2/liyan/GwcNet/checkpoints/vkitti2/"

    figdir = '{}/move/Match_error/'.format(root)
    os.makedirs(figdir, exist_ok=True)

    logdir_dict = {
                 # 'GwcNet': ['gwcnet-gc-sml1-lr1e-4/move/',False,'b','o'],
                 #   "L_log": ['gwcnet-gc-elu-dropout-l1-logs-lr1e-4/move/',True,'gray','*'],
                   "L_log (EPE<5)" : ['gwcnet-gc-elu-dropout-l1-logs-inliers-lr1e-4/move/',True,'b','^'],
                   # 'SEDNet (1)': ['gwcnet-gc-elu-dropout-l1-logs-kl-1std-lr1e-4/move/',True,'orange','p'],
                   'SEDNet (3)': ['gwcnet-gc-elu-dropout-l1-logs-kl-3std-lr1e-4/move/',True,'orange','s'],
                   # 'SEDNet (5)':['gwcnet-gc-elu-dropout-l1-logs-kl-5std-logspace-lr1e-4/move/',True,'purple','X']
    }

    AUC_dict = {'GwcNet': {'Opt':[0, 0.0031712924, 0.0063834563, 0.009686767, 0.013129185, 0.016745182, 0.020549804, 0.02454273, 0.028726932, 0.033162817, 0.038040675, 0.043578245, 0.049909994, 0.057237525, 0.06586277, 0.07623062, 0.089025296, 0.105375245, 0.12752575, 0.16213766, 0.42532974]},
                   "L_log": {'Opt':[0, 0.000476998, 0.0011203266, 0.0022011823, 0.0038988185, 0.00614452, 0.008852633, 0.011982508, 0.015532968, 0.019541023, 0.024076624, 0.0292349, 0.035137583, 0.04193947, 0.049844217, 0.05914491, 0.07035261, 0.08465803, 0.10483008, 0.13870881, 0.46177587],
                             'Est':[0, 0.009861632, 0.019600796, 0.030464008, 0.040408775, 0.048688922, 0.05613797, 0.06325532, 0.07015739, 0.07642124, 0.08285309, 0.08901067, 0.095082924, 0.10134157, 0.10789776, 0.114935555, 0.122911364, 0.13300532, 0.16450815, 0.21478693, 0.46177632]},
                   "L_log (EPE<5)" : {'Opt':[0, 0.0004618802, 0.0010785316, 0.002189264, 0.003954489, 0.0062635317, 0.008994535, 0.012062632, 0.015437651, 0.019134827, 0.02321704, 0.027804509, 0.033087526, 0.0392841, 0.046592783, 0.055238076, 0.065613694, 0.07869191, 0.09672398, 0.12588243, 0.42310426],
                             'Est':[0, 0.039230105, 0.039733738, 0.039536513, 0.050089207, 0.059611198, 0.06892533, 0.07754414, 0.0865834, 0.09525113, 0.10361199, 0.11179184, 0.120072246, 0.12890233, 0.13799278, 0.14815703, 0.15970495, 0.17381525, 0.19284634, 0.22390923, 0.42310426]},
                   'SEDNet (1)': {'Opt':[0, 0.000362075, 0.00088199944, 0.0018668689, 0.0034448565, 0.005503799, 0.007969383, 0.010804514, 0.014003181, 0.017597789, 0.021652732, 0.026258193, 0.031532854, 0.03762937, 0.044726998, 0.053072438, 0.06303665, 0.07546749, 0.0925511, 0.12033643, 0.42204928],
                             'Est':[0, 0.013993952, 0.023866475, 0.036439054, 0.04758619, 0.057699103, 0.067007475, 0.07583854, 0.0842171, 0.09212079, 0.1002193, 0.10810217, 0.11614111, 0.12448061, 0.13343179, 0.14330566, 0.15453827, 0.16813168, 0.18636449, 0.21623799, 0.422049]},
                   'SEDNet (3)': {'Opt':[0, 0.00035219366, 0.00085194636, 0.0018305258, 0.0034439166, 0.005497162, 0.0078787105, 0.010570191, 0.013576331, 0.016931407, 0.020704243, 0.024995461, 0.029934544, 0.035672754, 0.042387493, 0.05032859, 0.059909068, 0.07202594, 0.08869609, 0.1158027, 0.3577025],
                             'Est':[0, 0.0069051557, 0.015316448, 0.027420655, 0.037593503, 0.046491105, 0.055012234, 0.062139902, 0.06927812, 0.07582772, 0.08247172, 0.08913747, 0.095861375, 0.10280216, 0.11013204, 0.11811559, 0.12725946, 0.13825488, 0.15250196, 0.17541602, 0.35770315]},
                   'SEDNet (5)':{'Opt':[0, 0.000417343, 0.00094360736, 0.0018876449, 0.0034500794, 0.0055033243, 0.0079315435, 0.010700174, 0.013810642, 0.017294923, 0.021216402, 0.025665775, 0.03075956, 0.036637913, 0.04348316, 0.051562816, 0.061318446, 0.073654436, 0.090715095, 0.11843464, 0.3861524],
                             'Est':[0, 0.013070227, 0.023812352, 0.033491455, 0.041755512, 0.049303845, 0.055395897, 0.06199956, 0.06804872, 0.0739241, 0.0800098, 0.0861673, 0.09245811, 0.09880343, 0.105457, 0.11266701, 0.12085909, 0.13089406, 0.14526245, 0.22617853, 0.3861511]}}




    plt.figure()
    for net in logdir_dict.keys():
        args = create_args(root+logdir_dict[net][0],logdir_dict[net][1])
        # outputs = process(args)
        plot_AUC(AUC_dict,args,net,logdir_dict[net][2],logdir_dict[net][3])

    n_key = 10
    x_tickers = np.linspace(0, 100, n_key + 1)
    plt.xticks(x_tickers,fontsize=15)
    plt.yticks(fontsize=15)
    plt.xlabel('Density(%)',fontsize=20)
    plt.ylabel('Accumulative EPE',fontsize=20)
    plt.legend(ncol=1,fontsize=15)
    fig = plt.gcf()
    fig.set_size_inches(8, 6)
    fig.savefig(os.path.join(figdir, 'Accum_EPE.png'))
    plt.close()

    # samples = torch.randint(0,423761313,(500000,))
    #
    # for net in logdir_dict.keys():
    #     plt.figure()
    #     args = create_args(root+logdir_dict[net][0],logdir_dict[net][1])
    #     outputs = process(args)
    #     aligned_EPE_and_S(outputs,args,samples)
    #     plt.legend(fontsize=15)
    #     plt.ylabel('Ratio of Pixels',fontsize=20)
    #     plt.xlabel('EPE',fontsize=20)
    #     plt.ylim((0,0.3))
    #     plt.xlim((0,20))
    #     fig = plt.gcf()
    #     fig.savefig(figdir + net + '.png')
    #     plt.close()
