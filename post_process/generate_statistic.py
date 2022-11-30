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

def train_linear_regression(X,Y,outdir,filename):
    X = X.reshape(-1,1)
    Y = Y.reshape(-1,1)
    x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size = 0.2)
    model = LinearRegression()
    model.fit(x_train, y_train)

    print('Weight: {}  Bias: {}'.format(model.coef_,model.intercept_))
    predictions = model.predict(x_test)

    loss = np.mean((y_test-predictions)**2)

    plt.clf()
    plt.plot(x_train, y_train, 'go', label='Train data', alpha=0.5,markersize=1)
    plt.plot(x_test, y_test, 'ro', label='Test data', alpha=0.5, markersize=1)
    plt.plot(x_test, predictions, '--', label='Predictions', linewidth= 3, alpha=1)
    plt.legend(loc='best')
    plt.title(filename)
    fig = plt.gcf()
    fig.savefig(os.path.join(outdir, '{}.png'.format(filename)))
    plt.show()

    return model.coef_\
        ,model.intercept_,loss

def liner_regression(X,Y):
    X = X.reshape(-1, 1)
    Y = Y.reshape(-1, 1)
    model = LinearRegression()
    model.fit(X,Y)
    predictions = model.predict(X)
    return model.coef_, model.intercept_, predictions

def Normal(x,mu,sigma):
    return torch.exp(-(x-mu)**2/(2*sigma**2))/(torch.sqrt(2*torch.tensor(np.pi))*sigma)

def Uniform(x,a,b):
    return torch.ones(x.size())*(1/(b-a))

def Laplace(x,mu,beta):
    return torch.exp(-torch.abs(x-mu)/beta)/(2*beta)

def EM_for_Gussian_and_Uniform(x,mu=0,sigma=1,alpha=0.5,epsilon=0.00001,maxdisp=96,iter_num=1000):
    # p(e) = w*N(mu,sigma**2)+(1-w)*U(a,b)  N is inlier, U is outlie

    N = x.size(0)

    # init

    Mu = mu
    Sigma = sigma
    Alpha = alpha
    MAXDISP = maxdisp

    i = 0

    while (i<iter_num):

        # Expectation

        gauss = Normal(x, Mu, Sigma)
        # uniform = Uniform(x,-MAXDISP,MAXDISP)
        uniform = Uniform(x, 0, MAXDISP)
        M = Alpha*gauss + (1-Alpha)*uniform

        # Maximization

        # Update

        old_sigma = Sigma
        old_mu = Mu

        SigmaSqure = torch.dot((Alpha*gauss / M), (x - Mu) ** 2) / torch.sum(Alpha*gauss/ M)
        Mu = torch.dot((Alpha*gauss / M), x) / torch.sum(Alpha*gauss/M)
        Alpha = torch.sum(Alpha*gauss/ M) / N

        Sigma = np.sqrt(SigmaSqure.item())
        Mu = Mu.item()
        Alpha = Alpha.item()

        print('iteration {}: Sigma: {}  Mu: {}  Alpha: {}'.format(i,Sigma,Mu,Alpha))

        if abs(Mu-old_mu)<=epsilon and abs(Sigma-old_sigma)<=epsilon:
            break
        i += 1

    gauss = Normal(x, Mu, Sigma)
    # uniform = Uniform(x, -MAXDISP, MAXDISP)
    uniform = Uniform(x, 0, MAXDISP)

    assignment = (gauss > uniform)

    return Mu, Sigma, Alpha, assignment

def EM_for_Laplace_and_Uniform(x,mu=0,beta=1,alpha=0.5,epsilon=0.001,maxdisp=96,iter_num=100):
    # p(e) = w*L(mu,b)+(1-w)*U(a,b)  N is inlier, U is outlie

    N = x.size(0)

    # init

    Mu = mu
    Beta = beta
    Alpha = alpha
    MAXDISP = maxdisp

    i = 0

    while (i<iter_num):

        # Expectation

        laplace = Laplace(x,Mu,Beta)
        uniform = Uniform(x, -MAXDISP, MAXDISP)
        M = Alpha*laplace + (1-Alpha)*uniform

        # Maximization

        # Update

        old_beta = Beta
        old_mu = Mu

        P = Alpha*laplace/M
        Beta = torch.dot(P,torch.abs(x-Mu)) / torch.sum(P)

        n_inv = x/torch.abs(x-Mu)
        n_idx = 1- torch.isnan(n_inv)
        d_inv = 1 / torch.abs(x - Mu)
        d_idx = 1 - torch.isnan(d_inv)
        idx = n_idx*d_idx

        Mu = torch.sum(n_inv[idx]*P[idx]) / torch.sum(d_inv[idx]*P[idx])
        Alpha = torch.sum(P) / N

        Beta = Beta.item()
        Mu = Mu.item()
        Alpha = Alpha.item()

        print('iteration {}: Beta: {}  Mu: {}  Alpha: {}'.format(i,Beta,Mu,Alpha))

        if abs(Mu-old_mu)<=epsilon and abs(Beta-old_beta)<=epsilon:
            break
        i += 1

    laplace = Laplace(x,Mu,Beta)
    # uniform = Uniform(x, -MAXDISP, MAXDISP)
    uniform = Uniform(x, 0, MAXDISP)

    assignment = (laplace > uniform)

    return Mu, Beta, Alpha, assignment

def align_distribution(abs_errors, s_est, samples,keyEPE,keyS, offset,outdir):
    # X_train = np.array(keyS[1:])
    # Y_train = np.array(keyEPE[1:])
    #
    # X_train = X_train.reshape(-1,1)
    # Y_train = Y_train.reshape(-1,1)
    #
    # model = LinearRegression()
    # model.fit(X_train,Y_train)
    #
    # W,b = model.coef_, model.intercept_

    idx0 = np.where(samples == offset[0])[0][0]
    idx1 = np.where(samples == offset[1])[0][0]

    x0 = keyS[idx0]
    x1 = keyS[idx1]
    y0 = keyEPE[idx0]
    y1 = keyEPE[idx1]

    W = (y0-y1)/(x0-x1)
    b = y0 - W*x0

    s_est_aligned = W*s_est+b

    W = np.round(W,4)
    b = np.round(b,4)

    plt.figure()
    bins = np.linspace(0,1,100)
    plt.hist(abs_errors,bins,alpha=0.5,label='EPE')
    plt.hist(s_est_aligned,bins,alpha=0.5,label='{}*S+{}'.format(W,b))
    plt.legend()
    fig = plt.gcf()
    fig.savefig(outdir + 'aligned_EPE_s_estimation_distribution_{}_and_{}.png'.format(offset[0],offset[1]))
    plt.close()

def compute_auc_EPE_S(signed_errors, abs_errors, s_est, gt_disp, outdir,suffix,args):

    epe_std = np.std(abs_errors)
    epe_miu = np.mean(abs_errors)
    dist_to_miu = np.abs(abs_errors - epe_miu)
    txt_file = outdir+'statistic.txt'

    if args.inliers > 0:
        if args.mask in ['soft']:
            epe_std = np.std(abs_errors)
            epe_miu = np.mean(abs_errors)
            dist_to_miu = np.abs(abs_errors - epe_miu)
            inliers = (dist_to_miu < args.inliers * epe_std)
            with open(txt_file, 'a') as f:
                f.write('inliers def: epe< mu+ {}sigma'.format(args.inliers))
                f.write('\n')
            f.close()
        else:
            inliers = (abs_errors < args.inliers)
            with open(txt_file, 'a') as f:
                f.write('inliers def: epe< {}'.format(args.inliers))
                f.write('\n')
            f.close()
        pct = abs_errors[inliers].shape[0]/abs_errors.shape[0]
        with open(txt_file, 'a') as f:
            f.write('inliers pct: {:.4f}'.format(pct))
            f.write('\n')
        f.close()
    else:
        inliers = np.ones(dist_to_miu.shape,dtype=np.bool)

    # inliers = (abs_errors<5)

    #
    # plt.figure()
    # plt.hist(signed_errors, bins=500, range=(-5, 5))
    # fig = plt.gcf()
    # fig.savefig(outdir + 'signed_errors_distribution{}.png'.format(suffix))
    # plt.close()
    #
    # plt.figure()
    # plt.hist(abs_errors, bins=500, range=(0, 5))
    # fig = plt.gcf()
    # fig.savefig(outdir + 'abs_errors_distribution{}.png'.format(suffix))
    # plt.close()
    #
    # plt.figure()
    # plt.hist(s_est, bins=500,range=(0, 5))
    # fig = plt.gcf()
    # fig.savefig(outdir + 's_est_distribution{}.png'.format(suffix))
    # plt.close()
    #

    order = abs_errors.argsort()
    idx = int(0.95*abs_errors.shape[0])
    match_errors = np.abs(abs_errors-s_est)

    # table0 = PrettyTable()
    # table0.title = 'Global statistic'
    # table0.field_names = ['name', 'mean', 'std']
    # table0.add_row(['signed EPE',np.round(np.mean(signed_errors),4),np.round(np.std(signed_errors),4)])
    # table0.add_row(['EPE', np.round(np.mean(abs_errors), 4), np.round(np.std(abs_errors),4)])
    # table0.add_row(['S', np.round(np.mean(s_est), 4), np.round(np.std(s_est),4)])
    # table0.add_row(['D1', np.round(D1(abs_errors,gt_disp), 6), '-'])
    # table0.add_row(['|EPE-S|', np.round(np.mean(match_errors), 4), np.round(np.std(match_errors),4)])
    # table0.add_row(['|EPE-S|(95%)', np.round(np.mean(match_errors[order[:idx]]), 4), np.round(np.std(match_errors[order[:idx]]), 4)])
    # table0.add_row(['EPE(in)', np.round(np.mean(abs_errors[inliers]), 4), np.round(np.std(abs_errors[inliers]), 4)])
    # table0.add_row(['S(in)', np.round(np.mean(s_est[inliers]), 4), np.round(np.std(s_est[inliers]), 4)])
    # table0.add_row(['D1(in)', np.round(D1(abs_errors[inliers],gt_disp[inliers]), 6), '-'])
    # print(table0)

    table0 = PrettyTable()
    table0.title = 'Global statistic'
    table0.field_names = ['name', 'mean', 'std']
    table0.add_row(['EPE', np.round(np.mean(abs_errors), 4), np.round(np.std(abs_errors),4)])
    table0.add_row(['S', np.round(np.mean(s_est), 4), np.round(np.std(s_est),4)])
    table0.add_row(['D1', np.round(D1(abs_errors,gt_disp), 6), '-'])
    table0.add_row(['|EPE-S|', np.round(np.mean(match_errors), 4), np.round(np.std(match_errors),4)])
    table0.add_row(['|EPE-S|(mid)', np.round(np.median(match_errors[order[:idx]]), 4), '-'])
    print(table0)

    with open(txt_file,'a') as f:
        f.write(str(table0))
        f.write('\n')
    f.close()



    #
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

    ##########################################
    #  sort by sigma
    ##########################################
    print("-> Sort by s")

    sigma_order = s_est.argsort()
    # order = cost_conf.argsort()[::-1]

    accum_EPE_s = [0]
    accum_D1_s = [0]
    samples = [0]
    keyS = [0]
    for idx in tqdm(index[1:], leave=False):
        idx = np.int(idx)
        samples.append(idx)
        current_error = abs_errors[sigma_order[:idx]]
        current_gt = gt_disp[sigma_order[:idx]]
        keyS.append(s_est[sigma_order[idx - 1]])
        accum_EPE_s.append(np.average(current_error))
        accum_D1_s.append(D1(current_error,current_gt))
    auc_EPE_s = np.trapz(accum_EPE_s, dx=1. / n_key) * 100.
    auc_D1_s = np.trapz(accum_D1_s, dx=1. / n_key) * 100.
    r = correlation_coefficient_np(s_est, abs_errors)
    table1.add_row(['s', auc_EPE_s,auc_D1_s,r])
    table11.add_column('S', keyS)

    # ##########################################
    # #  baseline: sort by error(inliers)
    # ##########################################
    # print("-> Sort by error (inliers)")
    #
    # abs_errors_in = abs_errors[inliers]
    # gt_order = abs_errors_in.argsort()  # small uncerts go first, but big confidence go first
    # index = np.linspace(0, abs_errors_in.shape[0], n_key+1)
    # accum_EPE_opt_in = [0]
    # accum_D1_opt_in = [0]
    # samples = [0]
    # keyEPEin = [0]
    # for idx in tqdm(index[1:], leave=False):
    #     idx = np.int(idx)
    #     samples.append(idx)
    #     current_error = abs_errors_in[gt_order[:idx]]
    #     keyEPEin.append(abs_errors_in[gt_order[idx - 1]])
    #     # q1 = np.percentile(current_error,10)
    #     # q3 = np.percentile(current_error,90)
    #     # current_error = current_error[current_error > q1]
    #     # current_error = current_error[current_error < q3]
    #     accum_EPE_opt_in.append(np.average(current_error))
    #     accum_D1_opt_in.append(D1(current_error))
    # auc_EPE_opt_in = np.trapz(accum_EPE_opt_in, dx=1. / n_key) * 100.
    # b3_in = accum_D1_opt_in[-1]
    # auc_D1_opt_in = (b3_in + (1 - b3_in) * np.log(1 - b3_in)) * 100.
    # r = correlation_coefficient_np(abs_errors_in, abs_errors_in)
    # table1.add_row(['gt_error(inliners)', auc_EPE_opt_in,auc_D1_opt_in, r])
    # table11.add_column('EPE(inliers)', keyEPEin)
    #
    # ##########################################
    # #  sort by sigma(inliers)
    # ##########################################
    # print("-> Sort by s(inliers)")
    #
    # s_est_in = s_est[inliers]
    # sigma_order = s_est_in.argsort()
    # # order = cost_conf.argsort()[::-1]
    #
    # accum_EPE_s_in = [0]
    # accum_D1_s_in = [0]
    # samples = [0]
    # keySin=[0]
    # for idx in tqdm(index[1:], leave=False):
    #     idx = np.int(idx)
    #     samples.append(idx)
    #     current_error = abs_errors_in[sigma_order[:idx]]
    #     current_s = s_est_in[sigma_order[:idx]]
    #     keySin.append(s_est_in[sigma_order[idx - 1]])
    #     accum_EPE_s_in.append(np.average(current_error))
    #     accum_D1_s_in.append(D1(current_error))
    # auc_EPE_s_in = np.trapz(accum_EPE_s_in, dx=1. / n_key) * 100.
    # auc_D1_s_in = np.trapz(accum_D1_s_in, dx=1. / n_key) * 100.
    # r = correlation_coefficient_np(s_est_in, abs_errors_in)
    # table1.add_row(['s(inliers)', auc_EPE_s_in,auc_D1_s_in, r])
    # table11.add_column('S(inliers)', keySin)

    #
    ########################################
    # create AUC plot
    ########################################
    x_tickers = np.linspace(0, 100, n_key + 1)

    plt.figure()
    plt.plot(x_tickers, accum_EPE_opt, marker="^", color='blue',label='sort by error')
    plt.plot(x_tickers, accum_EPE_s, marker="o", color='orange',label='sort by s')
    # plt.plot(x_tickers, accum_EPE_opt_in, marker="^", linestyle='--', color='blue', label='sort by error(inliers)')
    # plt.plot(x_tickers, accum_EPE_s_in, marker="o", linestyle='--', color='orange', label='sort by s(inliers)')
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
    plt.plot(x_tickers, accum_D1_s, marker="o", color='orange', label='sort by s')
    # plt.plot(x_tickers, accum_D1_opt_in, marker="^", linestyle='--', color='blue', label='sort by error(inliers)')
    # plt.plot(x_tickers, accum_D1_s_in, marker="o", linestyle='--', color='orange', label='sort by s(inliers)')
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

    with open(txt_file,'a') as f:
        f.write(str(table11))
        f.write('\n')
    f.close()

    # table2 = PrettyTable()
    # table2.title = 'S vs. Prediction'
    # sigma = s_est
    # table2.field_names = ['Thres', '% of px', 'EPE']
    # N = [1, 2, 3, 4, 8, 16, 32, 64]
    # table2.add_row(['all', np.round(np.size(abs_errors)/np.size(abs_errors),4)*100, np.round(np.mean(abs_errors),4)])
    # for n in N:
    #     group = abs_errors[abs_errors <= n * sigma]
    #     table2.add_row(['<= {}s'.format(n), np.round(np.size(group)/np.size(abs_errors),4)*100, np.round(np.mean(group),4)])
    # print(table2)
    #
    # with open(txt_file,'a') as f:
    #     f.write(str(table2))
    #     f.write('\n')
    # f.close()
    #
    # print('\n')
    #
    # table3 = PrettyTable()
    # table3.title = 'Precision statistic'
    # table3.field_names = ['Thres', '% of px', 'abs_error_avg', 'abs_error_std', 's_avg', 's_std']
    # table3.add_row(['all',
    #                 np.round(np.size(abs_errors) / np.size(abs_errors), 4) * 100,
    #                 np.round(np.mean(abs_errors), 4),
    #                 np.round(np.std(abs_errors), 4),
    #                 np.round(np.mean(s_est), 4),
    #                 np.round(np.std(s_est), 4)])
    # thres = [1, 3, 5, 7, 9]
    # for i in range(len(thres)):
    #     group_mask = (dist_to_miu < thres[i] * epe_std)
    #     table3.add_row([' |epe - epe_miu| < {} epe_std'.format(thres[i]),
    #                     np.round(np.size(abs_errors[group_mask]) / np.size(abs_errors), 4) * 100,
    #                     np.round(np.mean(abs_errors[group_mask]), 4),
    #                     np.round(np.std(abs_errors[group_mask]), 4),
    #                     np.round(np.mean(s_est[group_mask]), 4),
    #                     np.round(np.std(s_est[group_mask]), 4)])
    # print(table3)
    # with open(txt_file, 'a') as f:
    #     f.write(str(table3))
    #     f.write('\n')
    # f.close()

    print('\n')

def compute_auc_EPE_conf(signed_errors, abs_errors, conf_est, gt_disp,outdir,suffix,args):

    txt_file = outdir+'statistic.txt'

    table0 = PrettyTable()
    table0.title = 'Global statistic'
    table0.field_names = ['name', 'mean', 'std']
    table0.add_row(['signed_error',np.round(np.mean(signed_errors),4),np.round(np.std(signed_errors),4)])
    table0.add_row(['abs_error', np.round(np.mean(abs_errors), 4), np.round(np.std(abs_errors),4)])
    table0.add_row(['conf_est', np.round(np.mean(conf_est), 4), np.round(np.std(conf_est),4)])
    table0.add_row(['D1', np.round(D1(abs_errors,gt_disp), 6), '-'])

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
    #
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
        accum_D1_opt.append(D1(current_error, current_gt))
    auc_EPE_opt = np.trapz(accum_EPE_opt, dx=1. / n_key) * 100.
    b3 = accum_D1_opt[-1]
    auc_D1_opt = (b3 + (1 - b3) * np.log(1 - b3)) * 100.
    r = correlation_coefficient_np(abs_errors, abs_errors)
    table1.add_row(['gt_error', auc_EPE_opt, auc_D1_opt, r])
    table11.add_column('%', x_tickers)
    table11.add_column('EPE', keyEPE)

    ##########################################
    #  sort by conf
    ##########################################
    print("-> Sort by conf")

    conf_est = -conf_est
    conf_order = conf_est.argsort()
    # order = cost_conf.argsort()[::-1]

    accum_EPE_conf = [0]
    accum_D1_conf = [0]
    samples = [0]
    keyConf = [0]
    for idx in tqdm(index[1:], leave=False):
        idx = np.int(idx)
        samples.append(idx)
        current_error = abs_errors[conf_order[:idx]]
        current_gt = gt_disp[conf_order[:idx]]
        keyConf.append(conf_est[conf_order[idx - 1]])
        accum_EPE_conf.append(np.average(current_error))
        accum_D1_opt.append(D1(current_error, current_gt))
    auc_EPE_conf = np.trapz(accum_EPE_conf, dx=1./n_key)*100.
    auc_D1_conf = np.trapz(accum_D1_conf, dx=1./n_key)*100.
    r = correlation_coefficient_np(conf_est, abs_errors)
    table1.add_row(['conf', auc_EPE_conf,auc_D1_conf,r])
    table11.add_column('Conf', keyConf)

    #
    ########################################
    # create AUC plot
    ########################################
    x_tickers = np.linspace(0, 100, n_key + 1)

    plt.figure()
    plt.plot(x_tickers, accum_EPE_opt, marker="^", color='blue',label='sort by error')
    plt.plot(x_tickers, accum_EPE_conf, marker="o", color='orange',label='sort by conf')
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
    plt.plot(x_tickers, accum_D1_conf, marker="o", color='orange', label='sort by conf')
    plt.xticks(x_tickers)
    plt.xlabel('Sample Size(%)')
    plt.ylabel('Accumulative D1(%)')
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

def compute_auc_EPE(signed_errors, abs_errors, gt_disp, outdir,suffix,args):

    epe_std = np.std(abs_errors)
    epe_miu = np.mean(abs_errors)
    dist_to_miu = np.abs(abs_errors - epe_miu)
    txt_file = outdir+'statistic.txt'

    inliers = np.ones(dist_to_miu.shape,dtype=np.bool)

    # inliers = (abs_errors<5)

    #
    # plt.figure()
    # plt.hist(signed_errors, bins=500, range=(-5, 5))
    # fig = plt.gcf()
    # fig.savefig(outdir + 'signed_errors_distribution{}.png'.format(suffix))
    # plt.close()
    #
    # plt.figure()
    # plt.hist(abs_errors, bins=500, range=(0, 5))
    # fig = plt.gcf()
    # fig.savefig(outdir + 'abs_errors_distribution{}.png'.format(suffix))
    # plt.close()
    #
    # plt.figure()
    # plt.hist(s_est, bins=500,range=(0, 5))
    # fig = plt.gcf()
    # fig.savefig(outdir + 's_est_distribution{}.png'.format(suffix))
    # plt.close()

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


def compute_buckets(datas,names,n_buckets,outdir,suffix='_with_0'):
    n_samples = datas[0].shape[0]
    plt.figure()
    width = 0.8 / len(datas)
    x = np.arange(1, n_buckets + 1)

    table = PrettyTable()
    table.title = 'Bucket statistic'
    # table.field_names = ['group', 'avg_abs_error', 'std_abs_errors','avg_s','std_s']
    groupnames = ['G{}'.format(i) for i in range(n_buckets)]
    table.add_column('group',groupnames)

    selections = np.linspace(0, n_samples, n_buckets+1,dtype=int)
    for j in range(len(datas)):
        data = datas[j]
        avg = []
        std = []
        N = []
        for i in range(n_buckets):
            current = data[selections[i]:selections[i+1]]
            avg.append(np.average(current))
            std.append(np.std(current))
            N.append(selections[i+1]-selections[i])
        plt.bar(x-0.4+j*width, height=avg,width=width,label=names[j])
        if j == 0:
            table.add_column('n_samples', N)
        table.add_column('avg_{}'.format(names[j]),np.round(avg,4))
        table.add_column('std_{}'.format(names[j]), np.round(std, 4))

    plt.xticks(x,groupnames)
    plt.legend()
    fig = plt.gcf()
    fig.set_size_inches(15, 8)
    fig.savefig(outdir+'avg_over_buckets{}.png'.format(suffix))

    print(table)

def aligned_EPE_and_S(abs_errors,s_est,outdir):
    epe_std = torch.std(abs_errors)
    epe_miu = torch.mean(abs_errors)
    dist_to_miu = torch.abs(abs_errors - epe_miu)

    if args.inliers > 0:
        inliers = inliers = (dist_to_miu < args.inliers * epe_std)
    else:
        inliers = torch.ones(dist_to_miu.size(), dtype=torch.long)

    epe_std_in = torch.std(abs_errors[inliers])
    epe_miu_in = torch.mean(abs_errors[inliers])

    if args.bin_scale == 'log':
        mark = math.log(6) / math.log(10)
        bins = torch.logspace(0, mark, args.n_bins) - 1
    else:
        bins = torch.linspace(0, 5, args.n_bins)
    print('========> computing distribution ....')
    epe_d, s_d = soft_assignment(abs_errors[inliers][:6000000], s_est[inliers][:6000000], epe_miu_in, epe_std_in, bins)
    kl_loss = F.kl_div(s_d.log(), epe_d, reduction='sum')

    txt_file = outdir + 'statistic.txt'
    with open(txt_file, 'a') as f:
        f.write('epe_d: \n')
        f.write(str(epe_d) + '\n')
        f.write('s_d: \n')
        f.write(str(s_d) + '\n')
        f.write('kl_loss: {:.4f}'.format(kl_loss))
        f.write('\n')
    f.close()
    print('sd:')
    print(s_d)
    print('epe_d:')
    print(epe_d)
    print('kl score: {:.4f}'.format(kl_loss))

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

    plt.figure()
    plt.title('Soft Assigment for EPE and S (inliers)')
    plt.bar(epe_bins, epe_d, width=widths * epe_std, alpha=0.5, label='EPE')
    plt.bar(epe_bins, s_d, width=widths * epe_std, alpha=0.5, label='S')
    plt.legend()
    fig = plt.gcf()
    fig.savefig(outdir + 'aligned_EPE_s_estimation_soft_assigned_inliers.png')
    plt.close()


def eval_statistc(outputs,outdir,args):
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

        compute_auc_EPE_S(signed_errors.numpy(), abs_errors.numpy(), s_est.numpy(), disp_gt.numpy(), outdir, '',args)
    elif args.conf:
        conf_est_raw = outputs['conf_est']
        conf_est = conf_est_raw[mask].detach()
        del conf_est_raw
        compute_auc_EPE_conf(signed_errors.numpy(), abs_errors.numpy(), conf_est.numpy(), disp_gt.numpy(), outdir, '',args)
    else:
        compute_auc_EPE(signed_errors.numpy(), abs_errors.numpy(),disp_gt.numpy(),  outdir, '', args)

    print("\n-> Done!")

if __name__ == "__main__":
    warnings.simplefilter("ignore", UserWarning)
    parser = argparse.ArgumentParser(description='Runing Global statistic on the results')
    parser.add_argument('--epochs', type=int, required=True, help='number of epochs to train')
    parser.add_argument('--maxdisp', type=int, default=192, help='max disparity range')
    parser.add_argument('--logdir', required=True, help='the directory to save logs and checkpoints')
    parser.add_argument('--inliers', type=int, default=3, help='how many std to include for inliers')
    parser.add_argument('--mask', type=str, default='soft', help='type of mask assignment',choices=['soft','hard'])
    parser.add_argument('--bin_scale', type=str, default='line', help='how to create the distribution, line or log')
    parser.add_argument('--n_bins', type=int, default=11, help='how many bins to create the distribution')
    parser.add_argument('--uncert', action='store_true', help='estimate uncertainty or not')
    parser.add_argument('--conf', action='store_true', help='estimate conf or not')
    parser.add_argument('--dataset', required=True, help='dataset name')
    # parse arguments, set seeds
    args = parser.parse_args()

    # ##################################
    # # processing the raw outputs
    # ##################################
    logdir = args.logdir
    epoch_idx = args.epochs
    # key_list = ['disp_est','disp_gt','conf_est']
    key_list = ['disp_est', 'disp_gt']
    if args.uncert:
        key_list.append('uncert_est')
    if args.conf:
        key_list.append('conf_est')
    outputs = {}
    for key in key_list:
        outputs[key] = []

    save_ouputs = "{}/test_outputs_{:0>6}.pth".format(logdir, epoch_idx)
    outputs = torch.load(save_ouputs, map_location=torch.device('cpu'))
    for key in key_list:
        outputs[key] = torch.cat(outputs[key], dim=0)

    if args.dataset in ['drivingstereo']:
        top_pad = 80
        right_pad = 79
        for key in key_list:
            outputs[key] = outputs[key][:,top_pad:,:-right_pad]

    if args.dataset in ['apolloscape']:
        fg_mask_list = './filenames/apolloscape_fg_mask.txt'
        datapath = "/mnt/Data2/datasets/apolloscape/"
        lines = read_all_lines(fg_mask_list)
        splits = [line.split() for line in lines]
        fg_mask_images = [x[0] for x in splits]
        fg_masks = []
        for index in range(len(fg_mask_images)):
            fg_mask = load_mask(datapath+fg_mask_images[index])
            fg_mask = ndi.zoom(fg_mask,zoom=0.25)
            fg_mask[fg_mask>0] = 1
            fg_mask = np.expand_dims(fg_mask,axis=0)
            fg_masks.append(fg_mask)
        fg_masks = np.concatenate(fg_masks)
        outputs['fg_mask'] = torch.from_numpy(fg_masks.astype('float32'))

        top_pad = 256 - 240
        right_pad = 800 - 782
        for key in key_list:
            outputs[key] = outputs[key][:, top_pad:, :-right_pad]*outputs['fg_mask']

    # only for LAF
    # outputs['disp_est'] = outputs['disp_est']*args.maxdisp
    # outputs['disp_gt'] = outputs['disp_gt'] * args.maxdisp
    #
    ##################################
    # loading full test results
    ##################################
    figdir = '{}/statistic/test/'.format(logdir)
    os.makedirs(figdir, exist_ok=True)
    eval_statistc(outputs,figdir,args)
