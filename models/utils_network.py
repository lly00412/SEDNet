import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
import torch.utils.model_zoo as model_zoo
from models.utils_dropout import DropCluster4D, Nodrop
from datasets import __datasets__
from torch.utils.data import DataLoader, random_split
from models.submodule import *
from collections import OrderedDict
from torch.hub import load_state_dict_from_url
import os
import math
from torch.utils.data.sampler import WeightedRandomSampler
from torch import Tensor
from typing import Type, Any, Callable, Union, List, Optional

def set_dropout(args,model):
    if isinstance(model,torch.nn.DataParallel):
        model = model.module
    '''
    Set up the network dropout layer. Only set up dropout layer after the first 2 convolutional layers.
    :param args: dropout method
    :param model: network model
    :return: network with dropout layer
    '''
    if args.dropout == 'dropout':
        model.dropout1 = nn.Dropout(p=args.drop_prob)
        model.dropout2 = nn.Dropout(p=args.drop_prob)
    elif args.dropout == 'dropcluster':
        model.dropout1 = DropCluster4D(drop_prob=args.drop_prob)
        model.dropout2 = DropCluster4D(drop_prob=args.drop_prob)
        set_transform_matrices(args, model)
    return model

def sample_data(args):
    StereoDataset = __datasets__[args.dataset]
    train_dataset = StereoDataset(args.datapath, args.trainlist, True, args)
    n_val = len(train_dataset ) - 100
    train_set, val_set = random_split(train_dataset, [100, n_val], generator=torch.Generator().manual_seed(0))
    trainloader = torch.utils.data.DataLoader(train_set, batch_size=args.batch_size, pin_memory=True, num_workers=4)
    return trainloader

def sample_cost(args,model,dataloader):
    matrix_file = "{}/sample_cost.pth".format(args.maskdir)
    if os.path.exists(matrix_file):
        sample_costs = torch.load(matrix_file, map_location=torch.device('cpu'))
    else:
        model.eval()
        total_batch = 0
        for batch_idx, sample in enumerate(dataloader):
            imgL, imgR, disp_gt = sample['left'], sample['right'], sample['disparity']
            imgL = imgL.cuda()
            imgR = imgR.cuda()

            features_left = model.feature_extraction(imgL)
            features_right = model.feature_extraction(imgR)

            gwc_volume = build_gwc_volume(features_left["gwc_feature"], features_right["gwc_feature"], args.maxdisp // 4,
                                          model.num_groups)
            if model.use_concat_volume:
                concat_volume = build_concat_volume(features_left["concat_feature"], features_right["concat_feature"],
                                                    args.maxdisp // 4)
                volume = torch.cat((gwc_volume, concat_volume), 1)
            else:
                volume = gwc_volume

            cost0 = model.dres0(volume)  # (8, 32, 48, 64, 128)
            cost0 = cost0.detach()
            if not os.path.exists(args.maskdir):
                os.makedirs(args.maskdir)
            torch.save(cost0, "{}/sample_cost_{:0>6}.pth".format(args.maskdir, batch_idx))
            total_batch += 1

        sample_costs = []
        for batch_idx in range(total_batch):
            save_costs = "{}/sample_cost_{:0>6}.pth".format(args.maskdir, batch_idx)
            sample_costs.append(torch.load(save_costs, map_location=torch.device('cpu')))
        sample_costs = torch.cat(sample_costs, dim=0)

        torch.save(sample_costs, matrix_file)

    return sample_costs

def compute_clusters(args,model,sample_costs):
    _, transform_matrices = model.dropout1.compute_transform_matrices_kMeans(sample_costs,args.n_cluster)

    torch.save(transform_matrices ,"{}/transform_matrices.pth".format(args.maskdir))
    print('Saving the transform matrices...')
    return transform_matrices

def set_transform_matrices(args,model):
    if args.compute_cluster:
        trainloader = sample_data(args)
        sample_costs = sample_cost(args,model,trainloader)
        clusters = compute_clusters(args,model,sample_costs)
    else:
        clusters = torch.load("{}/transform_matrices.pth".format(args.maskdir), map_location=torch.device('cpu'))
    model.dropout1.transform_matrices = clusters
    model.dropout2.transform_matrices = clusters