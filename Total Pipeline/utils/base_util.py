#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
@Author: Wonjun Lee
@Contact: won5830@gmail.com
@File: util.py
@Time: 2021/08/10 6:35 PM
"""


import os
import glob
import h5py
import numpy as np
import collections
import argparse
import glob
import torch
from scipy import interp
import matplotlib.pyplot as plt
from itertools import cycle
from torch.utils.data import Dataset
import open3d as o3d
from utils.skel_util import knn_contraction
import pickle


def load_ply(partition):
    CUR_DIR = os.path.dirname(os.path.abspath(__file__))
    BASE_DIR = os.path.abspath(os.path.join(CUR_DIR, os.pardir))
    DATA_DIR = os.path.join(BASE_DIR, 'data')
    all_data = []
    for ply_name in glob.glob(os.path.join(DATA_DIR, 'input_ply_data_%s*.ply'%partition)):
        ply = o3d.io.read_point_cloud(ply_name)
        data = np.array(ply.points)
        all_data.append(data) # numpy array in list
    return all_data

def ply2skel(args, ply_list):
    all_skel = []
    for i in range(len(ply_list)):
        tmp_skel = knn_contraction(args, ply_list[i], k = 12, G_k = 16, iter_num = 4, h_repul = 0.005, sigma_thresh = 0.95, fps_pts_num = 20000, mu = 0.7)
        all_skel.append(tmp_skel)
    return all_skel

def skel_partitioner(args, data, partition_num = 17):
    '''
    input: list of skeletons [N, 3] each
    output: 
        output: [N, 2400, 3] pytorch tensor for skeleton segmentation
        norm_info: [N, 4]
        chamber_num: [N] 
    '''
    
    all_data = []
    all_ids = []
    all_norm_info = []

    num_skel = len(data)
    for i in range(len(data)):
        pts_num = len(data[i])
        tmp_pc = data[i][data[i][:, 0].argsort()]
        
        seeds = np.round(np.linspace(0, pts_num - args.num_points, num = partition_num))
        seeds = np.repeat(seeds, args.num_points)
        incre = np.tile(np.arange(args.num_points), partition_num)
        partition_ind = seeds + incre
        partitioned = np.take(tmp_pc, partition_ind.astype('int'), axis = 0).reshape((partition_num, args.num_points, 3)) # [partition_num, 2400, 3]
        #Normalization 
        base = np.min(partitioned, axis=1)
        scale = np.max(partitioned[:,:,0], axis = 1) - np.min(partitioned[:,:,0], axis = 1) # scale along x-axis 
        norm_partitioned = partitioned - np.repeat(np.expand_dims(base, axis=1), args.num_points, axis=1)
        norm_partitioned /= np.broadcast_to(np.resize(scale, (partition_num, 1, 1)), (partition_num, args.num_points, 3))
        norm_info = np.concatenate((base, np.expand_dims(scale, axis=-1)), axis = -1)
        ids = np.ones(partition_num)*i # (partition_num, )
 
        all_norm_info.append(norm_info)
        all_ids.append(ids)
        all_data.append(norm_partitioned)

    all_data = np.concatenate(all_data, axis=0) #(total set number number, point number per each point, 3) = (N, 2400, 3)
    all_norm_info = np.concatenate(all_norm_info, axis=0) #(total set number, 4) = (15*20, 4)
    all_ids = np.concatenate(all_ids, axis=0) #(total set number, 1) = (15*20, 1)
    return all_data, all_norm_info, all_ids

class PointDataset(Dataset):
    def __init__(self, args):
        self.args = args
        if args.from_skel:
            data = load_ply(partition = 'skel')
        else:
            data = load_ply(partition = 'origin')
            data = ply2skel(args, data)
        self.data, self.norm_info, self.ids = skel_partitioner(args, data, partition_num = 20)
    
    def __getitem__(self, item): 
        ids = self.ids[item]
        pointcloud = self.data[item][:self.args.num_points] #(2400, 3) 
        norm_info = self.norm_info[item]
        return pointcloud, ids, norm_info

    def __len__(self):
        return self.data.shape[0] 
        
        


    
    
