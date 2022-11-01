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
import matplotlib.pyplot as plt
from itertools import cycle
import pickle



class IOStream():
    def __init__(self, path):
        self.f = open(path, 'a')

    def cprint(self, text):
        print(text)
        self.f.write(text+'\n')
        self.f.flush()

    def close(self):
        self.f.close()


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def Merge_Vote(t_data, t_ids, t_logits, t_norm_info):
    final_data = []
    final_logits =[]
    final_ids =[]

    unq_ids = list(set(t_ids))
    for i in unq_ids:
        tmp_ind = np.where(t_ids == i)
        tmp_ids = t_ids[tmp_ind]
        tmp_logits = t_logits[tmp_ind][:]
        tmp_datas = t_data[tmp_ind][:] 
        tmp_norm_info = t_norm_info[tmp_ind][:]

        #denormalization 
        tmp_datas = tmp_datas*tmp_norm_info[:,3][:, None]
        tmp_datas = tmp_datas + tmp_norm_info[:,:3]
        tmp_datas = np.round(tmp_datas*1000)/1000
        
        unq_data, unq_ind, unq_cnt = np.unique(tmp_datas, return_index=True, return_counts = True, axis =0)
        multi_tp_ind = np.where(unq_cnt>1)
        multi_data = unq_data[multi_tp_ind][:]
        multi_ind = unq_ind[multi_tp_ind]

        for i in list(multi_tp_ind[0]):
            sme_idx = np.where((tmp_datas == unq_data[i][:]).all(axis=1))[0]
            sme_logits = tmp_logits[sme_idx,:]
            sme_logits = np.mean(sme_logits, axis=0)
            tmp_logits[sme_idx,:] = sme_logits  #replace tmp_logits

        final_data.append(tmp_datas[unq_ind][:])
        final_logits.append(tmp_logits[unq_ind][:])
        final_ids.append(tmp_ids[unq_ind])
    final_data = np.concatenate(final_data, axis=0)
    final_logits = np.concatenate(final_logits, axis=0)
    final_ids = np.concatenate(final_ids, axis=0)
    return final_data, final_logits, final_ids


def Perturb_remove(tmp_data, tmp_pred, tmp_ids):
    tmp_data = np.round(tmp_data*100)/100
    unq_data, unq_ind =np.unique(tmp_data, return_index=True, axis=0)
    return unq_data, tmp_pred[unq_ind], tmp_ids[unq_ind]
    
