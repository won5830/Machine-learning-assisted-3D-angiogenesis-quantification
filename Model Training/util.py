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
import torch.nn as nn
import torch.nn.functional as F
from sklearn.preprocessing import label_binarize
from sklearn.metrics import roc_curve, auc
from scipy import interp
from sklearn.metrics import roc_auc_score
import matplotlib.pyplot as plt
from itertools import cycle
import pickle

class cal_loss_pointnet(nn.Module):
    def __init__(self, device, mat_diff_loss_scale = 0.001):
        super(cal_loss_pointnet,self).__init__()
        # #calculating weight for loss criterion
        BASE_DIR = os.path.dirname(os.path.abspath(__file__))
        DATA_DIR = os.path.join(BASE_DIR, 'data')
        all_label=[]
        for h5_name in glob.glob(os.path.join(DATA_DIR, 'test_skel_ply_hdf5_data_train*.h5')):
            f = h5py.File(h5_name)  
            label = f['label'][:].astype('int64') #(point number, 1) 
            f.close()
            all_label.append(label)
        all_label = np.concatenate(all_label, axis=0) #(total set number, point number per each point) = (15*20, 2400)
        occurences=collections.Counter(all_label.reshape(-1,1).squeeze())
        occ_sort=sorted(occurences.items())
        #weights=torch.tensor([i[1] for i in occ_sort],dtype=torch.float32 )
        weights=torch.tensor([10, 4, 6, 8, 2],dtype=torch.float32 )
        weights = weights / weights.sum() #normalize
        weights = (1.0 / weights) #inversing
        self.weights = (weights/ weights.sum()).to(device) #normalize
        self.mat_diff_loss_scale = mat_diff_loss_scale

    def feature_transform_reguliarzer(self, trans):
        d = trans.size()[1]
        I = torch.eye(d)[None, :, :]
        if trans.is_cuda:
            I = I.cuda()
        loss = torch.mean(torch.norm(torch.bmm(trans, trans.transpose(2, 1)) - I, dim=(1, 2)))
        return loss

    def forward(self, pred, gold, trans_feat, weight_balancing=True, smoothing=True):
        ''' Calculate cross entropy loss, apply label smoothing if needed. '''
        gold = gold.contiguous().view(-1)
        if smoothing:
            eps = 0.15
            n_class = pred.size(1)

            one_hot = torch.zeros_like(pred).scatter(1, gold.view(-1, 1), 1)
            one_hot = one_hot * (1 - eps) + (1 - one_hot) * eps / (n_class - 1)
            log_prb = F.log_softmax(pred, dim=1)
            if weight_balancing:
                wt=one_hot*self.weights
                temp_loss=-(one_hot * log_prb)
                loss=temp_loss.sum()/wt.sum()
            else:
                loss = -(one_hot * log_prb).sum(dim=1).mean()
        else:
            loss = F.cross_entropy(pred, gold, weight=self.weights ,reduction='mean')

        mat_diff_loss = self.feature_transform_reguliarzer(trans_feat)
        loss = loss + mat_diff_loss * self.mat_diff_loss_scale
        return loss

class cal_loss(nn.Module):
    def __init__(self,device):
        super(cal_loss,self).__init__()
        # #calculating weight for loss criterion
        BASE_DIR = os.path.dirname(os.path.abspath(__file__))
        DATA_DIR = os.path.join(BASE_DIR, 'data')
        all_label=[]
        for h5_name in glob.glob(os.path.join(DATA_DIR, 'test_skel_ply_hdf5_data_train*.h5')):
            f = h5py.File(h5_name)  
            label = f['label'][:].astype('int64') #(point number, 1) 
            f.close()
            all_label.append(label)
        all_label = np.concatenate(all_label, axis=0) #(total set number, point number per each point) = (15*20, 2400)
        occurences=collections.Counter(all_label.reshape(-1,1).squeeze())
        occ_sort=sorted(occurences.items())
        #weights=torch.tensor([i[1] for i in occ_sort],dtype=torch.float32 )
        #weights=torch.tensor([30, 1, 5, 10, 0.5],dtype=torch.float32 )
        weights=torch.tensor([15, 2, 5, 10, 5],dtype=torch.float32 ) #[15,1,5,10,1]
        weights = weights / weights.sum() #normalize
        weights = (1.0 / weights) #inversing
        self.weights = (weights/ weights.sum()).to(device) #normalize

    def forward(self, pred, gold, weight_balancing=True, smoothing=True):
        ''' Calculate cross entropy loss, apply label smoothing if needed. '''
        gold = gold.contiguous().view(-1)
        if smoothing:
            eps = 0.15
            n_class = pred.size(1)

            one_hot = torch.zeros_like(pred).scatter(1, gold.view(-1, 1), 1)
            one_hot = one_hot * (1 - eps) + (1 - one_hot) * eps / (n_class - 1)
            log_prb = F.log_softmax(pred, dim=1)
            if weight_balancing:
                wt=one_hot*self.weights
                temp_loss=-(one_hot * log_prb)
                loss=temp_loss.sum()/wt.sum()
            else:
                loss = -(one_hot * log_prb).sum(dim=1).mean()
        else:
            loss = F.cross_entropy(pred, gold, weight=self.weights ,reduction='mean')
        return loss



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

def IOU(true,pred, mean=True): 
    cls_list=sorted(set(true))
    t_unique, t_counts = np.unique(true, return_counts=True)
    p_unique, p_counts = np.unique(pred, return_counts=True)
    t_zip=dict(zip(t_unique,t_counts))
    p_zip=dict(zip(p_unique,p_counts))
    iou=[] 
    for i in cls_list:
        if i not in p_zip:
            p_zip[i]=0
        inner=len([1 for j in range(len(true)) if ((true[j]==i) and (pred[j]==i))])
        outer=t_zip[i]+p_zip[i]-inner
        iou.append(inner/outer)
    if mean==True:
        return sum(iou)/len(iou)
    else:
        return iou


def ROC_cal(args, true, score):
    fpr = dict()
    tpr = dict()
    roc_auc = dict() 
    n_class=len(np.unique(true))
    true_bin = label_binarize(true, classes = [i for i in range(n_class)])
    print(true_bin.shape)
    print(score.shape)
    for i in range(n_class):
        fpr[i], tpr[i], _ = roc_curve(true_bin[:, i], score[:, i]) 
        roc_auc[i] = auc(fpr[i], tpr[i])

    # Compute micro-average ROC curve and ROC area
    fpr["micro"], tpr["micro"], _ = roc_curve(true_bin.ravel(), score.ravel())
    roc_auc["micro"] = auc(fpr["micro"], tpr["micro"]) 
    
    # First aggregate all false positive rates
    all_fpr = np.unique(np.concatenate([fpr[i] for i in range(n_class)]))

    # Then interpolate all ROC curves at this points 
    mean_tpr = np.zeros_like(all_fpr)
    for i in range(n_class):
        mean_tpr += interp(all_fpr, fpr[i], tpr[i])

    # Finally average it and compute AUC
    mean_tpr /= n_class

    fpr["macro"] = all_fpr
    tpr["macro"] = mean_tpr
    roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])

    # Plot all ROC curves
    lw = 1 #line width
    plt.figure(figsize=(9, 6))
    plt.plot(fpr["micro"], tpr["micro"],
             label='micro-average ROC curve (area = {0:0.2f})'
                   ''.format(roc_auc["micro"]),
            color='deeppink', linestyle=':', linewidth=4)

    plt.plot(fpr["macro"], tpr["macro"],
             label='macro-average ROC curve (area = {0:0.2f})'
                   ''.format(roc_auc["macro"]),
            color='navy', linestyle=':', linewidth=4)
    cls_name = ['Root', 'Joint', 'Link', 'Noise', 'Tip Cell']
    colors = cycle(['aqua', 'darkorange', 'cornflowerblue','palegreen','darkmagenta'])
    for i, color in zip(range(n_class), colors):
        plt.plot(fpr[i], tpr[i], color=color, lw=lw,
                 label='class {0} - {1} (area = {2:0.4f})'
                 ''.format(i, cls_name[i], roc_auc[i]))
    
    with open('checkpoints/%s/models/ROC.pickle'%args.exp_name,'wb') as f:
        save_roc = {'fpr': fpr, 'tpr': tpr}
        pickle.dump(save_roc, f, pickle.HIGHEST_PROTOCOL)

    plt.plot([0, 1], [0, 1], 'k--', lw=lw)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('1 - Specificity')
    plt.ylabel('Sensitivity')
    plt.title('ROC - VesselNet')
    plt.legend(loc="lower right")
    plt.show()
    return roc_auc

def Merge_Vote(t_data, t_ids, t_logits, t_true, t_norm_info):
    final_data = []
    final_logits =[]
    final_ids =[]
    final_true =[]

    unq_ids = list(set(t_ids))
    for i in unq_ids:
        tmp_ind = np.where(t_ids == i)
        tmp_ids = t_ids[tmp_ind]
        tmp_logits = t_logits[tmp_ind][:]
        tmp_true = t_true[tmp_ind]
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
            sme_true = tmp_true[sme_idx]
            assert np.all(sme_true == sme_true[0]), "Matching Error!!"
            sme_logits = np.mean(sme_logits, axis=0)
            tmp_logits[sme_idx,:] = sme_logits  #replace tmp_logits

        final_data.append(tmp_datas[unq_ind][:])
        final_logits.append(tmp_logits[unq_ind][:])
        final_ids.append(tmp_ids[unq_ind])
        final_true.append(tmp_true[unq_ind])

    final_data = np.concatenate(final_data, axis=0)
    final_logits = np.concatenate(final_logits, axis=0)
    final_ids = np.concatenate(final_ids, axis=0)
    final_true = np.concatenate(final_true, axis=0)
    return final_data, final_logits, final_ids, final_true


def Perturb_remove(tmp_data, tmp_pred, tmp_true, tmp_ids):
    tmp_data = np.round(tmp_data*100)/100
    unq_data, unq_ind =np.unique(tmp_data, return_index=True, axis=0)
    return unq_data, tmp_pred[unq_ind], tmp_true[unq_ind], tmp_ids[unq_ind]
    
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv2d') != -1:
        nn.init.xavier_normal_(m.weight.data)
        #nn.init.constant_(m.bias.data, 0.0)
    elif classname.find('Linear') != -1:
        nn.init.xavier_normal_(m.weight.data)
        #nn.init.constant_(m.bias.data, 0.0)
    elif classname.find('Conv1d') != -1:
        nn.init.xavier_normal_(m.weight.data)
        #nn.init.constant_(m.bias.data, 0.0)