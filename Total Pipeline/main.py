
from __future__ import print_function
#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
@Modified By: Wonjun Lee
@Contact: won5830@gmail.com
@File: main.py
@Time: 2021/08/10 6:35 PM
"""

import os
import collections
import argparse
import glob
import torch
import torch.nn as nn
from model import PointNet, DGCNN, PointNet2, ADGCNN, SPHADGCNN
import numpy as np
from torch.utils.data import DataLoader
from utils import seg_util as segutil
from utils import skel_util as skutil
from utils import base_util as butil
from utils import eval_util as eutil
import pickle

def _init_():
    if not os.path.exists('checkpoints'):
        os.makedirs('checkpoints')
    if not os.path.exists('data'):
        os.makedirs('data')
    if not os.path.exists('checkpoints/'+args.exp_name):
        os.makedirs('checkpoints/'+args.exp_name)
    os.system('cp main.py checkpoints'+'/'+args.exp_name+'/'+'main.py.backup')
    os.system('cp model.py checkpoints' + '/' + args.exp_name + '/' + 'model.py.backup')



def skel_segment(args, io):
    # Data pre-processing
    data_loader = DataLoader(butil.PointDataset(args), num_workers=4,
                             batch_size=args.batch_size, shuffle=False, drop_last=False)
    device = torch.device("cuda" if args.cuda else "cpu")
    #Try to load models
    if args.model == 'pointnet':
        model = PointNet(part_num=5, normal_channel=False).to(device)
    elif args.model == 'dgcnn':
        model = DGCNN(args).to(device)
    elif args.model == 'pointnet2':
        model = PointNet2(num_classes=5, normal_channel=False).to(device)
    elif args.model == 'adgcnn':
        model = ADGCNN(args).to(device)
    elif args.model == 'sphadgcnn':
        model = SPHADGCNN(args).to(device)
    else:
        raise Exception("Not implemented")
    print(str(model))
    
    model = nn.DataParallel(model)
    model.load_state_dict(torch.load(args.model_path))
    model = model.eval()

    eval_datas = []
    eval_logits = []
    eval_ids = []
    eval_norm_info = []
    for data, ids, norm_info in data_loader:
        data, ids, norm_info = data.to(device).float(), ids.to(device).squeeze(), norm_info.to(device)
        eval_datas.append(data.view(-1,3).cpu().detach().numpy())
        data = data.permute(0, 2, 1) #(batch_size, num_points, 3) -> (batch_size, 3, num_points)
        batch_size = data.size()[0]
        if (args.model == 'pointnet') or (args.model == 'pointnet2'):
            logits, trans_feat = model(data)
        else:
            logits = model(data)
        logits = logits.permute(0,2,1).contiguous() # (batch_size, seg_num_all, num_points) -> (batch_size, num_points, seg_num_all)
        eval_logits.append(logits.view(-1,5).cpu().detach().numpy())
        ids = ids.repeat_interleave(args.num_points)
        eval_ids.append(ids.cpu().numpy().reshape(-1))
        norm_info = torch.repeat_interleave(norm_info, args.num_points, dim=0)
        eval_norm_info.append(norm_info.cpu().detach().numpy())
    
    eval_datas = np.concatenate(eval_datas, axis=0).squeeze() #(total_batch_size*num_points, 3)
    eval_ids = np.concatenate(eval_ids) #(total_batch_size*num_points): 1-D array
    eval_logits = np.concatenate(eval_logits, axis=0).squeeze() #(total_batch_size*num_points, seg_num_all)
    eval_norm_info = np.concatenate(eval_norm_info, axis=0) #(tot_batch_size*num_points, 4) 
    final_data, final_logits, final_ids = segutil.Merge_Vote(eval_datas, eval_ids, eval_logits, eval_norm_info)

    final_pred = final_logits.argmax(axis=1)
    
    unq = list(set(final_ids))
    labeled_data = {} 
    for ind, val in enumerate(unq):
        tmp_ind = np.where(final_ids == val)
        tmp_ids = final_ids[tmp_ind]
        tmp_pred = final_pred[tmp_ind]
        tmp_data = final_data[tmp_ind][:] 
        tmp_data, tmp_pred, tmp_ids = segutil.Perturb_remove(tmp_data, tmp_pred, tmp_ids)
        labeled_data[ind]={'ids':tmp_ids, 'pred':tmp_pred, 'point': tmp_data}
    
    with open('checkpoints/{}/test_labeled.pickle'.format(args.exp_name),'wb') as f:
        pickle.dump(labeled_data,f, pickle.HIGHEST_PROTOCOL)
    
    return labeled_data

if __name__ == "__main__":
    # Training settings
    parser = argparse.ArgumentParser(description='Angiogenesis Quantification')
    parser.add_argument('--exp_name', type=str, default='exp', metavar='N',
                        help='Name of the experiment')
    parser.add_argument('--from_skel', type=segutil.str2bool, nargs='?',const=True, default=True,
                        help='skeletonize the input with temporary knn-contraction (deprecated)')
    parser.add_argument('--model', type=str, default='dgcnn', metavar='N',
                        choices=['pointnet', 'dgcnn','pointnet2', 'adgcnn', 'sphadgcnn'],
                        help='Model to use, [pointnet, dgcnn, pointnet2, adgcnn, sphadgcnn]')
    parser.add_argument('--model_path', type=str, default='', metavar='N',
                        help='Pretrained model path')
    parser.add_argument('--batch_size', type=int, default = 4, metavar='batch_size',
                        help='Size of batch)')
    parser.add_argument('--no_cuda', type=segutil.str2bool, nargs='?',const=True, default=False,
                        help='enables CUDA training')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--num_points', type=int, default=2400,
                        help='num of points to use')
    parser.add_argument('--k', type=int, default=20, metavar='N',
                        help='Num of nearest neighbors to use')
    parser.add_argument('--dropout', type=float, default=0.5,
                        help='dropout rate')
    parser.add_argument('--emb_dims', type=int, default=1024, metavar='N',
                        help='Dimension of embeddings')
    args = parser.parse_args()

    _init_()
    
    io = segutil.IOStream('checkpoints/' + args.exp_name + '/run.log')
    io.cprint(str(args))

    args.cuda = not args.no_cuda and torch.cuda.is_available()
    torch.manual_seed(args.seed)
    if args.cuda:
        io.cprint(
            'Using GPU : ' + str(torch.cuda.current_device()) + ' from ' + str(torch.cuda.device_count()) + ' devices')
        torch.cuda.manual_seed(args.seed)
    else:
        io.cprint('Using CPU')

    # SKeleton Extraction & Segmentation 
    labeled_data = skel_segment(args, io)
    
    # Angiogenesis Analysis 
    eutil.VesselEval(labeled_data, args.exp_name)
