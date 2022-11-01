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
from utils.pointnet2 import pointnet2_utils as pn2_utils
import pickle



def knn(x, y, k):
    '''
    Input: [Batch, channel, n_points]
    Output: [Batch, n_points of x, k]
        Search nearest neighborhoods from y (source: x) 
    '''
    inner = -2*torch.matmul(x.transpose(2, 1), y)
    xx = torch.sum(x**2, dim=1, keepdim=True)
    yy = torch.sum(y**2, dim=1, keepdim=True)
    pair_sum = xx + yy.transpose(2,1)
    pairwise_distance = - pair_sum.transpose(2,1) - inner
    
    idx = pairwise_distance.topk(k=k, dim=-1)[1]   # (batch_size, num_points, k)
    return idx


def get_graph_feature(x, y, k=20, return_idx=True):
    # Taking x as the source point cloud  
    batch_size, _, num_points = x.size()
    _, num_dims, y_num_points = y.size()
    
    x = x.view(batch_size, -1, num_points)

    idx = knn(x, y, k=k)   # (batch_size, num_points, k)
    device = x.device
    
    idx_base = torch.arange(0, batch_size, device=device).view(-1, 1, 1)*y_num_points
    idx = idx + idx_base # 
    idx = idx.view(-1) 
    x = x.transpose(2, 1).contiguous()
    y = y.transpose(2, 1).contiguous()   # (batch_size, num_points, num_dims)  
    #it was originaly view 
    feature = y.view(batch_size*y_num_points, -1)[idx, :] # -> (batch_size*num_points, num_dims) 

    feature = feature.view(batch_size, num_points, k, num_dims) # -> batch_size * num_points * k + range(0, batch_size*num_points)
    x = x.view(batch_size, num_points, 1, num_dims).repeat(1, 1, k, 1)
    #feature = torch.cat((feature-x, x), dim=3).permute(0, 3, 1, 2).contiguous() # [B, C, N, k]
    feature = (feature-x).permute(0, 3, 1, 2).contiguous() # [B, C, N, k]
    if return_idx:
        return feature, idx
    else:
        return feature


def farthest_point_sample(xyz, npoint):
    """
    Input:
        xyz: pointcloud data, [B, N, 3]
        npoint: number of samples
    Return:
        centroids: sampled pointcloud index, [B, npoint]
    """
    device = xyz.device
    B, N, C = xyz.shape 
    centroids = torch.zeros(B, npoint, dtype=torch.long).to(device)
    distance = torch.ones(B, N).to(device) * 1e10
    farthest = torch.randint(0, N, (B,), dtype=torch.long).to(device)
    batch_indices = torch.arange(B, dtype=torch.long).to(device)
    for i in range(npoint):
        centroids[:, i] = farthest
        centroid = xyz[batch_indices, farthest, :].view(B, 1, 3)
        dist = torch.sum((xyz - centroid) ** 2, -1)
        mask = dist < distance
        distance[mask] = dist[mask]
        farthest = torch.max(distance, -1)[1]
    return centroids

# Functions for repulsion regularization 
def theta(mat, h):
    tmp = torch.exp(-mat**2/((h/4)**2))
    return tmp

def repulsion_regularization(tmp_pc, h=0.04, eta = 0.45):
    # Device Formating 
    device = tmp_pc.device
    # Pre Value calculating
    point_num = tmp_pc.size(2)
    point_set = tmp_pc.cpu().squeeze().transpose(1,0).contiguous()
    R_v =  point_set.repeat(1,point_num) - point_set.reshape(1,-1).repeat(point_num, 1)
    realR_v = R_v.reshape(-1,3).norm(dim = -1).reshape(point_num, -1)
    v = torch.sum(realR_v, dim = -1) # Distance Summation 
    w = torch.sum(theta(realR_v, h), dim = -1)
    weighting = w.view(1,-1).repeat(point_num, 1)
    beta = -theta(realR_v, h)*weighting/realR_v
    beta = beta.fill_diagonal_(0)
    sum_beta = torch.sum(beta, dim = -1)

    # Final Repulsion summation 
    beta = beta.repeat_interleave(3, dim = -1)
    sum_beta = sum_beta.view(-1,1).contiguous() # For row_wise division
    to_be_added = eta*R_v*beta/sum_beta 
    to_be_added = to_be_added.reshape(point_num, point_num, 3) # [n_points, n_points, 3]
    to_be_added = torch.sum(to_be_added, dim = 1) # [n_points, 3]
    to_be_added = to_be_added.unsqueeze(0).transpose(2,1).contiguous() # # [1, 3, n_points]
    
    return to_be_added.to(device)

def WLOP_L1_regularization(tmp_pc, original_pc, h=0.004):
    eps = 0.0001
    # Device Formating 
    device = tmp_pc.device
    # Pre Value calculating
    point_num = tmp_pc.size(2)
    original_num = original_pc.size(2)
    # Preprocessing
    point_set = tmp_pc.cpu().squeeze().transpose(1,0).contiguous() #[point_num, 3]
    original_point_set = original_pc.cpu().squeeze().transpose(1,0).contiguous() #[ori_points, 3]
    # Value Calculating 
    R_v =  point_set.repeat(1, original_num) - original_point_set.reshape(1,-1).repeat(point_num, 1)
    realR_v = R_v.reshape(-1,3).norm(dim = -1).reshape(point_num, -1)
    realR_v[realR_v==0] = eps
    v = torch.sum(realR_v, dim = -1) # Distance Summation 
    v = v.view(-1,1).contiguous()
    
    alpha = theta(realR_v, h)/realR_v 
    alpha = alpha/v
    
    sum_alpha = torch.sum(alpha, dim = -1)
    
    # Final L1 regularization
    alpha = alpha.repeat_interleave(3, dim = -1)
    sum_alpha = sum_alpha.view(-1,1).contiguous() # For row_wise division
    to_be_added = R_v*alpha/sum_alpha 
    to_be_added = to_be_added.reshape(point_num, original_num, 3) # [n_points, n_points, 3]
    to_be_added = torch.sum(to_be_added, dim = 1) # [n_points, 3]
    to_be_added = to_be_added.unsqueeze(0).transpose(2,1).contiguous() # # [1, 3, n_points]
    return to_be_added.to(device)

def CurveRegularization(skel_pc, search_knn = 10, curve_thresh = 0.92):
    '''
    Input: skel_pc [1, 3, N]
    Output: new_skel_pc [1,3,N]
    ''' 
    # Eigen Decomposition
    tmp_G_knn = get_graph_feature(skel_pc, skel_pc, k = search_knn, return_idx = False).squeeze() #[3, N, k]
    tmp_proj = tmp_G_knn.permute(1,0,2).contiguous() #[N, 3, k]
    
    tmp_outer = tmp_G_knn.transpose(1,0).contiguous() #[N, 3, k]
    tmp_outer = torch.einsum('bik,bjk->bijk', tmp_outer, tmp_outer) #[8192, 3, 3, 8]
    tmp_outer = tmp_outer.sum(dim = -1) #[8192, 3, 3]
    
    tmp_eival, tmp_eivec = torch.linalg.eig(tmp_outer)
    tmp_prin_eivec = tmp_eivec[:,:,0].real #[8192, 3] 
    tmp_sig = tmp_eival[:,0]/tmp_eival.sum(-1)
    tmp_sig = tmp_sig.real

    # Projection 
    tmp_front_dist = torch.sum(tmp_prin_eivec.unsqueeze(dim = 1).expand(-1, search_knn, -1)*tmp_proj.transpose(2,1).contiguous(), dim=-1, keepdim = False)  #[N, k, 3] -> [N, k]
    tmp_back_dist = tmp_front_dist.clone()
    tmp_front_dist[tmp_front_dist <= 0] = 1000000. # [N, k] take min
    tmp_back_dist[tmp_back_dist >= 0] = -1000000. # [N, k] take max
    
    front_nearest_ind = torch.argmin(tmp_front_dist, dim=1, keepdim=True).unsqueeze(-1).expand(-1,3,-1) # [N,3,1] 
    back_nearest_ind = torch.argmax(tmp_back_dist, dim=1, keepdim=True).unsqueeze(-1).expand(-1,3, -1) # [N, 3,1] 
    
    front_nearest_diff = torch.gather(tmp_proj, -1, front_nearest_ind).squeeze() # [N,3]
    back_nearest_diff = torch.gather(tmp_proj, -1, back_nearest_ind).squeeze() # [N,3]
    diff_x = (front_nearest_diff + back_nearest_diff)/2 # [N,3]
    is_not_curve = tmp_sig < curve_thresh
    diff_x[is_not_curve, :] = 0
    
    return skel_pc + diff_x.transpose(1,0).unsqueeze(dim=0).contiguous()


def knn_contraction(args, input, k = 12, G_k = 16, iter_num = 4, h_repul = 0.005, sigma_thresh = 0.95, fps_pts_num = 20000, mu = 0.7):
    '''
    Knn contraction base meso-skeleton extraction algorithm (Temporary implementation for redistribution)
    Strongly deprecated due to the absence of further processing and softening modules. 
    
    input: [n_points, 3] numpy array
    output: [fps_pts_num, 3] numpy array
    '''
    if args.cuda:
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')
    
    torch.cuda.empty_cache()

    with torch.no_grad():
    
        # Load Point Cloud
        original_pc = torch.tensor(input, dtype = torch.float, device = device).unsqueeze(dim = 0).transpose(2,1).contiguous() # [B, 3, N]
    
        # Normalize
        min_rng, _ = torch.min(original_pc, dim = 2, keepdims= True)
        max_rng, _ = torch.max(original_pc, dim = 2, keepdims = True)
        scale = max_rng - min_rng
        original_pc = (original_pc - min_rng) / scale[0, 0, 0]
    
        # Farthest Point Sampling (Downsampling)
        tmp_pc = original_pc.transpose(2,1).contiguous() # -> [B, N, 3]
        fp_ind = farthest_point_sample(tmp_pc, npoint = fps_pts_num) #[B, N, 3]
        tmp_pc = tmp_pc.transpose(2,1).contiguous() # -> [B, 3, N]
        tmp_pc = pn2_utils.gather_operation(tmp_pc, fp_ind.type(torch.int32)) # [B, 3, N]

        tmp_fps = tmp_pc.clone().detach()
    
        # Generation of initial convergence mask 
        is_converged = torch.zeros(fps_pts_num, device = device).bool()

        for iter in range(iter_num):
            tmp_u = get_graph_feature(tmp_pc, tmp_pc, k = k, return_idx = False).squeeze() #[3, 8192, 8]
            ###### WEIGHTING #######
            #weight_u_nom = theta(tmp_u, h = 70)
            #weight_u_denom = torch.sum(weight_u_nom, dim = -1, keepdims = True)
            #weight_u = weight_u_nom/weight_u_denom
            #tmp_u = torch.sum(weight_u*tmp_u, dim = -1)
            ########################
            tmp_u = torch.sum(tmp_u/k, dim = -1)
            tmp_u = tmp_u.transpose(1,0).contiguous()

            # Eigen Decomposition
            tmp_G_knn = get_graph_feature(tmp_pc, tmp_pc, k = G_k, return_idx = False).squeeze()

            tmp_outer = tmp_G_knn.transpose(1,0).contiguous()
            tmp_outer = torch.einsum('bik,bjk->bijk', tmp_outer, tmp_outer) #[8192, 3, 3, 8]
            tmp_outer = tmp_outer.sum(dim = -1)

            tmp_eival, tmp_eivec = torch.linalg.eig(tmp_outer)
            tmp_prin_eivec = tmp_eivec[:,:,0].real
            tmp_sig = tmp_eival[:,0]/tmp_eival.sum(-1)
            tmp_sig = tmp_sig.real

            # delta_x calculation 
            norm_u = torch.norm(tmp_u, dim = -1).unsqueeze(dim = -1) #[8196]
            proj_u2v = torch.cos(torch.sum(tmp_u * tmp_prin_eivec, dim=-1, keepdim = True)) #[8196]
            proj_u2v = proj_u2v*torch.norm(tmp_u, dim = -1, keepdim= True)
            sub_sig = (1- tmp_sig).unsqueeze(dim = -1) #[8196] 
            tmp_is_converged = tmp_sig > sigma_thresh
            is_converged = is_converged  + tmp_is_converged
        
            # WLOP
            repul_x = repulsion_regularization(tmp_pc, h = h_repul, eta = 0.45)
            repul_x = repul_x.squeeze().transpose(1,0).contiguous() #[N,3]
            repul_x[is_converged, :] = 0
            repul_x = repul_x.unsqueeze(dim = 0).transpose(2,1).contiguous()
            #L1_x = WLOP_L1_regularization(tmp_pc, original_pc, h = h)

            # Point Cloud Update
            del_x = norm_u*proj_u2v*sub_sig*tmp_prin_eivec + (tmp_u - norm_u*proj_u2v*tmp_prin_eivec)
            del_x[is_converged, :] = 0 
            del_x = del_x.unsqueeze(dim = 0).transpose(2,1).contiguous()
        
            tmp_pc = tmp_pc + del_x  + mu*repul_x 
        
        
            # NaN Filtering
            nonan_ind = ~torch.any(tmp_pc.isnan(), dim=1).squeeze()
            tmp_pc = tmp_pc[:, :, nonan_ind].contiguous()
            is_converged = is_converged[nonan_ind].contiguous()

            # Curve regularization
            tmp_pc = CurveRegularization(tmp_pc, search_knn = 10, curve_thresh = 0.9)

            # Final PC
            final_pc = tmp_pc.squeeze().transpose(1,0).contiguous()
    
            # Update k & G_k 
            if (iter>0) and (iter%8 ==0):
                k = k*2
            if (iter>0) and (iter % 20 ==0):
                G_k = 2*G_k
        
        # Denormalization
        final_pc = final_pc*scale[0,0,0] + min_rng.squeeze(dim = -1)
        final_pc = final_pc.cpu().detach().numpy()
        return final_pc



    
