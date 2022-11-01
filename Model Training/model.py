#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
@Modified By: Wonjun Lee
@Contact: won5830@gmail.com
@File: model.py
@Time: 2021/08/10 6:35 PM
"""

import os
import sys
import copy
import math
import numpy as np
import torch
from torch.autograd import Variable
import torch.nn.init as init
import torch.nn as nn
import torch.nn.functional as F
from pointnet2_utils import PointNetSetAbstractionMsg,PointNetSetAbstraction,PointNetFeaturePropagation


def knn(x, k):
    inner = -2*torch.matmul(x.transpose(2, 1), x)
    xx = torch.sum(x**2, dim=1, keepdim=True)
    pairwise_distance = -xx - inner - xx.transpose(2, 1)
 
    idx = pairwise_distance.topk(k=k, dim=-1)[1]   # (batch_size, num_points, k)
    return idx

def dknn(x, k, d):
    inner = -2*torch.matmul(x.transpose(2, 1), x)
    xx = torch.sum(x**2, dim=1, keepdim=True)
    pairwise_distance = -xx - inner - xx.transpose(2, 1)
    
    idx = pairwise_distance.topk(k=k*d, dim=-1)[1]   # (batch_size, num_points, k*d)
    
    selector = torch.range(start = 0, end = k-1, dtype = torch.int64)*d
    
    selector = selector.expand(idx.size(0), idx.size(1), k) #k
    
    idx = torch.gather(idx, -1, selector) # (batch_size, num_points, k) 
    
    return idx


def get_graph_feature(x, k=20, idx=None):
    batch_size = x.size(0)
    num_points = x.size(2)
    x = x.view(batch_size, -1, num_points)
    if idx is None:
        idx = knn(x, k=k)   # (batch_size, num_points, k)
    device = torch.device('cuda')

    idx_base = torch.arange(0, batch_size, device=device).view(-1, 1, 1)*num_points

    idx = idx + idx_base

    idx = idx.view(-1)
 
    _, num_dims, _ = x.size()

    x = x.transpose(2, 1).contiguous()   # (batch_size, num_points, num_dims)  -> (batch_size*num_points, num_dims) #   batch_size * num_points * k + range(0, batch_size*num_points)
    feature = x.view(batch_size*num_points, -1)[idx, :]
    feature = feature.view(batch_size, num_points, k, num_dims) 
    x = x.view(batch_size, num_points, 1, num_dims).repeat(1, 1, k, 1)
    
    feature = torch.cat((feature-x, x), dim=3).permute(0, 3, 1, 2).contiguous()
    return feature


class Transform_Net(nn.Module):
    def __init__(self, args):
        super(Transform_Net, self).__init__()
        self.args = args
        self.k = 3
        self.bn1 = nn.BatchNorm2d(64)
        self.bn2 = nn.BatchNorm2d(128)
        self.bn3 = nn.BatchNorm1d(1024)
        self.conv1 = nn.Sequential(nn.Conv2d(6, 64, kernel_size=1, bias=False),
                                   self.bn1,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv2 = nn.Sequential(nn.Conv2d(64, 128, kernel_size=1, bias=False),
                                   self.bn2,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv3 = nn.Sequential(nn.Conv1d(128, 1024, kernel_size=1, bias=False),
                                   self.bn3,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.linear1 = nn.Linear(1024, 512, bias=False)
        self.bn3 = nn.BatchNorm1d(512) 
        self.linear2 = nn.Linear(512, 256, bias=False)
        self.bn4 = nn.BatchNorm1d(256) 

        self.transform = nn.Linear(256, 3*3)
        init.constant_(self.transform.weight, 0)
        init.eye_(self.transform.bias.view(3, 3))

    def forward(self, x):
        batch_size = x.size(0)

        x = self.conv1(x)                       # (batch_size, 3*2, num_points, k) -> (batch_size, 64, num_points, k)
        x = self.conv2(x)                       # (batch_size, 64, num_points, k) -> (batch_size, 128, num_points, k)
        x = x.max(dim=-1, keepdim=False)[0]     # (batch_size, 128, num_points, k) -> (batch_size, 128, num_points)

        x = self.conv3(x)                       # (batch_size, 128, num_points) -> (batch_size, 1024, num_points)
        x = x.max(dim=-1, keepdim=False)[0]     # (batch_size, 1024, num_points) -> (batch_size, 1024)

        x = F.leaky_relu(self.bn3(self.linear1(x)), negative_slope=0.2)     # (batch_size, 1024) -> (batch_size, 512)
        x = F.leaky_relu(self.bn4(self.linear2(x)), negative_slope=0.2)     # (batch_size, 512) -> (batch_size, 256)

        x = self.transform(x)                   # (batch_size, 256) -> (batch_size, 3*3)
        x = x.view(batch_size, 3, 3)            # (batch_size, 3*3) -> (batch_size, 3, 3)
        return x 


class STN3d(nn.Module):
    def __init__(self, channel):
        super(STN3d, self).__init__()
        self.conv1 = torch.nn.Conv1d(channel, 64, 1)
        self.conv2 = torch.nn.Conv1d(64, 128, 1)
        self.conv3 = torch.nn.Conv1d(128, 1024, 1)
        self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 9)
        self.relu = nn.ReLU()

        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(1024)
        self.bn4 = nn.BatchNorm1d(512)
        self.bn5 = nn.BatchNorm1d(256)

    def forward(self, x):
        batchsize = x.size()[0]
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = torch.max(x, 2, keepdim=True)[0]
        x = x.view(-1, 1024)

        x = F.relu(self.bn4(self.fc1(x))) #
        x = F.relu(self.bn5(self.fc2(x)))
        x = self.fc3(x)

        iden = Variable(torch.from_numpy(np.array([1, 0, 0, 0, 1, 0, 0, 0, 1]).astype(np.float32))).view(1, 9).repeat(
            batchsize, 1)
        if x.is_cuda:
            iden = iden.cuda()
        x = x + iden
        x = x.view(-1, 3, 3)
        return x


class STNkd(nn.Module):
    def __init__(self, k=64):
        super(STNkd, self).__init__()
        self.conv1 = torch.nn.Conv1d(k, 64, 1)
        self.conv2 = torch.nn.Conv1d(64, 128, 1)
        self.conv3 = torch.nn.Conv1d(128, 1024, 1)
        self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, k * k)
        self.relu = nn.ReLU()

        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(1024)
        self.bn4 = nn.BatchNorm1d(512)
        self.bn5 = nn.BatchNorm1d(256)

        self.k = k

    def forward(self, x):
        batchsize = x.size()[0]
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = torch.max(x, 2, keepdim=True)[0]
        x = x.view(-1, 1024)

        x = F.relu(self.bn4(self.fc1(x)))
        x = F.relu(self.bn5(self.fc2(x)))
        x = self.fc3(x)

        iden = Variable(torch.from_numpy(np.eye(self.k).flatten().astype(np.float32))).view(1, self.k * self.k).repeat(
            batchsize, 1)
        if x.is_cuda:
            iden = iden.cuda()
        x = x + iden
        x = x.view(-1, self.k, self.k)
        return x


class PointNet(nn.Module):
    def __init__(self, part_num=5, normal_channel=False):
        super(PointNet, self).__init__()
        if normal_channel:
            channel = 6
        else:
            channel = 3 
        self.part_num = part_num
        self.stn = STN3d(channel)
        self.conv1 = torch.nn.Conv1d(channel, 64, 1)
        self.conv2 = torch.nn.Conv1d(64, 128, 1)
        self.conv3 = torch.nn.Conv1d(128, 128, 1)
        self.conv4 = torch.nn.Conv1d(128, 512, 1)
        self.conv5 = torch.nn.Conv1d(512, 2048, 1)
        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(128)
        self.bn4 = nn.BatchNorm1d(512)
        self.bn5 = nn.BatchNorm1d(2048)
        self.fstn = STNkd(k=128)
        self.convs1 = torch.nn.Conv1d(4928, 256, 1)
        self.convs2 = torch.nn.Conv1d(256, 256, 1)
        self.convs3 = torch.nn.Conv1d(256, 128, 1)
        self.convs4 = torch.nn.Conv1d(128, part_num, 1)
        self.bns1 = nn.BatchNorm1d(256)
        self.bns2 = nn.BatchNorm1d(256)
        self.bns3 = nn.BatchNorm1d(128)

    def forward(self, point_cloud):
        B, D, N = point_cloud.size()
        trans = self.stn(point_cloud)
        point_cloud = point_cloud.transpose(2, 1)
        if D > 3:
            point_cloud, feature = point_cloud.split(3, dim=2)
        point_cloud = torch.bmm(point_cloud, trans)
        if D > 3:
            point_cloud = torch.cat([point_cloud, feature], dim=2)

        point_cloud = point_cloud.transpose(2, 1)

        out1 = F.relu(self.bn1(self.conv1(point_cloud))) # (batch_size, 3, num_points) -> (batch_size, 64, num_points)
        out2 = F.relu(self.bn2(self.conv2(out1))) # (batch_size, 64, num_points) -> (batch_size, 128, num_points)
        out3 = F.relu(self.bn3(self.conv3(out2))) # (batch_size, 128, num_points) -> (batch_size, 128, num_points)

        trans_feat = self.fstn(out3) # trans_feat = (batch, 128, 128)
        x = out3.transpose(2, 1)
        net_transformed = torch.bmm(x, trans_feat)
        net_transformed = net_transformed.transpose(2, 1) # (batch_size, 128, num_points) -> (batch_size, 128, num_points)

        out4 = F.relu(self.bn4(self.conv4(net_transformed))) # (batch_size, 128, num_points) -> (batch_size, 512, num_points)
        out5 = self.bn5(self.conv5(out4)) # (batch_size, 512, num_points) -> (batch_size, 2048, num_points)
        out_max = torch.max(out5, 2, keepdim=True)[0] # (batch_size, 2048, num_points) -> (batch_size, 2048, 1)
        out_max = out_max.view(-1, 2048) # (batch_size, 2048, 1) -> (batch_size, 2048)

        #out_max = torch.cat([out_max,label.squeeze(1)],1)
        expand = out_max.view(-1, 2048, 1).repeat(1, 1, N) # (batch_size, 2048) -> (batch_size, 2048, num_points)
        concat = torch.cat([expand, out1, out2, out3, out4, out5], 1) # (batch_size, 2048, num_points) -> (batch_size, 4928, num_points)
        net = F.relu(self.bns1(self.convs1(concat))) # (batch_size, 2048, num_points) -> (batch_size, 256, num_points)
        net = F.relu(self.bns2(self.convs2(net))) # (batch_size, 256, num_points) -> (batch_size, 256, num_points)
        net = F.relu(self.bns3(self.convs3(net))) # (batch_size, 256, num_points) -> (batch_size, 128, num_points)
        net = self.convs4(net) # (batch_size, 128, num_points) -> (batch_size, 5, num_points)
        net = net.transpose(2, 1).contiguous() # (batch_size, 5, num_points) -> (batch_size, num_points, 5)
        #net = F.log_softmax(net.view(-1, 5), dim=-1) #apply softmax to each segmentation vector of 5-dim
        #net = net.view(B, N, self.part_num) # [B, N, 5]
        net = net.transpose(2, 1) # (batch_size, num_points, 5) -> (batch_size, 5, num_points)
        return net, trans_feat


class PointNet2(nn.Module):
    def __init__(self, num_classes, normal_channel=False):
        super(PointNet2, self).__init__()
        if normal_channel:
            additional_channel = 3
        else:
            additional_channel = 0
        self.normal_channel = normal_channel
        self.sa1 = PointNetSetAbstractionMsg(1024, [0.05, 0.1, 0.2], [32, 64, 128], 3+additional_channel, [[32, 32, 64], [64, 64, 128], [64, 96, 128]])
        self.sa2 = PointNetSetAbstractionMsg(512, [0.2, 0.2], [64, 128], 128+128+64, [[128, 128, 256], [128, 196, 256]])
        self.sa3 = PointNetSetAbstraction(npoint=None, radius=None, nsample=None, in_channel=512 + 3, mlp=[256, 512, 1024], group_all=True)
        self.fp3 = PointNetFeaturePropagation(in_channel=1536, mlp=[256, 256])
        self.fp2 = PointNetFeaturePropagation(in_channel=576, mlp=[256, 128])
        self.fp1 = PointNetFeaturePropagation(in_channel=131+additional_channel, mlp=[128, 128])
        self.conv1 = nn.Conv1d(128, 128, 1)
        self.bn1 = nn.BatchNorm1d(128)
        self.drop1 = nn.Dropout(0.5)
        self.conv2 = nn.Conv1d(128, num_classes, 1)

    def forward(self, xyz): 
        # Set Abstraction layers
        B,C,N = xyz.shape
        if self.normal_channel:
            l0_points = xyz
            l0_xyz = xyz[:,:3,:]
        else:
            l0_points = xyz
            l0_xyz = xyz
        l1_xyz, l1_points = self.sa1(l0_xyz, l0_points)
        l2_xyz, l2_points = self.sa2(l1_xyz, l1_points)
        l3_xyz, l3_points = self.sa3(l2_xyz, l2_points)
        # Feature Propagation layers
        l2_points = self.fp3(l2_xyz, l3_xyz, l2_points, l3_points)
        l1_points = self.fp2(l1_xyz, l2_xyz, l1_points, l2_points)
        l0_points = self.fp1(l0_xyz, l1_xyz, l0_points, l1_points)
        # FC layers
        feat = F.relu(self.bn1(self.conv1(l0_points)))
        x = self.drop1(feat)
        x = self.conv2(x)
        #x = F.log_softmax(x, dim=1)
        #x = x.permute(0, 2, 1) 
        return x, l3_points


class DGCNN(nn.Module):
    def __init__(self, args, seg_num_all = 5): #seg_num_all = number of class 
        super(DGCNN, self).__init__()
        self.args = args
        self.seg_num_all = seg_num_all
        self.k = args.k
        self.transform_net = Transform_Net(args)
        
        self.bn1 = nn.BatchNorm2d(64)
        self.bn2 = nn.BatchNorm2d(64)
        self.bn3 = nn.BatchNorm2d(64)
        self.bn4 = nn.BatchNorm2d(64)
        self.bn5 = nn.BatchNorm2d(64)
        self.bn6 = nn.BatchNorm1d(args.emb_dims)
        self.bn7 = nn.BatchNorm1d(64)
        self.bn8 = nn.BatchNorm1d(256)
        self.bn9 = nn.BatchNorm1d(256)
        self.bn10 = nn.BatchNorm1d(128)

        self.conv1 = nn.Sequential(nn.Conv2d(6, 64, kernel_size=1, bias=False),
                                   self.bn1,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv2 = nn.Sequential(nn.Conv2d(64, 64, kernel_size=1, bias=False),
                                   self.bn2,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv3 = nn.Sequential(nn.Conv2d(64*2, 64, kernel_size=1, bias=False),
                                   self.bn3,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv4 = nn.Sequential(nn.Conv2d(64, 64, kernel_size=1, bias=False),
                                   self.bn4,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv5 = nn.Sequential(nn.Conv2d(64*2, 64, kernel_size=1, bias=False),
                                   self.bn5,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv6 = nn.Sequential(nn.Conv1d(192, args.emb_dims, kernel_size=1, bias=False),
                                   self.bn6,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv8 = nn.Sequential(nn.Conv1d(1216, 256, kernel_size=1, bias=False),
                                   self.bn8,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.dp1 = nn.Dropout(p=args.dropout)
        self.conv9 = nn.Sequential(nn.Conv1d(256, 256, kernel_size=1, bias=False),
                                   self.bn9,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.dp2 = nn.Dropout(p=args.dropout)
        self.conv10 = nn.Sequential(nn.Conv1d(256, 128, kernel_size=1, bias=False),
                                   self.bn10,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv11 = nn.Conv1d(128, self.seg_num_all, kernel_size=1, bias=False)
        

    def forward(self, x):
        batch_size = x.size(0)
        num_points = x.size(2) 
        
        x0 = get_graph_feature(x, k=self.k)     # (batch_size, 3, num_points) -> (batch_size, 3*2, num_points, k)
        t = self.transform_net(x0)              # (batch_size, 3, 3)
        x = x.transpose(2, 1)                   # (batch_size, 3, num_points) -> (batch_size, num_points, 3)
        x = torch.bmm(x, t)                     # (batch_size, num_points, 3) * (batch_size, 3, 3) -> (batch_size, num_points, 3)
        x = x.transpose(2, 1)                   # (batch_size, num_points, 3) -> (batch_size, 3, num_points)

        x = get_graph_feature(x, k=self.k)      # (batch_size, 3, num_points) -> (batch_size, 3*2, num_points, k)
        x = self.conv1(x)                       # (batch_size, 3*2, num_points, k) -> (batch_size, 64, num_points, k)
        x = self.conv2(x)                       # (batch_size, 64, num_points, k)  -> (batch_size, 64, num_points, k) 
        x1 = x.max(dim=-1, keepdim=False)[0]    # (batch_size, 64, num_points, k) -> (batch_size, 64, num_points) 

        x = get_graph_feature(x1, k=self.k)     # (batch_size, 64, num_points) -> (batch_size, 64*2, num_points, k)
        x = self.conv3(x)                       # (batch_size, 64*2, num_points, k) -> (batch_size, 64, num_points, k)
        x = self.conv4(x)                       # (batch_size, 64, num_points, k) -> (batch_size, 64, num_points, k)
        x2 = x.max(dim=-1, keepdim=False)[0]    # (batch_size, 64, num_points, k) -> (batch_size, 64, num_points)

        x = get_graph_feature(x2, k=self.k)     # (batch_size, 64, num_points) -> (batch_size, 64*2, num_points, k)
        x = self.conv5(x)                       # (batch_size, 64*2, num_points, k) -> (batch_size, 64, num_points, k)
        x3 = x.max(dim=-1, keepdim=False)[0]    # (batch_size, 64, num_points, k) -> (batch_size, 64, num_points)

        x = torch.cat((x1, x2, x3), dim=1)      # (batch_size, 64*3, num_points)

        x = self.conv6(x)                       # (batch_size, 64*3, num_points) -> (batch_size, emb_dims, num_points)
        x = x.max(dim=-1, keepdim=True)[0]      # (batch_size, emb_dims, num_points) -> (batch_size, emb_dims, 1) / emb_dims = 1024

        #l = l.view(batch_size, -1, 1)           # (batch_size, num_categoties, 1)  
        #l = self.conv7(l)                       # (batch_size, num_categoties, 1) -> (batch_size, 64, 1)

        #x = torch.cat((x, l), dim=1)            # (batch_size, 1088, 1)
        x = x.repeat(1, 1, num_points)          # (batch_size, emb_dims, num_points) 

        x = torch.cat((x, x1, x2, x3), dim=1)   # (batch_size, 1024+64*3, num_points)

        x = self.conv8(x)                       # (batch_size, 1024+64*3, num_points) -> (batch_size, 256, num_points)
        x = self.dp1(x)
        x = self.conv9(x)                       # (batch_size, 256, num_points) -> (batch_size, 256, num_points)
        x = self.dp2(x)
        x = self.conv10(x)                      # (batch_size, 256, num_points) -> (batch_size, 128, num_points)
        x = self.conv11(x)                      # (batch_size, 256, num_points) -> (batch_size, seg_num_all, num_points)
        
        return x
    

class AttentionPooling(nn.Module):
    def __init__(self, args, in_channel):
        super(AttentionPooling, self).__init__()
        self.args = args
        self.alpha = 0.2
        self.bn1 = nn.BatchNorm2d(int(in_channel/2))
        self.bn2 = nn.BatchNorm2d(in_channel)
        self.conv1 = nn.Sequential(nn.Conv2d(in_channel, int(in_channel/2), kernel_size=1, bias=False),
                                   self.bn1,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv2 = nn.Sequential(nn.Conv2d(int(in_channel/2), in_channel, kernel_size=1, bias=False),
                                   self.bn2,
                                   nn.LeakyReLU(negative_slope=0.2))

    def forward(self, x):
        '''
        Input:
            x: edge convoluted data [batch_size, in_channel, num_points, k] 
        Return:
            attention_pooling: results of graph pooling [batch_size, in_channel, num_points] 
        ''' 
        
        B, C, npoints, k = x.size() 
        x = self.conv1(x) # [batch_size, in_channel/2, num_points, k]
        x = self.conv2(x) # [batch_size, in_channel, num_points, k]
        attention = F.softmax(x, dim=-1) # [batch_size, in_channel, num_points, k]
        
        attention_pooling = torch.sum(torch.mul(attention, x),dim = -1) # [batch_size, in_channel, num_points]
        #attention_pooling  = attention_pooling.permute(0, 2, 1).contiguous()
        return attention_pooling
        
class ADGCNN(nn.Module):
    def __init__(self, args, seg_num_all = 5): #seg_num_all = number of class 
        super(ADGCNN, self).__init__()
        self.args = args
        self.seg_num_all = seg_num_all
        self.k = args.k
        self.transform_net = Transform_Net(args)
        
        self.bn1 = nn.BatchNorm2d(64)
        self.bn2 = nn.BatchNorm2d(64)
        self.bn3 = nn.BatchNorm2d(64)
        self.bn4 = nn.BatchNorm2d(64)
        self.bn5 = nn.BatchNorm2d(64)
        self.bn6 = nn.BatchNorm1d(args.emb_dims)
        self.bn7 = nn.BatchNorm1d(64)
        self.bn8 = nn.BatchNorm1d(256)
        self.bn9 = nn.BatchNorm1d(256)
        self.bn10 = nn.BatchNorm1d(128)

        self.conv1 = nn.Sequential(nn.Conv2d(6, 64, kernel_size=1, bias=False),
                                   self.bn1,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv2 = nn.Sequential(nn.Conv2d(64, 64, kernel_size=1, bias=False),
                                   self.bn2,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv3 = nn.Sequential(nn.Conv2d(64*2, 64, kernel_size=1, bias=False),
                                   self.bn3,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv4 = nn.Sequential(nn.Conv2d(64, 64, kernel_size=1, bias=False),
                                   self.bn4,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv5 = nn.Sequential(nn.Conv2d(64*2, 64, kernel_size=1, bias=False),
                                   self.bn5,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv6 = nn.Sequential(nn.Conv1d(192, args.emb_dims, kernel_size=1, bias=False),
                                   self.bn6,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.attpool1 = AttentionPooling(args, in_channel = 64)
        self.attpool2 = AttentionPooling(args, in_channel = 64)
        self.attpool3 = AttentionPooling(args, in_channel = 64)
        self.conv8 = nn.Sequential(nn.Conv1d(1216, 256, kernel_size=1, bias=False),
                                   self.bn8,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.dp1 = nn.Dropout(p=args.dropout)
        self.conv9 = nn.Sequential(nn.Conv1d(256, 256, kernel_size=1, bias=False),
                                   self.bn9,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.dp2 = nn.Dropout(p=args.dropout)
        self.conv10 = nn.Sequential(nn.Conv1d(256, 128, kernel_size=1, bias=False),
                                   self.bn10,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv11 = nn.Conv1d(128, self.seg_num_all, kernel_size=1, bias=False)
        

    def forward(self, x):
        batch_size = x.size(0)
        num_points = x.size(2)

        x0 = get_graph_feature(x, k=self.k)     # (batch_size, 3, num_points) -> (batch_size, 3*2, num_points, k)
        t = self.transform_net(x0)              # (batch_size, 3, 3)
        x = x.transpose(2, 1)                   # (batch_size, 3, num_points) -> (batch_size, num_points, 3)
        x = torch.bmm(x, t)                     # (batch_size, num_points, 3) * (batch_size, 3, 3) -> (batch_size, num_points, 3)
        x = x.transpose(2, 1)                   # (batch_size, num_points, 3) -> (batch_size, 3, num_points)

        x = get_graph_feature(x, k=self.k)      # (batch_size, 3, num_points) -> (batch_size, 3*2, num_points, k)
        x = self.conv1(x)                       # (batch_size, 3*2, num_points, k) -> (batch_size, 64, num_points, k)
        x = self.conv2(x)                       # (batch_size, 64, num_points, k) -> (batch_size, 64, num_points, k)
        x1 = self.attpool1(x)                   # (batch_size, 64, num_points, k) -> (batch_size, 64, num_points)

        x = get_graph_feature(x1, k=self.k)     # (batch_size, 64, num_points) -> (batch_size, 64*2, num_points, k)
        x = self.conv3(x)                       # (batch_size, 64*2, num_points, k) -> (batch_size, 64, num_points, k)
        x = self.conv4(x)                       # (batch_size, 64, num_points, k) -> (batch_size, 64, num_points, k)
        x2 = self.attpool2(x)                   # (batch_size, 64, num_points, k) -> (batch_size, 64, num_points)

        x = get_graph_feature(x2, k=self.k)     # (batch_size, 64, num_points) -> (batch_size, 64*2, num_points, k)
        x = self.conv5(x)                       # (batch_size, 64*2, num_points, k) -> (batch_size, 64, num_points, k)
        x3 = self.attpool3(x)                   # (batch_size, 64, num_points, k) -> (batch_size, 64, num_points)

        x = torch.cat((x1, x2, x3), dim=1)      # (batch_size, 64*3, num_points)

        x = self.conv6(x)                       # (batch_size, 64*3, num_points) -> (batch_size, emb_dims, num_points)

        x = torch.cat((x, x1, x2, x3), dim=1)   # (batch_size, 1024+64*3, num_points)

        x = self.conv8(x)                       # (batch_size, 1024+64*3, num_points) -> (batch_size, 256, num_points)
        x = self.dp1(x)
        x = self.conv9(x)                       # (batch_size, 256, num_points) -> (batch_size, 256, num_points)
        x = self.dp2(x)
        x = self.conv10(x)                      # (batch_size, 256, num_points) -> (batch_size, 128, num_points)
        x = self.conv11(x)                      # (batch_size, 256, num_points) -> (batch_size, seg_num_all, num_points)
        
        return x
        
        
def get_graph_feature_geo(x, k=20, idx=None):
    batch_size = x.size(0)
    num_points = x.size(2)
    x = x.view(batch_size, -1, num_points)
    if idx is None:
        idx = knn(x, k=k)   # (batch_size, num_points, k)
    device = torch.device('cuda')

    idx_base = torch.arange(0, batch_size, device=device).view(-1, 1, 1)*num_points

    idx = idx + idx_base

    idx = idx.view(-1)
 
    _, num_dims, _ = x.size()

    x = x.transpose(2, 1).contiguous()   # (batch_size, num_points, num_dims)  -> (batch_size*num_points, num_dims) #   batch_size * num_points * k + range(0, batch_size*num_points)
    feature = x.view(batch_size*num_points, -1)[idx, :]
    feature = feature.view(batch_size, num_points, k, num_dims) 
    x = x.view(batch_size, num_points, 1, num_dims).repeat(1, 1, k, 1)
    
    feature = torch.cat((feature-x, x), dim=3).permute(0, 3, 1, 2).contiguous()
  
    return feature, idx

def car2sph(x):
    '''
    Input: 
        x: cartesian coordinate representation of input data #[batch_size, 3, num_points, k]
    Return: 
        x': Points converted to spherical coordinate #[batch_size, 6, num_points, k] 
    '''
    x = x.permute(0,2,3,1).contiguous() # [batch_size, 3, num_points, k] -> [batch_size, num_points, k, 3]
    x_sph = torch.zeros(size = x.shape) 
    XsqPlusYsq = x[:,:,:,0]**2 + x[:,:,:,1]**2
    x_sph[:,:,:,0] = torch.sqrt(XsqPlusYsq + x[:,:,:,2]**2)   # r
    x_sph[:,:,:,1] = torch.atan2(x[:,:,:,2], torch.sqrt(XsqPlusYsq))    # theta
    x_sph[:,:,:,2] = torch.atan2(x[:,:,:,1], x[:,:,:,0])                           # phi

    x_sph_avg = torch.mean(x_sph, dim=-1, keepdim = True)
    x_sph_dev = x_sph - x_sph_avg 
    x_sph_cat = torch.cat((x_sph,x_sph_dev), dim=-1) # [batch_size, num_points, k, 3] -> [batch_size, num_points, k, 6]
    x_sph_cat = x_sph_cat.permute(0,3,1,2).contiguous() # [batch_size, num_points, k, 6] -> [batch_size, 6, num_points, k] 
    return x_sph_cat.cuda()


class SphericalFeaturePropagation(nn.Module): 
    def __init__(self, args):
        super(SphericalFeaturePropagation, self).__init__()
        self.args = args
        self.k = args.k
        self.bn1 = nn.BatchNorm1d(6)
        self.bn2 = nn.BatchNorm1d(3)
        self.bn3 = nn.BatchNorm1d(1)
        self.conv1 = nn.Sequential(nn.Conv1d(6, 6, kernel_size=1, bias=False),
                                   self.bn1,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv2 = nn.Sequential(nn.Conv1d(6, 3, kernel_size=1, bias=False),
                                   self.bn2,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv3 = nn.Sequential(nn.Conv1d(3, 1, kernel_size=1, bias=False),
                                   self.bn3,
                                   nn.LeakyReLU(negative_slope=0.2))
                                   
        
    def forward(self, x_loc, x, k= 30): 
        '''
        Input: 
            x_loc: cartesian coordinate representation of input data [batch_size, 3, num_points] 
            x: pooled data [batch_size, in_channel, num_points] 
        Return: 
            x': feature propagated data [batch_size, in_channel, num_points] 
        ''' 
        batch_size, num_dims, npoints = x.size()
        x_loc, idx = get_graph_feature_geo(x_loc, k = k) # [batch_size, 3, num_points] -> [batch_size, 6, num_points, k]
        x_loc = x_loc[:,:3, :, :].cuda() # [batch_size, 6, num_points] -> [batch_size, 3, num_points, k] 
        x_loc = car2sph(x_loc) # [batch_size, 3, num_points, k] -> [batch_size, 6, num_points, k]
        x_loc = x_loc.permute(0,2,1,3).contiguous() # [batch_size, 6, num_points, k] -> [batch_size, num_points, 6, k]
        x_loc = x_loc.view(batch_size*npoints, 6, k) # [batch_size, num_points, 6, k] -> [batch_size*num_points, 6, k]
        x_loc = self.conv1(x_loc) # [batch_size*num_points, 6, k] -> [batch_size*num_points, 6, k]
        x_loc = self.conv2(x_loc) # [batch_size*num_points, 6, k] -> [batch_size*num_points, 3, k]
        x_loc = self.conv3(x_loc) # [batch_size*num_points, 3, k] -> [batch_size*num_points, 1, k]
        x_loc = x_loc.view(batch_size, npoints, 1, k) # [batch_size*num_points, 1, k] -> [batch_size, num_points, 1, k]
        
        spherical_att = F.softmax(x_loc, dim=-1)
        spherical_att = spherical_att.expand(batch_size, npoints, num_dims, k) # [batch_size, num_points, 1, k] -> [batch_size, num_points, in_channel, k]
        spherical_att = spherical_att.permute(0,2,1,3).contiguous() # [batch_size, num_points, in_channel, k] ->  [batch_size, in_channel, num_points, k]

        x_k = x.transpose(2, 1).contiguous()   # [batch_size, in_channel, num_points] -> [batch_size, num_points, in_channel]
        x_k = x_k.view(batch_size*npoints, -1)[idx, :] #
        x_k = x_k.view(batch_size, npoints, k, num_dims) # -> [batch_size, num_points, k, in_channel] 
        x_k = x_k.permute(0,3,1,2).contiguous() # [batch_size, num_points, k, in_channel] -> [batch_size, in_channel, num_points, k]
        
        agg_feat = torch.sum(torch.mul(x_k, spherical_att), dim = -1) #[batch_size, in_channel, num_points, k] -> [batch_size, in_channel, num_points]

        return x+agg_feat

class SPHADGCNN(nn.Module):
    def __init__(self, args, seg_num_all = 5): #seg_num_all = number of class 
        super(SPHADGCNN, self).__init__()
        self.args = args    
        self.seg_num_all = seg_num_all
        self.k = args.k
        self.transform_net = Transform_Net(args)
        
        self.bn1 = nn.BatchNorm2d(64)
        self.bn2 = nn.BatchNorm2d(64)
        self.bn3 = nn.BatchNorm2d(64)
        self.bn4 = nn.BatchNorm2d(64)
        self.bn5 = nn.BatchNorm2d(64)
        self.bn6 = nn.BatchNorm1d(args.emb_dims)
        self.bn7 = nn.BatchNorm1d(64)
        self.bn8 = nn.BatchNorm1d(256) 
        self.bn9 = nn.BatchNorm1d(256)
        self.bn10 = nn.BatchNorm1d(128)

        self.conv1 = nn.Sequential(nn.Conv2d(6, 64, kernel_size=1, bias=False),
                                   self.bn1,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv2 = nn.Sequential(nn.Conv2d(64, 64, kernel_size=1, bias=False),
                                   self.bn2,
                                   nn.LeakyReLU(negative_slope=0.2)) 
        self.conv3 = nn.Sequential(nn.Conv2d(64*2, 64, kernel_size=1, bias=False),
                                   self.bn3,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv4 = nn.Sequential(nn.Conv2d(64, 64, kernel_size=1, bias=False),
                                   self.bn4,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv5 = nn.Sequential(nn.Conv2d(64*2, 64, kernel_size=1, bias=False),
                                   self.bn5,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv6 = nn.Sequential(nn.Conv1d(192, args.emb_dims, kernel_size=1, bias=False),
                                   self.bn6,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv8 = nn.Sequential(nn.Conv1d(1216, 256, kernel_size=1, bias=False),
                                   self.bn8,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.dp1 = nn.Dropout(p=args.dropout)
        self.conv9 = nn.Sequential(nn.Conv1d(256, 256, kernel_size=1, bias=False),
                                   self.bn9,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.dp2 = nn.Dropout(p=args.dropout)
        self.conv10 = nn.Sequential(nn.Conv1d(256, 128, kernel_size=1, bias=False),
                                   self.bn10,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv11 = nn.Conv1d(128, self.seg_num_all, kernel_size=1, bias=False)
        
        self.attpool1 = AttentionPooling(args, in_channel = 64)
        self.attpool2 = AttentionPooling(args, in_channel = 64)
        self.attpool3 = AttentionPooling(args, in_channel = 64)
        
        #change this!!!
        self.sphagg1 = SphericalFeaturePropagation(args)
        self.sphagg2 = SphericalFeaturePropagation(args)
        self.sphagg3 = SphericalFeaturePropagation(args)
        

    def forward(self, x_input):
        batch_size = x_input.size(0)
        num_points = x_input.size(2)
        
        x0 = get_graph_feature(x_input, k=self.k)     # (batch_size, 3, num_points) -> (batch_size, 3*2, num_points, k)
        t = self.transform_net(x0)              # (batch_size, 3, 3)
        x = x_input.transpose(2, 1)             # (batch_size, 3, num_points) -> (batch_size, num_points, 3)
        x = torch.bmm(x, t)                     # (batch_size, num_points, 3) * (batch_size, 3, 3) -> (batch_size, num_points, 3)
        x = x.transpose(2, 1)                   # (batch_size, num_points, 3) -> (batch_size, 3, num_points)

        x = get_graph_feature(x, k=self.k)      # (batch_size, 3, num_points) -> (batch_size, 3*2, num_points, k)
        x = self.conv1(x)                       # (batch_size, 3*2, num_points, k) -> (batch_size, 64, num_points, k)
        x = self.conv2(x)                       # (batch_size, 64, num_points, k) -> (batch_size, 64, num_points, k)
        x = self.attpool1(x)                    # (batch_size, 64, num_points, k) -> (batch_size, 64, num_points)
        x1 = self.sphagg1(x_input, x, k=self.k) # (batch_size, 64, num_points) -> (batch_size, 64, num_points)
        
        x = get_graph_feature(x1, k=self.k)     # (batch_size, 64, num_points) -> (batch_size, 64*2, num_points, k)
        x = self.conv3(x)                       # (batch_size, 64*2, num_points, k) -> (batch_size, 64, num_points, k)
        x = self.conv4(x)                       # (batch_size, 64, num_points, k) -> (batch_size, 64, num_points, k)
        x = self.attpool2(x)                   # (batch_size, 64, num_points, k) -> (batch_size, 64, num_points)
        x2 = self.sphagg2(x_input, x, k=self.k) # (batch_size, 64, num_points) -> (batch_size, 64, num_points)

        x = get_graph_feature(x2, k=self.k)     # (batch_size, 64, num_points) -> (batch_size, 64*2, num_points, k)
        x = self.conv5(x)                       # (batch_size, 64*2, num_points, k) -> (batch_size, 64, num_points, k)
        x = self.attpool3(x)                   # (batch_size, 64, num_points, k) -> (batch_size, 64, num_points)
        x3 = self.sphagg3(x_input, x, k=self.k) # (batch_size, 64, num_points) -> (batch_size, 64, num_points)

        x = torch.cat((x1, x2, x3), dim=1)      # (batch_size, 64*3, num_points)

        x = self.conv6(x)                       # (batch_size, 64*3, num_points) -> (batch_size, emb_dims, num_points)

        x = torch.cat((x, x1, x2, x3), dim=1)   # (batch_size, 1024+64*3, num_points)

        x = self.conv8(x)                       # (batch_size, 1024+64*3, num_points) -> (batch_size, 256, num_points)
        x = self.dp1(x)
        x = self.conv9(x)                       # (batch_size, 256, num_points) -> (batch_size, 256, num_points)
        x = self.dp2(x)
        x = self.conv10(x)                      # (batch_size, 256, num_points) -> (batch_size, 128, num_points)
        x = self.conv11(x)                      # (batch_size, 256, num_points) -> (batch_size, seg_num_all, num_points)
        
        return x
