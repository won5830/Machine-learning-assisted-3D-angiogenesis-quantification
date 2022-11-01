#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
@Author: Wonjun Lee
@Contact: won5830@gmail.com
@File: data.py
@Time: 2021/08/10 6:35 PM
"""

import os
import sys
import glob
import h5py
import numpy as np
from torch.utils.data import Dataset
import warnings

with warnings.catch_warnings():
    warnings.simplefilter('ignore', category = UserWarning)
    from torchvision import transforms
from math import *

def load_data(partition):
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    DATA_DIR = os.path.join(BASE_DIR, 'data')
    all_data = [] 
    all_label = []
    all_ids = []
    all_norm_info = []
    for h5_name in glob.glob(os.path.join(DATA_DIR, 'test_skel_ply_hdf5_data_%s*.h5'%partition)):
        f = h5py.File(h5_name) 
        data = f['data'][:].astype('float32') #(set number, point number per each point, 3) = (20, 2400, 3)
        label = f['label'][:].astype('int64') #(set number, point number per each point) = (20, 2400)
        ids = f['ids'][:].astype('int64') #(set number, 1) = (20, 1) 
        norm_info = f['norm_info'][:].astype('float32') #(set number, 4) = (20, 4)
        f.close() 
        all_data.append(data)
        all_label.append(label)
        all_ids.append(ids)
        all_norm_info.append(norm_info) 
    all_data = np.concatenate(all_data, axis=0) #(total set number number, point number per each point, 3) = (15*20, 2400, 3)
    all_label = np.concatenate(all_label, axis=0) #(total set number, point number per each point) = (15*20, 2400)
    all_ids = np.concatenate(all_ids,axis=0) #(total set number, 1) = (15*20, 1)
    all_norm_info = np.concatenate(all_norm_info, axis=0) #(total set number, 4) = (15*20, 4)
    return all_data, all_label, all_ids, all_norm_info


class RandomFlip(object): 
    def __init__(self, p: float): 
      self.p=p; 
    def __call__(self, input):
      if np.random.choice([0,1],p=[self.p,1-self.p]): #X-axis flip
        input=np.concatenate((-input[:,0].reshape(-1,1),input[:,1].reshape(-1,1),input[:,2].reshape(-1,1)),axis=1)
      return input

class RandomPerturbation(object):
    def __init__(self, num):
        self.gauss_perc=num

    def __call__(self, input):
        #Gaussian Noise addition 
        #give value of its own distance here base
        relative_size=np.linalg.norm(input, axis=1) #distance of each KNN points
        noise = np.random.normal(0, self.gauss_perc/100, (input.shape))
        input = input + noise
        return input

class RandomZRotation(object): 
    def __init__(self,v: list):  #list given as random rotation range
      self.min_ang=v[0]
      self.max_ang=v[1] 
    def __call__(self, input):
      rot_ang=np.random.uniform(self.min_ang,self.max_ang)
      input=[[cos(rot_ang*pi/180), -sin(rot_ang*pi/180), 0], [sin(rot_ang*pi/180), cos(rot_ang*pi/180),0],[0,0,1]]@np.transpose(input)
      input=np.transpose(input)
      return input

class Normalization(object):
    #Nomalization base on x-axis (only used for training)
    def __call__(self, input): 
        norm_pointcloud = input - np.min(input, axis=0) 
        norm_pointcloud /= np.max(input[:,0]) - np.min(input[:,0])
        return  norm_pointcloud
        

class PointDataset(Dataset):
    def __init__(self, num_points, partition='train', augmentation = True):
        self.data, self.label, self.ids, self.norm_info = load_data(partition)
        self.partition = partition
        self.num_points = num_points 
        if augmentation:
            self.transforms=transforms.Compose([
                        RandomPerturbation(0.01),
                        RandomZRotation([-5,5]), 
                        RandomFlip(p=0.5),
                        Normalization()]) 
        else:
            self.transforms=transforms.Compose([])       
    
    def __getitem__(self, item): 
        ids = self.ids[item]
        pointcloud = self.data[item][:self.num_points] #(2400, 3) 
        label = self.label[item][:self.num_points]  
        norm_info = self.norm_info[item]

        if self.partition == 'train':
            indices = list(range(pointcloud.shape[0]))
            np.random.shuffle(indices)
            pointcloud = pointcloud[indices]
            label = label[indices]
        return self.transforms(pointcloud), label, ids, norm_info

    def __len__(self):
        return self.data.shape[0] 


if __name__ == '__main__':
    train = PointDataset(50)
    test = PointDataset(50, 'test')
    for data, label in train:
        print(data.shape)
        print(label.shape)
