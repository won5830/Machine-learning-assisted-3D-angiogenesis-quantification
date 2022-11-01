#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
@Author: Byoungkwon Yoon
@Contact: dbss@gmail.com
@File: eval_util.py
@Time: 2021/08/10 6:35 PM
"""

import heapq
import numpy as np
from sklearn.cluster import DBSCAN
import pickle
from scipy.interpolate import UnivariateSpline
import math
from scipy.optimize import curve_fit
import operator
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neighbors import NearestNeighbors
from sklearn.linear_model import LinearRegression
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import open3d as o3d
from scipy.io import savemat
import os
import csv


class PathTracking():
  def __init__(self, data, knn_search=7):
    link_num = len(data['Link'].keys()) # with 4th index as 1
    joint_num = len(data['Joint'].keys()) # with 4th index as 2
    #tot_skel=np.zeros((link_num+joint_num+gel_num+2,4))
    link_stack = np.empty((0,4))
    for i in range(link_num):
      tmp=data['Link'][str(i)]['points']
      pos_ind = np.concatenate((tmp, np.tile(1, (len(tmp),1))), axis=1)
      link_stack = np.append(link_stack, pos_ind, axis=0) 
    joint_stack = np.empty((0,4))
    for i in range(joint_num):
      tmp = data['Joint'][str(i)]['points']
      pos_ind = np.concatenate((tmp, np.tile(2, (len(tmp),1))), axis=1) 
      joint_stack = np.append(joint_stack, pos_ind, axis=0)
    gel_ind=0 
    gel_stack = np.concatenate((data['Gel'], np.tile(gel_ind, (len(data['Gel']),1))), axis=1)

    self.tot_skel = np.concatenate((link_stack, joint_stack, gel_stack), axis=0)
    self.tip_skel = data['Tipcell']
    self.knn_search = knn_search # should substract 1 for final search range(excepting itself)
    self.nbrs_knn = NearestNeighbors(n_neighbors=3*knn_search, algorithm='ball_tree').fit(self.tot_skel[:,:3])
    self.nbrs = NearestNeighbors(n_neighbors=1, algorithm='ball_tree').fit(self.tot_skel[:,0:3])
  
  def dir_knn(self, n, pt, ptcloud, dir): 
    indices = self.nbrs_knn.kneighbors(np.expand_dims(pt,axis=0))[1] #take index only
    indices = indices[:,1:] 
    next_pt = np.squeeze(ptcloud[indices, :], axis=0)
    nxt_dir = next_pt[:,:3] - pt
    dot = np.inner(nxt_dir,dir)
    ind = np.array([i for i in range(dot.size) if dot[i]>0]) #In the same direction!!
    if ind.size>n:
      ind=ind[:n]
    return next_pt[ind, :]
  
  def Regression3D(self, A):
    #input should be nX3 matrix 
    arr0 = np.transpose(A[-1,:])
    mat = np.transpose(A) - np.expand_dims(arr0, axis=1)
    U = np.linalg.svd(mat)[0]
    d = U[:,0]
    t = np.transpose(d)@mat
    fin = np.repeat(np.expand_dims(t,axis=0), 3, axis=0)*np.expand_dims(d, axis=1) + np.repeat(np.expand_dims(arr0,axis=1), len(A), axis=1)
    return fin

  def __call__(self, tip_index):
    temp= self.tip_skel[str(tip_index)]['points'] 
    temp = np.expand_dims(np.mean(temp, axis=0), axis=0)
    temp = np.concatenate((temp, [[3]]), axis=1) 

    #Find 1st nearest point
    path_list= np.empty((0,4))
    path_list = np.append(path_list, temp, axis=0)

    indices = self.nbrs.kneighbors(temp[:,:3])[1] #take index only 
    next_pt = self.tot_skel[indices, :]
    next_pt = np.squeeze(next_pt, axis=0)
    path_list = np.append(path_list, next_pt, axis=0) 

    # Tracking
    step=1
    while (True): 
      #Second index 
      momentum_range = 5 #range of tail 
      if (step > momentum_range): 
        temp_dir = np.empty((0,3)) 
        #3D linear regression
        pt_reg = path_list[step-momentum_range:step+1, :3] 
        reg_lin = self.Regression3D(pt_reg)
        temp_direc = reg_lin[:, -1] - reg_lin[:, -2]
        n_temp_direc = temp_direc/np.linalg.norm(temp_direc, axis =0)
      elif ((momentum_range>= step) and (step>2)):
        temp_dir = np.empty((0,3))
        #3D linear regression with 3 tails
        pt_reg = path_list[step-2:step+1, :3]
        reg_lin = self.Regression3D(pt_reg) 
        temp_direc = reg_lin[:, -1] - reg_lin[:, -2]
        n_temp_direc = temp_direc/np.linalg.norm(temp_direc, axis =0)
    
      else: 
        temp_direc =  path_list[step,:3] - path_list[step-1,:3]
        n_temp_direc = temp_direc/np.linalg.norm(temp_direc, axis =0)
    
  
      #directional
      nxt_candidate_wlabel = self.dir_knn(self.knn_search, path_list[step,:3], self.tot_skel, n_temp_direc)
  
      nxt_candidate = nxt_candidate_wlabel[:, 0:3]
      candidate_direc = nxt_candidate - path_list[step, :3]
      n_candidate_direc = candidate_direc /np.linalg.norm(candidate_direc, axis =1)[:,None]
      dir_change = n_candidate_direc - n_temp_direc
      dir_change[:,1] = 1*dir_change[:,1] #weighting: multiplying constant infront of it
      direc_dist = np.linalg.norm(dir_change, axis =1)
  
      k=1
      try:
        #pull-down if you can 
        if (path_list[step,3] == 2): #if current anchor point is joint
          min_ind = np.argmin(direc_dist)
          tmp_pt = np.expand_dims(nxt_candidate_wlabel[min_ind,:], axis=0) 
          while ((tmp_pt[0,1] - path_list[step,1])>0):
            k+=1 
            #pull-down 
            sorted_ind = np.argsort(direc_dist)[:k]
            min_ind = sorted_ind[k-1]
            tmp_pt = np.expand_dims(nxt_candidate_wlabel[min_ind,:], axis=0)
        else: 
          min_ind = np.argmin(direc_dist)
      except:
        #if error occurs? stop pull-down
        min_ind = np.argmin(direc_dist)

      nxt_pt = np.expand_dims(nxt_candidate_wlabel[min_ind,:], axis=0)
      path_list = np.append(path_list, nxt_pt, axis=0)
      step+=1
      if (nxt_candidate_wlabel[min_ind, 3]==0):
        #stop if point arrives gel / or iteration exceeds the limit
        if len(path_list)<5:
            path_list = np.empty((0,4))
        break
      elif (step>5000):
          path_list = np.empty((0,4))
          break
    return path_list

def Clustering(points, eps, minpts):
    ### DBSCAN
    model = DBSCAN(eps, min_samples=minpts)  ### DBSCAN Parameters
    model.fit_predict(points)
    prediction = model.fit_predict(points)
    labels = model.labels_
    
    return labels

def ProcessJoint(points, eps, minpts):
    labels_joint = Clustering(points, eps, minpts)
    
    
    ###### KNN for noise Clustered joint
    noise_idx = np.where(labels_joint==-1)[0]
    noise_points = points_joint[noise_idx]
    classifier = KNeighborsClassifier(n_neighbors = 5)
   
    ###### make KNN data
    temp_points = np.concatenate((points_gel,points_link,points_noise,points_tip),axis=0)  
    temp_type = np.sort(target)
    temp_type = np.delete(temp_type, np.where(temp_type==1))
    classifier.fit(temp_points, temp_type)
    guesses = classifier.predict(noise_points)
    
    t = target
    joint_index = np.where(target==1)[0]
    #print(guesses)
    for i in range(len(noise_idx)):
        t[joint_index[noise_idx[i]]] = guesses[i]
        
    labels_joint = np.delete(labels_joint, noise_idx)
    #points_joint = np.delete(points_joint, noise_idx)
    
    return labels_joint, t 

def ProcessGel(points):
    labels_gel = Clustering(points, 35, 5)
    
    return labels_gel

def ProcessTipcell(points):
    labels_tip = Clustering(points, 35, 0)
    
    return labels_tip

def func(X,a,b,c,d):  ### Set Function type

    return a*(X**3) + b*(X**2) + c*(X**1) + d

def LinkLength(inpoints, labels):
    length = []  ## Length
    endToend = [] ## End to End Length
    
    EndPoints = [] ## End Points of Link
    
    Density = [] ## Density of Links, for clustering evaluation
    ### Fitting And Length Calculation
    for i in range(labels.max()+1):
        #print(i)
        data = inpoints[labels == i]  ## Get Pos of 1 link
    
        x = data[:,0] ## Get X coord
        y = data[:,1] ## Get Y coord
        z = data[:,2] ## Get Z coord
        
        ## sort pos by y value
        s = y.argsort()  
        x = x[s]
        y = y[s]
        z = z[s]
            
        data[:,0] = x ## Overlab sorted Coord
        data[:,1] = y
        data[:,2] = z
        color = 'red'
        if len(x) > 10:   ## no fit for link less than 10 pts
        
            if len(x) > 50:  ## pts>50 : cut in half and fit
                color = 'lime'
                n1 = math.ceil(len(x)/2) ## Cut Index
                tempIdx = [np.arange(n1+10),np.arange(n1-10,len(x))]  ### [[0 1 2 3 4 5][6 7 8 9 10]] shape. used in next FOR
                #print(tempIdx)
            else:  ## else: fit at once
                color = 'red'
                tempIdx = [np.arange((len(x)))]
                
                
            ### list of fitted points
            xnew = []
            ynew = []
            znew = []
            
            
            for j in tempIdx:
                tempx = x[j]  ## load points
                tempy = y[j]
                tempz = z[j]
                
                u = np.arange(len(tempx)) ## make artificial axis for fitting
          
                poptx, pcovx = curve_fit(func, u, tempx) ## fit n vs x
                    
                tempXnew = func(u,poptx[0],poptx[1],poptx[2],poptx[3])
                xnew.append(tempXnew)
                    
                popty, pcovy = curve_fit(func, u, tempy) ## fit n vs y
                    
                tempYnew = func(u,popty[0],popty[1],popty[2],popty[3])
                ynew.append(tempYnew)
                 
                poptz, pcovz = curve_fit(func, u, tempz) ## fit n vs z
                    
                tempZnew = func(u,poptz[0],poptz[1],poptz[2],poptz[3])
                znew.append(tempZnew)
        
            if len(xnew) == 2:
                #print(xnew)
                xnew = np.concatenate((xnew[0][:n1],xnew[1][10:]))
                ynew = np.concatenate((ynew[0][:n1],ynew[1][10:]))
                znew = np.concatenate((znew[0][:n1],znew[1][10:]))
            
            else:
                xnew = xnew[0]
                ynew = ynew[0]
                znew = znew[0]
                
        else:
            
            ## links with pts<10 : no fitting
            
            xnew = x
            ynew = y
            znew = z
                    
        ##calculate length
        
        endLength = math.sqrt(pow(abs(xnew[0]-xnew[-1]),2) 
                              + pow(abs(ynew[0]-ynew[-1]),2) 
                              + pow(abs(znew[0]-znew[-1]),2))
        
        EndPoints.append([xnew[0],ynew[0],znew[0]]) ## Append End Points
        EndPoints.append([xnew[-1],ynew[-1],znew[-1]]) ## Append End Points
        
        tempLength = 0
        for j in range(len(x)-1):
            tempLength += math.sqrt(pow(abs(xnew[j]-xnew[j+1]),2)
                                    + pow(abs(ynew[j]-ynew[j+1]),2)
                                    + pow(abs(znew[j]-znew[j+1]),2))
        
        length.append(tempLength)

        endToend.append(endLength)
        
        Density.append(len(x)/endLength)
        
    return length, endToend, Density, EndPoints

def ProcessLink(inpoints, z_multiple, eps1, eps2):
    #ori_points_z = points[:,2]  ## store original points
    inpoints[:,2] = inpoints[:,2]*z_multiple  ## multiply z coord for seperation
    labels_link = Clustering(inpoints, eps1, 3)  ## Clustering Links
    
    inpoints[:,2]= inpoints[:,2]/z_multiple   ## restore z coord
       
    length, endToend, Density, EndPoints= LinkLength(inpoints,labels_link)  #### 1st Length Calc
    
    avg_density = np.average(Density)  ### Find Average Link Density
    labels_repeat = np.where(Density>avg_density)[0] ### Re clusteing High density links
    
    first_labels_link = labels_link[:]
    
    #print('AVG DENSITY:'+str(avg_density))
   
    #### Sepereate HD/LD Link######
    HDLink = np.array(inpoints[np.where(labels_link==0)])  ###Link points with High Density
    for i in labels_repeat[1:]:
        HDLink = np.append(HDLink, inpoints[np.where(labels_link==i)],axis=0)
    LDLink = inpoints[:]
    
    a1_rows = LDLink.view([('', LDLink.dtype)] * LDLink.shape[1])
    a2_rows = HDLink.view([('', HDLink.dtype)] * HDLink.shape[1])
    LDLink = np.setdiff1d(a1_rows, a2_rows).view(LDLink.dtype).reshape(-1, LDLink.shape[1])
    
    ##### Cluster HD/LD Link#####
    LDLink[:,2] = LDLink[:,2]*z_multiple
    HDLink[:,2] = HDLink[:,2]*z_multiple
    labels_LDLink = Clustering(LDLink, eps1, 3)
    labels_HDLink = Clustering(HDLink, eps2, 3)
    LDLink[:,2]= LDLink[:,2]/z_multiple
    HDLink[:,2]= HDLink[:,2]/z_multiple
    
    noiseIndex = np.where(labels_HDLink==-1)[0]
    labels_HDLink = labels_HDLink+(1+max(labels_LDLink))
    labels_HDLink[noiseIndex] = -1
    #print(labels_HDLink)
    #print(labels_LDLink)
    
    labels_link = np.concatenate((labels_LDLink,labels_HDLink))
    final_link_points = np.concatenate((LDLink,HDLink),axis=0) 
    
    length, endToend, Density,EndPoints = LinkLength(final_link_points,labels_link)
        
        
    return labels_link, length, endToend, EndPoints, Density, [LDLink, HDLink], final_link_points, first_labels_link
        

def FindConnection(labels_main, labels_sub, points_main, points_sub, eps, minpts):
    connectivity = []
    
    points_main2 = points_main[:]
    points_sub2 = points_sub[:]
    
    points_main2[:,2] = points_main2[:,2]*2
    points_sub2[:,2] = points_sub2[:,2]*2
    
    for i in range(labels_main.max()+1):  ## Get 1 from Main points
        conn_links = []
        for j in range(labels_sub.max()+1):  ## Get 1 from Sub Points
            idx_joint = np.where(labels_main == i)  ##Get Joint of selected
            idx_link = np.where(labels_sub == j)
            sample = np.concatenate((points_main2[idx_joint], points_sub2[idx_link]), axis=0) ## combine Main / Sub points
            
            
            ## DBSCAN Combined Pts
            model = DBSCAN(eps, min_samples=minpts)
            model.fit_predict(sample)
            pred = model.fit_predict(sample)
            labels_sample= model.labels_
            
            ## Connected if Combined Pts are Clustered As single chunk
            if labels_sample.max() == 0:
                conn_links.append(j)
        
        connectivity.append(conn_links)
    
    return connectivity
        
def makedict(labels_joint, points_joint, labels_link, points_link, labels_tip, points_tip):  ## Make OUTPUT DICT
    dic = {'Joint':[],'Link':[],'Tipcell':[], 'Gel':[], 'Noise':[]}
    
    for i in range(len(labels_joint)):
        temp = np.append(points_joint[i],labels_joint[i])
        dic['Joint'].append(temp)
        
    for i in range(len(labels_link)):
        temp = np.append(points_link[i],labels_link[i])
        dic['Link'].append(temp)
        
    for i in range(len(labels_tip)):
        temp = np.append(points_tip[i],labels_tip[i])
        dic['Tipcell'].append(temp)
        
    dic['Gel'].append(points_gel)
    dic['Noise'].append(points_noise)
        
    return dic

def showPC(tip,link,joint,points) :
    ## Get Selected Pts
    idx_tip = np.where(target==4)
    idx_tip_selected = idx_tip[0][tip]
    
    idx_link = np.where(target==2)
    idx_link_selected = idx_link[0][link]
    
    idx_joint = np.where(target==1)
    idx_joint_selected = idx_joint[0][joint]
    
    idx_gel = np.where(target==0)
    
    colors=np.array([[0.5,0.5,0.5,1]]*len(points))
    
    ## Set color to Selected Pts
    try:
        #colors[idx_tip_selected] = [1,0,0,1]
        #colors[idx_link_selected] = [0,1,0,1]   
        
        #colors[idx_gel] = [0,0,0,1]
        colors[idx_joint_selected] = [0,1,1,1]
    except:
        pass
    
    ## PC SHOW
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    pcd.colors = o3d.utility.Vector3dVector(colors[:, :3])
    o3d.visualization.draw_geometries([pcd])
    
def Connectivity(n):
     
    conn_tip = []
    for i in range(labels_tip.max()+1): conn_tip.append([])
    
    conn_LinkEnds = [] ## Connectivity of LinkEndPoint
    for i in range(len(LinkEnds)): conn_LinkEnds.append([])
    
    for i in range(labels_tip.max()+1):
        tip_idx = np.where(labels_tip==i)[0]
        ### get tipcell ref pos
        if len(tip_idx) != 1:
            tip_pos = points_tip[tip_idx].mean(axis=0)
        else :
            tip_pos = points_tip[tip_idx][0]
        
        ### Find Nearest LinkEnd
        conn_tip[i] = [0]   ## Initial Loop
        CurLength = math.sqrt(pow(abs(tip_pos[0]-LinkEnds[0][0]),2) 
                              + pow(abs(tip_pos[1]-LinkEnds[0][1]),2) 
                              + pow(abs(tip_pos[2]-LinkEnds[0][2]),2))
        
        for j in range(1,len(LinkEnds)):
            temp = math.sqrt(pow(abs(tip_pos[0]-LinkEnds[j][0]),2) 
                              + pow(abs(tip_pos[1]-LinkEnds[j][1]),2) 
                              + pow(abs(tip_pos[2]-LinkEnds[j][2]),2))
            if temp < CurLength:
                conn_tip[i] = [int(j/2)]
                CurLength = temp
        ### Record conn_LinkEnds Data 
        Cur_Link = conn_tip[i][0]
        l1 = math.sqrt(pow(abs(tip_pos[0]-LinkEnds[Cur_Link*2][0]),2) 
                          + pow(abs(tip_pos[1]-LinkEnds[Cur_Link*2][1]),2) 
                          + pow(abs(tip_pos[2]-LinkEnds[Cur_Link*2][2]),2))
        l2 = math.sqrt(pow(abs(tip_pos[0]-LinkEnds[Cur_Link*2+1][0]),2) 
                          + pow(abs(tip_pos[1]-LinkEnds[Cur_Link*2+1][1]),2) 
                          + pow(abs(tip_pos[2]-LinkEnds[Cur_Link*2+1][2]),2))
        if l1 < l2:
            conn_LinkEnds[Cur_Link*2] = ['t'+str(i)]
        else:
            conn_LinkEnds[Cur_Link*2+1] = ['t'+str(i)]
    
    
    
    ### Find Joint/Gel - Link Connection
    classifier = KNeighborsClassifier(n_neighbors = n)
    KNN_labels = []  ## Labels for KNN
    KNN_points = np.concatenate((points_joint, points_gel))  ## points for KNN
    KNN_points = KNN_points.tolist()
    
    for i in labels_joint:
        KNN_labels.append('j'+str(i))
    
    for i in labels_gel:
        KNN_labels.append('g') 
        
    classifier.fit(KNN_points, KNN_labels)
        
    temp_conn = classifier.predict(LinkEnds)
    
    for i in range(len(conn_LinkEnds)):
        if conn_LinkEnds[i] == []:
            temp = temp_conn[i][:]
            conn_LinkEnds[i] = [temp]
            
    
    conn_link = []
    for i in range(labels_link.max()+1): conn_link.append([])
    conn_joint = []
    for i in range(labels_joint.max()+1): conn_joint.append([])
    conn_gel = []
    
    for i in range(len(conn_LinkEnds)):   ## Make Connectivity List
        if conn_LinkEnds[i][0][0] == 'j':
            conn_joint[int(conn_LinkEnds[i][0][1:])].append(int(i/2))
        elif conn_LinkEnds[i][0][0] == 'g':
            conn_gel.append(int(i/2))
            
    for i in range(len(conn_LinkEnds)):
        try:
            conn_link[int(i/2)].append(conn_LinkEnds[i][0])
        except:
            pass
        
        
    return conn_link, conn_joint, conn_tip, conn_gel


def ShowSkel(*args):
    
    colors=np.array([[0.5,0.5,0.5,1]]*len(points))
    if len(args) == 0:
        colors[np.where(target==0)[0]] = [1,1,0,1]   #gel
        colors[np.where(target==1)[0]] = [0,1,0,1]   #joint
        colors[np.where(target==2)[0]] = [0,0,1,1]   #link
        colors[np.where(target==3)[0]] = [0,0,0,1]   #noise
        colors[np.where(target==4)[0]] = [1,0,0,1]   #Tipcell
    elif args[0] == 0: ## Gel
        colors[np.where(target!=0)[0]] = [0,0,0,1]
        colors[np.where(target==0)[0]] = [1,1,0,1]
    elif args[0] == 1: ## joint
        paint_index = [0]*len(points)
        t = np.where(target==1)[0]
        for i in range(len(t)):
            paint_index[t[i]] = labels_joint[i]*3 
        colors = plt.get_cmap("hsv")(paint_index)
        colors[np.where(target!=1)[0]] = [0,0,0,1]
        colors[t[np.where(labels_joint==-1)[0]]] = [1,0,1,1]
    elif args[0] == 2: ## link
        paint_index = [0]*len(points)
        t = np.where(target==2)[0]
        for i in range(len(t)):
            paint_index[t[i]] = labels_link[i]*3 
        colors = plt.get_cmap("prism")(paint_index)
        colors[np.where(target!=2)[0]] = [0,0,0,1]
        colors[t[np.where(labels_link==-1)[0]]] = [1,0,1,1]
    elif args[0] == 4: ## tipcell
        paint_index = [0]*len(points)
        t = np.where(target==4)[0]
        for i in range(len(t)):
            paint_index[t[i]] = labels_tip[i]*3     
        colors = plt.get_cmap("hsv")(paint_index)
        colors[np.where(target!=4)[0]] = [0,0,0,1]
        colors[t[np.where(labels_tip==-1)[0]]] = [1,0,1,1]

    ###### Remove Overlapped points
    #points2 = points[np.where(target!=0)]
    #colors2 = colors[np.where(target!=0)]

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)

    pcd.colors = o3d.utility.Vector3dVector(colors[:, :3])
    vis = o3d.visualization.VisualizerWithEditing()
    vis.create_window()
    vis.add_geometry(pcd)
    vis.run()
    vis.destroy_window()
    print("")
    picked = vis.get_picked_points()
    for i in picked:
        temp = target[:i].tolist()
        if target[i] == 1:
            forecount = temp.count(1)
            print('label: '+str(labels_joint[forecount]))
        elif target[i] == 2:
            forecount = temp.count(2)
            print('label: '+str(labels_link[forecount]))
        elif target[i] == 4:
            forecount = temp.count(4)
            print('label: '+str(labels_tip[forecount]))
    
    return colors

def ReturnSkel(*args):
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    colors=np.array([[0.5,0.5,0.5,1]]*len(points))
    tag = np.array([[-1]]*len(points))
    if len(args) == 0:
        colors[np.where(target==0)[0]] = [0.92,0.5,0.2,1]   #gel
        colors[np.where(target==1)[0]] = [0,1,0,1]   #joint
        colors[np.where(target==2)[0]] = [0,0,1,1]   #link
        colors[np.where(target==3)[0]] = [0.5,0.5,0.5,1]   #noise
        colors[np.where(target==4)[0]] = [1,0,0,1]   #Tipcell
        tag = [target[i:i + 1] for i in range(0, len(target), 1)]
    elif args[0] == 0: ## Gel
        colors[np.where(target!=0)[0]] = [0,0,0,1]
        colors[np.where(target==0)[0]] = [1,1,0,1]
    elif args[0] == 1: ## joint
        paint_index = [0]*len(points)
        t = np.where(target==1)[0]
        for i in range(len(t)):
            paint_index[t[i]] = labels_joint[i]*3 
        colors = plt.get_cmap("hsv")(paint_index)
        colors[np.where(target!=1)[0]] = [0.5,0.5,0.5,0.5]
        colors[t[np.where(labels_joint==-1)[0]]] = [1,0,1,1]
        tag[np.where(target==1)]=[labels_joint[i:i + 1] for i in range(0, len(labels_joint), 1)]
    elif args[0] == 2: ## link
        paint_index = [0]*len(points)
        t = np.where(target==2)[0]
        for i in range(len(t)):
            paint_index[t[i]] = labels_link[i]*2 
        colors = plt.get_cmap("prism")(paint_index)
        colors[np.where(target!=2)[0]] = [0.5,0.5,0.5,1]
        colors[t[np.where(labels_link==-1)[0]]] = [1,0,1,1]
        tag[np.where(target==2)]=[labels_link[i:i + 1] for i in range(0, len(labels_link), 1)]
    elif args[0] == 4: ## tipcell
        paint_index = [0]*len(points)
        t = np.where(target==4)[0]
        for i in range(len(t)):
            paint_index[t[i]] = labels_tip[i]*3     
        colors = plt.get_cmap("hsv")(paint_index)
        colors[np.where(target!=4)[0]] = [0.5,0.5,0.5,1]
        colors[t[np.where(labels_tip==-1)[0]]] = [1,0,1,1]
        tag[np.where(target==4)]=[labels_tip[i:i + 1] for i in range(0, len(labels_tip), 1)]

    return colors, tag

def VesselEval(chambersData, out_path):
    
    OutputData = []
    PostData = []
    
    for b in range(len(chambersData)):
        global points
        global target
        global points_gel
        global points_joint
        global points_link
        global points_tip
        global points_noise
        global labels_gel
        global labels_joint
        global labels_link
        global labels_tip
        global labels_noise
        global LinkEnds
        
        
        
        chamber = b
        print(chamber)
        data = chambersData[chamber]
        points = data['point']
        ids = data['ids']
        prediction = data['pred']

        
        ####### Set Target Value ######
        target = prediction
        
        

        ######## GET True/ Pred DATA ######
        points_gel = points[target==0]
        points_joint = points[target==1]
        points_link = points[target==2]
        points_tip = points[target==4]
        points_noise = points[target==3]
        
        ##### Cluster / Process Pts ######  
        labels_joint, target = ProcessJoint(points_joint,30,5)
        points_gel = points[target==0]
        points_joint = points[target==1]
        points_link = points[target==2]
        points_tip = points[target==4]
        points_noise = points[target==3]
    
        labels_link, length, endToend, LinkEnds, LinkDensity, temp3, repts, first_labels_link = ProcessLink(points_link, 3, 25,18)
        points_link = repts
        labels_gel = ProcessGel(points_gel)
        labels_tip = ProcessTipcell(points_tip)
    
    
        link_index = np.where(target==2)[0]
        target = np.delete(target,link_index) ### ReConstruct With new Link Points
        points = np.delete(points, link_index,axis=0)
        points = np.concatenate((points,points_link),axis=0)
        target = np.concatenate((target,[2]*len(points_link)))
        
        
        
        link_index = np.where(target==2)[0]
        Points_Density = np.array([[0]]*len(points),dtype=np.float32)
        for i in range(len(LinkDensity)):
            Points_Density[link_index[np.where(labels_link==i)]] = LinkDensity[i]
    
    
        ##### Connectivity Find #######
        
        conn_link, conn_joint, conn_tip, conn_gel = Connectivity(2)
        
        ######## Write to Dic ##########
        dic = {'Joint':[],'Link':[],'Tipcell':[]}
        ## Write joint data
        temp = {}
        for i in range(labels_joint.max()+1):
            temp_idx = np.where(labels_joint == i)
            #print(temp_idx)
            temp[str(i)] = {'points':points_joint[temp_idx]}     
        dic['Joint'] = temp
    
        ## Write link data
        temp = {}
        for i in range(labels_link.max()+1):
            temp_idx = np.where(labels_link == i)
            #print(temp_idx)
            Tortuosity = length[i]/endToend[i]
            temp[str(i)] = {'points':points_link[temp_idx],'Length':length[i],'EndToEnd':endToend[i],'Tortuosity':Tortuosity}
        dic['Link'] = temp
        
        ## Write Tipcell data
        temp = {}
        for i in range(labels_tip.max()+1):
            temp_idx = np.where(labels_tip == i)
            #print(temp_idx)
            temp[str(i)] = {'points':points_tip[temp_idx],}     
        dic['Tipcell'] = temp
    
        dic['Gel'] = points_gel
        dic['Noise'] = points_noise
        
        
        
            ##### Path Find #####
        path = []
        PCs = o3d.geometry.PointCloud()
        PCs.points = o3d.utility.Vector3dVector(points)
    
        pt_chamber0 = PathTracking(dic)
        
        
        for i in range(labels_tip.max()+1):
            path.append([])
            try:
                path[-1] = (pt_chamber0(i))
            except:
                pass
        
        ##### Sproting Length #####
        Sp_Length = []
        for i in range(labels_tip.max()+1):
            Sp_Length.append([])
            if len(path[i]) != 0 :
                temp_Length = 0
                temp_points = path[i][:,:3]
                for j in range(len(temp_points)-1):
                    seg_len = math.sqrt(pow(abs(temp_points[j][0]-temp_points[j+1][0]),2)
                                        + pow(abs(temp_points[j][1]-temp_points[j+1][1]),2)
                                        + pow(abs(temp_points[j][2]-temp_points[j+1][2]),2))
                    if (seg_len>500):
                        seg_len = 0
                    else:
                        temp_Length += seg_len
                Sp_Length[-1] = (temp_Length)
            else:
                Sp_Length[-1] = None
            
        ##### TipCell Level #####
        Tip_Level = []
        for i in range(labels_tip.max()+1):
            Tip_Level.append([])
            if len(path[i]) != 0 :
                point_types = path[i][:,3]
                joint_idx = np.where(point_types==2)[0]
                if len(joint_idx) == 0:
                    count = 0
                else:
                    count = 1
                    for j in range(len(joint_idx)-1):
                        if joint_idx[j] +1 !=  joint_idx[j+1]:
                            count += 1
                Tip_Level[-1] = count
            else:
                Tip_Level[-1] = None
                
                
        #### Directionality #####
        
        def Linearfunc(X,a,b):  ### Set Function type
            return a*X + b
        
        Directionality = [] 
        for i in range(labels_tip.max()+1):
            Directionality.append([])
            if len(path[i]) != 0 :    
                temp_points = path[i][:,:3]
        
                tempx = temp_points[:,0]  ## load points
                tempy = temp_points[:,1]
                tempz = temp_points[:,2]
                
                u = np.arange(len(tempx)) ## make artificial axis for fitting
                
                poptx, pcovx = curve_fit(Linearfunc, u, tempx) ## fit n vs x              
                xnew = Linearfunc(u,poptx[0],poptx[1])
                
                popty, pcovy = curve_fit(Linearfunc, u, tempy) ## fit n vs x             
                ynew = Linearfunc(u,popty[0],popty[1])
    
                
                poptz, pcovz = curve_fit(Linearfunc, u, tempz) ## fit n vs x               
                znew = Linearfunc(u,poptz[0],poptz[1])
    
                VectorLength = math.sqrt(pow(abs(xnew[0]-xnew[-1]),2)
                                         + pow(abs(ynew[0]-ynew[-1]),2)
                                         + pow(abs(znew[0]-znew[-1]),2))
                tempVector = [(xnew[0]-xnew[-1])/VectorLength,
                          (ynew[0]-ynew[-1])/VectorLength,
                          (znew[0]-znew[-1])/VectorLength]
            
                #Directionality[-1] = tempVector
                degree = np.arctan2(tempVector[1],tempVector[0])
                Directionality[-1] = degree
            else:
                Directionality[-1] = None
        
        ####tortuosity####
        tortuosity = []
        for i in range(len(length)):
            tortuosity.append(length[i]/endToend[i])
        
        
        ######## Write to Dic ##########
        
        dic['Directionality'] = Directionality
        dic['TipCell_Level'] = Tip_Level
        dic['Sproting_Length'] = Sp_Length
        dic['path'] = path
        

        
        conn_count = []
        for i in conn_joint:
            i = set(i)
            i = list(i)
            conn_count.append(len(i))
        
        
        
        ####### Show Connectivity #######
        
        connectivity = conn_joint

        
        Parameter_dic = {'Total Length':sum(length),
                         'Link Length':length,
                         'Joint Number':max(labels_joint)+1,
                         'Tortousity':tortuosity,
                         'Tipcell Number':max(labels_tip)+1,
                         'Sprouting Length':Sp_Length,
                         'Sprouting Direction':Directionality,
                         'Tipcell Level':Tip_Level}
        OutputData.append(Parameter_dic)

        Post_dic = {'Total Length': Parameter_dic['Total Length'],
                    'Link Length':np.array([k for k in (Parameter_dic['Link Length'][1:]) if k!=None]).mean(),
                    'Joint Number':Parameter_dic['Joint Number'],
                    'Tortousity':np.array(Parameter_dic['Tortousity']).mean(),
                    'Tipcell Number':Parameter_dic['Tipcell Number'],
                    'Sprouting Length':np.array([k for k in (Parameter_dic['Sprouting Length']) if k!=None]).mean(),
                    'Sprouting Direction':np.array([k for k in (Parameter_dic['Sprouting Direction']) if k!=None]).mean(),
                    'Tipcell Level':np.array([k for k in (Parameter_dic['Tipcell Level']) if k!=None]).mean()}
        PostData.append(Post_dic)
    

    with open('checkpoints/'+out_path+'/'+'Result.csv',"w") as fw:
         wr = csv.writer(fw)
         wr.writerow(Post_dic.keys())
         for i in PostData:
             wr.writerow(i.values())

    
    return OutputData
          
            