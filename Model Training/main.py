
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
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
from data import PointDataset
from model import PointNet, DGCNN, PointNet2, ADGCNN, SPHADGCNN
import numpy as np
from torch.utils.data import DataLoader
from util import cal_loss_pointnet, cal_loss, IOStream, str2bool, IOU, ROC_cal, Merge_Vote, Perturb_remove, weights_init
import sklearn.metrics as metrics
from sklearn.model_selection import KFold
import pickle

def _init_():
    if not os.path.exists('checkpoints'):
        os.makedirs('checkpoints')
    if not os.path.exists('checkpoints/'+args.exp_name):
        os.makedirs('checkpoints/'+args.exp_name)
    if not os.path.exists('checkpoints/'+args.exp_name+'/'+'models'):
        os.makedirs('checkpoints/'+args.exp_name+'/'+'models')
    os.system('cp main.py checkpoints'+'/'+args.exp_name+'/'+'main.py.backup')
    os.system('cp model.py checkpoints' + '/' + args.exp_name + '/' + 'model.py.backup')
    os.system('cp util.py checkpoints' + '/' + args.exp_name + '/' + 'util.py.backup')
    os.system('cp data.py checkpoints' + '/' + args.exp_name + '/' + 'data.py.backup')

def train(args, io):
    #Sampling for class imbalance
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    DATA_DIR = os.path.join(BASE_DIR, 'data')

    train_loader = DataLoader(PointDataset(partition='train', num_points=args.num_points), num_workers=4,
                              batch_size=args.batch_size, shuffle=False, drop_last=True)#, sampler=sampler)
    test_loader = DataLoader(PointDataset(partition='test', num_points=args.num_points), num_workers=4,
                             batch_size=args.test_batch_size, shuffle=True, drop_last=False)

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
    model = model.apply(weights_init)
    print("Using", torch.cuda.device_count(), "GPUs")

    if args.use_sgd:
        print("Use SGD")
        opt = optim.SGD(model.parameters(), lr=args.lr*100, momentum=args.momentum, weight_decay=1e-4)
    else:
        print("Use Adam")
        opt = optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-4)

    scheduler = CosineAnnealingLR(opt, args.epochs, eta_min=args.lr)
    
    if (args.model == 'pointnet') or (args.model == 'pointnet2'):
        criterion = cal_loss_pointnet(device,  mat_diff_loss_scale = 0.001)
    else:
        criterion = cal_loss(device)

    best_test = 0
    train_history=[]
    test_history=[]
    for epoch in range(args.epochs):
        ####################
        # Train
        ####################
        train_loss = 0.0
        count = 0.0
        model.train()
        train_pred = []
        train_true = []
        for data, label, ids, norm_info in train_loader:
            data, label = data.float().to(device), label.to(device)
            data = data.permute(0, 2, 1)
            batch_size = data.size()[0]
            opt.zero_grad()
            if (args.model == 'pointnet') or (args.model == 'pointnet2'):
                logits, trans_feat = model(data)
            else:
                logits = model(data)
            logits = logits.permute(0,2,1).contiguous() # (batch_size, seg_num_all, num_points) -> (batch_size, num_points, seg_num_all)
            if (args.model == 'pointnet') or (args.model == 'pointnet2'):
                loss = criterion(logits.view(-1, 5), label.view(-1,1).squeeze(), trans_feat)
            else:
                loss = criterion(logits.view(-1, 5), label.view(-1,1).squeeze())
            loss.backward()
            opt.step()
            scheduler.step()
            pred = logits.max(dim=2)[1] #(batch_size, num_points)
            count += batch_size 
            train_loss += loss.item() * batch_size
            label_np = label.cpu().numpy() #(batch_size, num_points)
            pred_np = pred.detach().cpu().numpy() 
            train_true.append(label_np.reshape(-1))
            train_pred.append(pred_np.reshape(-1))
            
        train_true = np.concatenate(train_true)
        train_pred = np.concatenate(train_pred)
        outstr = 'Train %d, loss: %.6f, train acc: %.6f, train avg acc: %.6f, avg IOU: %.6f' % (epoch,
                                                                                 train_loss*1.0/count,
                                                                                 metrics.accuracy_score(
                                                                                     train_true, train_pred),
                                                                                 metrics.balanced_accuracy_score(
                                                                                     train_true, train_pred),
                                                                                     IOU(train_true,train_pred, mean=True))
        train_history.append(outstr)
        io.cprint(outstr)

        ####################
        # Test
        ####################
        test_loss = 0.0
        count = 0.0
        model.eval()
        
        test_pred = []
        test_true = []

        for data, label, ids, norm_info in test_loader:
            data, label, ids, norm_info = data.to(device), label.to(device), ids.to(device).squeeze(), norm_info.to(device)
            test_datas.append(data.view(-1,3).cpu().detach().numpy())
            data = data.permute(0, 2, 1)
            batch_size = data.size()[0]
            if args.model == ('pointnet') or (args.model == 'pointnet2'):
                logits, trans_feat = model(data)
            else:
                logits = model(data)
            logits = logits.permute(0,2,1).contiguous()
            if (args.model == 'pointnet') or (args.model == 'pointnet2'):
                loss = criterion(logits.view(-1, 5), label.view(-1,1).squeeze(), trans_feat)
            else:
                loss = criterion(logits.view(-1, 5), label.view(-1,1).squeeze())
            
            pred = logits.max(dim=2)[1]
            count += batch_size
            test_loss += loss.item() * batch_size
            label_np = label.cpu().numpy() #(batch_size, num_points)
            pred_np = pred.detach().cpu().numpy() 
            test_true.append(label_np.reshape(-1))
            test_pred.append(pred_np.reshape(-1))

        test_true = np.concatenate(test_true)
        test_pred = np.concatenate(test_pred) 
        test_acc = metrics.accuracy_score(test_true, test_pred)
        avg_per_class_acc = metrics.balanced_accuracy_score(test_true, test_pred)

        test_datas = np.concatenate(test_datas, axis=0).squeeze() #(total_batch_size*num_points, 3)
        test_true = np.concatenate(test_true) #(total_batch_size*num_points): 1-D array
        test_ids = np.concatenate(test_ids) #(total_batch_size*num_points): 1-D array
        test_logits = np.concatenate(test_logits, axis=0).squeeze() #(total_batch_size*num_points, seg_num_all)
        test_norm_info = np.concatenate(test_norm_info, axis=0) #(tot_batch_size*num_points, 4) 

        avg_iou=IOU(test_true,test_pred,mean=True)  
    
        outstr = 'Test %d, loss: %.6f, test acc: %.6f, test avg acc: %.6f, avg IOU: %.6f' % (epoch,
                                                                              test_loss*1.0/count,
                                                                              test_acc,
                                                                              avg_per_class_acc,
                                                                              avg_iou)
        test_history.append(outstr)
        io.cprint(outstr)
        if test_acc >= best_test:
            best_test = test_acc
            torch.save(model.state_dict(), 'checkpoints/%s/models/model.t7' % args.exp_name)
    
    with open('checkpoints/%s/models/history.pickle'%args.exp_name,'wb') as f:
        history_data={'train history':train_history,'test history':test_history}
        pickle.dump(history_data,f, pickle.HIGHEST_PROTOCOL)


def train_kfold(args, io):
    # later should be replaced thing 
    device = torch.device("cuda" if args.cuda else "cpu")
    
    # KFold generation 
    tot_dataset = PointDataset(partition='train', num_points=args.num_points)
    kf = KFold(n_splits = args.fold_num, random_state = 0, shuffle = True)
    tot_oa_fold_pred = []
    tot_oa_fold_true = []
    for fold, (train_ind, val_ind) in enumerate(kf.split(range(len(tot_dataset)))):
        print("=========================================K-FOLD: {}========================================".format(fold))
        train_sampler = torch.utils.data.sampler.SubsetRandomSampler(train_ind)
        valid_sampler = torch.utils.data.sampler.SubsetRandomSampler(val_ind)
        
        train_loader = DataLoader(PointDataset(partition='train', num_points=args.num_points, augmentation = True), 
                        num_workers=4, batch_size=args.batch_size, shuffle=False, drop_last=False)
        val_loader = DataLoader(PointDataset(partition='train', num_points=args.num_points, augmentation = False), 
                        num_workers=4, batch_size=args.test_batch_size, shuffle=False, drop_last=False)

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

        model = nn.DataParallel(model)
        model = model.apply(weights_init)
        print("Using", torch.cuda.device_count(), "GPUs")

        if args.use_sgd:
            print("Use SGD")
            opt = optim.SGD(model.parameters(), lr=args.lr*100, momentum=args.momentum, weight_decay=1e-4)
        else:
            print("Use Adam")
            opt = optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-4)
        scheduler = CosineAnnealingLR(opt, args.epochs, eta_min=args.lr)

        if (args.model == 'pointnet') or (args.model == 'pointnet2'):
            criterion = cal_loss_pointnet(device,  mat_diff_loss_scale = 0.001)
        else:
            criterion = cal_loss(device)

        best_test = 0
        tmp_oa_fold_pred = []
        tmp_oa_fold_true = []
        for epoch in range(args.epochs): 
            ####################
            # Train
            ####################
            train_loss = 0.0
            count = 0.0
            model.train()
            train_pred = []
            train_true = []
            for data, label, ids, norm_info in train_loader:
                data, label = data.float().to(device), label.to(device)
                data = data.permute(0, 2, 1) # [batch_size, num_points, 3] -> [batch_size, 3, num_points]
                batch_size = data.size()[0]
                opt.zero_grad()
                if (args.model == 'pointnet') or (args.model == 'pointnet2'):
                    logits, trans_feat = model(data)
                else:
                    logits = model(data)
                logits = logits.permute(0,2,1).contiguous() # (batch_size, seg_num_all, num_points) -> (batch_size, num_points, seg_num_all)
                if (args.model == 'pointnet') or (args.model == 'pointnet2'):
                    loss = criterion(logits.view(-1, 5), label.view(-1,1).squeeze(), trans_feat)
                else:
                    loss = criterion(logits.view(-1, 5), label.view(-1,1).squeeze())
                loss.backward()
                opt.step()
                scheduler.step()
                pred = logits.max(dim=2)[1] #(batch_size, num_points)
                count += batch_size 
                train_loss += loss.item() * batch_size
                label_np = label.cpu().numpy() #(batch_size, num_points)
                pred_np = pred.detach().cpu().numpy() 
                train_true.append(label_np.reshape(-1))
                train_pred.append(pred_np.reshape(-1))
            
            train_true = np.concatenate(train_true)
            train_pred = np.concatenate(train_pred)
            outstr = 'Train %d, loss: %.6f, train acc: %.6f, train avg acc: %.6f, avg IOU: %.6f' % (epoch,
                                                                                 train_loss*1.0/count,
                                                                                 metrics.accuracy_score(
                                                                                     train_true, train_pred),
                                                                                 metrics.balanced_accuracy_score(
                                                                                     train_true, train_pred),
                                                                                     IOU(train_true,train_pred, mean=True))
            io.cprint(outstr)
            
            ####################
            # Validation
            ####################
            test_loss = 0.0
            count = 0.0
            model.eval()
            
            test_pred = []
            test_true = []
            for data, label, ids, norm_info in val_loader:
                data, label = data.float().to(device), label.to(device)
                data = data.permute(0, 2, 1)
                batch_size = data.size()[0]
                if args.model == ('pointnet') or (args.model == 'pointnet2'):
                    logits, trans_feat = model(data)
                else:
                    logits = model(data)
                logits = logits.permute(0,2,1).contiguous()
                if (args.model == 'pointnet') or (args.model == 'pointnet2'):
                    loss = criterion(logits.view(-1, 5), label.view(-1,1).squeeze(), trans_feat)
                else:
                    loss = criterion(logits.view(-1, 5), label.view(-1,1).squeeze())
                
                pred = logits.max(dim=2)[1]
                count += batch_size
                test_loss += loss.item() * batch_size
                label_np = label.cpu().numpy() #(batch_size, num_points)
                pred_np = pred.detach().cpu().numpy() 
                test_true.append(label_np.reshape(-1))
                test_pred.append(pred_np.reshape(-1))
            
            test_true = np.concatenate(test_true)
            test_pred = np.concatenate(test_pred) 
            test_acc = metrics.accuracy_score(test_true, test_pred)
            avg_per_class_acc = metrics.balanced_accuracy_score(test_true, test_pred)
            avg_iou=IOU(test_true,test_pred,mean=True) 
                
            outstr = 'Test %d, loss: %.6f, test acc: %.6f, test avg acc: %.6f, avg IOU: %.6f' % (epoch,
                                                                              test_loss*1.0/count,
                                                                              test_acc,
                                                                              avg_per_class_acc,
                                                                              avg_iou)
            io.cprint(outstr)
            if test_acc > best_test: # Saving best overall accuracy
                best_test = test_acc
                tmp_oa_fold_pred = test_pred
                tmp_oa_fold_true = test_true
                torch.save(model.state_dict(), 'checkpoints/{}/models/model_oa_fold{}.t7'.format(args.exp_name, fold))
            
        tot_oa_fold_pred = np.concatenate((tot_oa_fold_pred, tmp_oa_fold_pred), axis = 0)
        tot_oa_fold_true = np.concatenate((tot_oa_fold_true, tmp_oa_fold_true), axis = 0)

    # Micro-averaging for final k-fold training evaluation 
    fold_acc = metrics.accuracy_score(tot_oa_fold_true, tot_oa_fold_pred)
    fold_per_class_acc = metrics.balanced_accuracy_score(tot_oa_fold_true, tot_oa_fold_pred)
    fold_iou = IOU(tot_oa_fold_true, tot_oa_fold_pred, mean=True)
    outstr = 'Final oAcc Evaluation ({}-Fold): oAcc: {}, mAcc: {}, mIOU: {}'.format(args.fold_num,
                                                                              fold_acc,
                                                                              fold_per_class_acc,
                                                                              fold_iou)
    io.cprint(outstr)


def test(args, io):
    test_loader = DataLoader(PointDataset(partition='test', num_points=args.num_points),
                             batch_size=args.test_batch_size, shuffle=True, drop_last=False)
    
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
    test_acc = 0.0
    count = 0.0
    test_datas = []
    test_logits = []
    test_true = []
    test_ids = []
    test_norm_info = []
    
    for data, label, ids, norm_info in test_loader:
        data, label, ids, norm_info = data.to(device), label.to(device).squeeze(), ids.to(device).squeeze(), norm_info.to(device)
        test_datas.append(data.view(-1,3).cpu().detach().numpy())
        data = data.permute(0, 2, 1) #(batch_size, num_points, 3) -> (batch_size, 3, num_points)
        batch_size = data.size()[0]
        if (args.model == 'pointnet') or (args.model == 'pointnet2'):
            logits, trans_feat = model(data)
        else:
            logits = model(data)
        logits = logits.permute(0,2,1).contiguous() # (batch_size, seg_num_all, num_points) -> (batch_size, num_points, seg_num_all)
        test_logits.append(logits.view(-1,5).cpu().detach().numpy())
        label_np = label.cpu().numpy() #(batch_size, num_points) 
        test_true.append(label_np.reshape(-1)) 
        ids = ids.repeat_interleave(args.num_points)
        test_ids.append(ids.cpu().numpy().reshape(-1))
        norm_info = torch.repeat_interleave(norm_info, args.num_points, dim=0)
        test_norm_info.append(norm_info.cpu().detach().numpy())
              
    final_data = np.concatenate(test_datas, axis=0).squeeze() #(total_batch_size*num_points, 3)
    final_true = np.concatenate(test_true) #(total_batch_size*num_points): 1-D array
    final_ids = np.concatenate(test_ids) #(total_batch_size*num_points): 1-D array
    final_logits = np.concatenate(test_logits, axis=0).squeeze() #(total_batch_size*num_points, seg_num_all)
    final_pred = final_logits.argmax(axis=1)

    test_acc = metrics.accuracy_score(final_true, final_pred)
    avg_per_class_acc = metrics.balanced_accuracy_score(final_true, final_pred)
    matrix=metrics.confusion_matrix(final_true, final_pred)
    class_acc=matrix.diagonal()/matrix.sum(axis=1)
    avg_per_class_iou=IOU(final_true,final_pred, mean=True)
    class_iou=IOU(final_true, final_pred, mean=False)
    roc_auc = ROC_cal(args, final_true, final_logits)

    str1='     class 1    class 2    class 3    class 4    class 5'
    str2='Acc:  %.4f     %.4f     %.4f     %.4f     %.4f'%(class_acc[0],class_acc[1],class_acc[2],class_acc[3],class_acc[4])
    str3='IOU:  %.4f     %.4f     %.4f     %.4f     %.4f'%(class_iou[0],class_iou[1],class_iou[2],class_iou[3],class_iou[4])
    str4='ROC:  %.4f     %.4f     %.4f     %.4f     %.4f'%(roc_auc[0],roc_auc[1],roc_auc[2],roc_auc[3],roc_auc[4])

    outstr = 'Test :: test acc: %.6f, test avg acc: %.6f, test avg iou: %.6f, roc area(micro): %.6f, roc area(macro): %.6f'%(test_acc, avg_per_class_acc, avg_per_class_iou, roc_auc['micro'], roc_auc['macro'])
    io.cprint(outstr)
    io.cprint(str1, str2, str3, str4, sep='\n')



if __name__ == "__main__":
    # Training settings
    parser = argparse.ArgumentParser(description='Skeleton Point Cloud Segmentation')
    parser.add_argument('--exp_name', type=str, default='exp', metavar='N',
                        help='Name of the experiment')
    parser.add_argument('--model', type=str, default='dgcnn', metavar='N',
                        choices=['pointnet', 'dgcnn','pointnet2', 'adgcnn', 'sphadgcnn'],
                        help='Model to use, [pointnet, dgcnn, pointnet2, adgcnn, sphadgcnn]')
    parser.add_argument('--dataset', type=str, default='PointDataset', metavar='N',
                        choices=['PointDataset'])
    parser.add_argument('--batch_size', type=int, default=32, metavar='batch_size',
                        help='Size of batch)')
    parser.add_argument('--test_batch_size', type=int, default=16, metavar='batch_size',
                        help='Size of batch)')
    parser.add_argument('--epochs', type=int, default=250, metavar='N',
                        help='number of episode to train ')
    parser.add_argument('--use_sgd', type=str2bool, nargs='?',const=True, default=True,
                        help='Use SGD')
    parser.add_argument('--lr', type=float, default=0.001, metavar='LR',
                        help='learning rate (default: 0.001, 0.1 if using sgd)')
    parser.add_argument('--momentum', type=float, default=0.9, metavar='M',
                        help='SGD momentum (default: 0.9)')
    parser.add_argument('--no_cuda', type=str2bool, nargs='?',const=True, default=False,
                        help='enables CUDA training')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--eval', type=str2bool, nargs='?',const=True, default=False,
                        help='evaluate the model')
    parser.add_argument('--num_points', type=int, default=2400,
                        help='num of points to use')
    parser.add_argument('--dropout', type=float, default=0.5,
                        help='dropout rate')
    parser.add_argument('--emb_dims', type=int, default=1024, metavar='N',
                        help='Dimension of embeddings')
    parser.add_argument('--k', type=int, default=20, metavar='N',
                        help='Num of nearest neighbors to use')
    parser.add_argument('--model_path', type=str, default='', metavar='N',
                        help='Pretrained model path')
    parser.add_argument('--kfold', type=str2bool, nargs='?',const=True, default=False,
                        help='Train with k-fold')
    parser.add_argument('--fold_num', type=int, default=5,
                        help='number of fold for cross-validation')                
    args = parser.parse_args()

    _init_()

    io = IOStream('checkpoints/' + args.exp_name + '/run.log')
    io.cprint(str(args))

    args.cuda = not args.no_cuda and torch.cuda.is_available()
    torch.manual_seed(args.seed)
    if args.cuda:
        io.cprint(
            'Using GPU : ' + str(torch.cuda.current_device()) + ' from ' + str(torch.cuda.device_count()) + ' devices')
        torch.cuda.manual_seed(args.seed)
    else:
        io.cprint('Using CPU')

    if not args.eval:
        if args.kfold:
            train_kfold(args, io)
        else:
            train(args, io)
    else:
        test(args, io)
