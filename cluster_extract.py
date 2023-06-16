import sys
import os
import argparse
import utility as utils
import torch
import torch.nn as nn
from torchvision import datasets, transforms
from torch.optim.lr_scheduler import StepLR

from torch.utils.data import DataLoader
from torch import optim
import time
from torch.nn.modules.loss import _Loss
import datetime
import torch.nn.functional as F
import numpy as np
import pickle

from ucf101_capsules import CapsNet
from ucf101_dataloader import UCF101DataLoader
from kmeans import batch_KMeans
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA



if __name__ == '__main__':

    # ----------------------------------
    
    cluster_fp = 'files/NAME_OF_CLUSTER_DATA.npy'
    
    model_file_path = './trained/active_learning/checkpoints_ucf101_capsules_i3d/NAME_OF_TRAINED_MODEL.pth'

    load_file_fp = 'training_annots.pkl'
    
    save_file_fp = 'files/clusterextract.pkl'
    
    # ----------------------------------


    USE_CUDA = True if torch.cuda.is_available() else False
    TRAIN_BATCH_SIZE = 6
    VAL_BATCH_SIZE = 6
    N_EPOCHS = 30
    LR = 0.001
    LR_step_size = 60
    LR_SCHEDULER = False     # True  
    IS_MASKING = True       # False
    pretrained_load = False   # False
    load_previous_weights = True # True
    HYBRID = True       # Enable hybrid vid/frame mode 
    
    MODEL_TYPE = 'i3d'  
    vid_size = [224, 224]
    
    # Initialize cluster object
    parser = argparse.ArgumentParser(description='Deep Clustering Network')
    parser.add_argument('--latent_dim', type=int, default=408,
                        help='latent space dimension')
    parser.add_argument('--n-clusters', type=int, default=5,
                        help='number of clusters in the latent space')
    parser.add_argument('--n-jobs', type=int, default=4,
                        help='number of jobs to run in parallel')
    parser.add_argument('--percent', type=int, default=0,
                        help='percent of frames')
    parser.add_argument('--percent_vids', type=int, default=0,
                        help='percent of vids')
    args = parser.parse_args()
    
    percent = str(args.percent)
    percent_vids = str(args.percent_vids)    
        
    print("="*40)
    print("TRAIN_BATCH_SIZE: ", TRAIN_BATCH_SIZE)
    print("VAL_BATCH_SIZE: ", VAL_BATCH_SIZE)
    print("N_EPOCHS: ", N_EPOCHS)
    print("Learning Rate: ", LR)
    print("Masking mode: ", IS_MASKING)
    print("pretrained_load: ", pretrained_load)
    print("load previous weights: ", load_previous_weights)
    print("Hybrid mode: ", HYBRID)
    print("Percent: ", percent)
    print("Percent vids: ", percent_vids)
    print("-"*40)
    
    evalset = UCF101DataLoader('train',vid_size, TRAIN_BATCH_SIZE, load_file_fp = load_file_fp, percent='100', percent_vids='100', use_random_start_frame=False)
    
    eval_data_loader = DataLoader(
        dataset=evalset,
        batch_size=TRAIN_BATCH_SIZE,
        num_workers=5,
        shuffle=False
    )
    
    model = CapsNet(pretrained_load=pretrained_load)
    if load_previous_weights:
        model.load_previous_weights(model_file_path)

    if USE_CUDA:
        model = model.cuda()
    
    beta=3e-8

    kmeans = batch_KMeans(args)
    print("KMeans object prepared")

    # Prepare initial cluster from pretraining
    model.eval()
    model.training = False 
    
    cluster_centers = np.load(open(cluster_fp,'rb'))
    kmeans.assign_cluster(cluster_centers)
    print("KMeans cluster loaded from ", cluster_fp)
    
    y_test = []
    y_pred = []
    latent_val = []
    vid_name_cat = {}
        
    with torch.no_grad():
        for k in range(2):
            for batch_idx, minibatch in enumerate(eval_data_loader):
                data = minibatch['data']
                data = data.type(torch.cuda.FloatTensor)
                action = minibatch['action']
                action = action.cuda()
                _, _, latent_X = model(data, action, latent=True)
                latent_X = latent_X.detach().cpu().numpy()
                for i in range(latent_X.shape[0]):
                    if len(latent_X[i]) == 408:
                        k_mean_pred = kmeans.update_assign(latent_X[i:i+1,:])[0]
                        vn = minibatch['vname'][i]
                        if vn in list(vid_name_cat.keys()):
                            vid_name_cat[vn]['pred'].append(k_mean_pred)
                            vid_name_cat[vn]['latent'].append(latent_X[i])
                        else:
                            vid_name_cat[vn] = {'pred':[k_mean_pred],
                                                'latent':[latent_X[i]]}
    
    latent_val = []
    y_pred = []
    vid_names = []
    for k,v in vid_name_cat.items():
        preds = v['pred']
        max_pred = np.bincount(preds).argmax()
        lat_arg = np.where(preds==max_pred)[0][0]
        vid_name_cat[k]['pred'] = max_pred
        vid_name_cat[k]['latent'] = vid_name_cat[k]['latent'][lat_arg]
        latent_val.append(vid_name_cat[k]['latent'])
        y_pred.append(max_pred)
        vid_names.append(k)
    
    latent_val = np.array(latent_val).reshape(-1,408)
    y_pred = np.array(y_pred)
    
    pca_tool = PCA(n_components=2)
    pca_tool = pca_tool.fit(latent_val)
    reduced_data = pca_tool.transform(latent_val)
    centroids = pca_tool.transform(kmeans.clusters)
    
    with open(save_file_fp,'wb') as wid:
        pickle.dump(vid_name_cat, wid, protocol=4)
    print("Data saved to: ", save_file_fp)
