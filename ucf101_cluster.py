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

from ucf101_capsules import CapsNet
from ucf101_dataloader import UCF101DataLoader
from kmeans import batch_KMeans

vid_name_cat = {}

""" Compute the Equation (5) in the original paper on a data batch """

class ClusterLoss(nn.Module):
    def __init__(self, beta):
        super(ClusterLoss, self).__init__()
        self.beta = beta

    def forward(self, latent_X, cluster_id, clusters):
        batch_size = latent_X.size()[0]

        # Regularization term on clustering
        dist_loss = torch.tensor(0.).cuda()
        for i in range(batch_size):
                diff_vec = latent_X[i] - clusters[cluster_id[i]]
                sample_dist_loss = torch.matmul(diff_vec.view(1, -1),
                                                diff_vec.view(-1, 1))
                dist_loss += 0.5 * self.beta * torch.squeeze(sample_dist_loss)

        return dist_loss


class DiceLoss(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(DiceLoss, self).__init__()

    def forward(self, inputs, targets, smooth=1):
        inputs = F.sigmoid(inputs)       

        inputs = inputs.view(-1)
        targets = targets.view(-1)

        intersection = (inputs * targets).sum()                            
        dice = (2.*intersection + smooth)/(inputs.sum() + targets.sum() + smooth)  

        return 1 - dice

 
class SpreadLoss(_Loss):

    def __init__(self, m_min=0.2, m_max=0.9, num_class=24):
        super(SpreadLoss, self).__init__()
        self.m_min = m_min
        self.m_max = m_max
        self.num_class = num_class

    def forward(self, x, target, r):
        target = target.long()
        b, E = x.shape
        assert E == self.num_class
        margin = self.m_min + r

        at = torch.cuda.FloatTensor(b).fill_(0)
        for i, lb in enumerate(target):
            at[i] = x[i][lb]
        at = at.view(b, 1).repeat(1, E)

        zeros = x.new_zeros(x.shape)
        absloss = torch.max(.9 - (at - x), zeros)
        loss = torch.max(margin - (at - x), zeros)
        absloss = absloss ** 2
        loss = loss ** 2
        absloss = absloss.sum() / b - .9 ** 2
        loss = loss.sum() / b - margin ** 2

        return loss, absloss


class CapsuleLoss(nn.Module):
    def __init__(self):
        super(CapsuleLoss, self).__init__()

    def forward(self, labels, classes):
        left = F.relu(0.9 - classes, inplace=True) ** 2
        right = F.relu(classes - 0.1, inplace=True) ** 2

        margin_loss = labels * left + 0.5 * (1. - labels) * right
        margin_loss = margin_loss.sum()

        return margin_loss


def get_accuracy(predicted_actor, actor):
    max, prediction = torch.max(predicted_actor, 1)
    prediction = prediction.view(-1, 1)
    actor = actor.view(-1, 1)
    correct = torch.sum(actor == prediction.float()).item()
    accuracy = correct / float(prediction.shape[0])
    return accuracy

def model_interface(minibatch, criterions, r=0, masking=False, kmeans=[], latent=False):
    data = minibatch['data']
    action = minibatch['action']
    action = action.cuda()
    segmentation = minibatch['segmentation']
    segmentation = segmentation
    vname = minibatch['vname']
    data = data.type(torch.cuda.FloatTensor)
    if latent:
        output, predicted_action, latent_x = model(data, action, latent=latent)
    else:
        output, predicted_action = model(data, action, latent=latent)
    
    if masking:
        mask_cls = minibatch['mask_cls']
        mask_cls = mask_cls.cuda()
        output = output * mask_cls
    
    criterion5 = criterions[0]
    class_loss, abs_class_loss = criterion5(predicted_action, action, r)

    criterion1 = nn.BCEWithLogitsLoss(size_average=True)
    loss1 = criterion1(output, segmentation.float().cuda())
    
    criterion2 = criterions[1]
    loss2 = criterion2(output, segmentation.float().cuda())
    
    if latent:        
        with torch.no_grad():
            _, _, latent_x_np = model(data, action, latent=latent)
            latent_x_np = latent_x_np.cpu().numpy()
            cluster_id = []
            for v in range(len(vname)):
                cluster_id.append(vid_name_cat[vname[v]])
            cluster_id = np.array(cluster_id)
            
        #print("Predict: ", kmeans.model.predict(latent_x_np))
        #print("KMeans: ", cluster_id)
        #print("Action: ", action)
        if model.training:
            elem_count = np.bincount(cluster_id, minlength=kmeans.n_clusters)
            for k in range(kmeans.n_clusters):
                # avoid empty slicing
                if elem_count[k] == 0:
                    continue
                kmeans.update_cluster(latent_x_np[cluster_id == k], k)
        clusters = torch.FloatTensor(kmeans.clusters).cuda()

        criterion4 = criterions[3]
        loss4 = criterion4(latent_x, cluster_id, clusters)
        
    else:
        loss4 = -1

    seg_loss = loss1 + loss2
    if latent:
        total_loss = seg_loss + class_loss + loss4
    else:
        total_loss = seg_loss + class_loss

    return (output, predicted_action, segmentation, action, total_loss, seg_loss, class_loss, loss4)


def train(model, train_loader, optimizer, criterions, epoch, r, save_path, kmeans=[], short=False, masking=False, m_info=None, latent=False, min_epoch_to_save=4):
    start_time = time.time()
    steps = len(train_loader)
    model.train(mode=True)
    model.training = True
    total_loss = []
    accuracy = []
    seg_loss = []
    class_loss = []
    class_loss_sent = []
    accuracy_sent = []
    print('epoch  step    loss   seg    class  accuracy')
    
    start_time = time.time()
    for batch_id, minibatch in enumerate(train_loader):
        if short:
            if batch_id > 40:
                break

        optimizer.zero_grad()

        output, predicted_action, segmentation, action, loss, s_loss, c_loss, cluster_loss = model_interface(minibatch, criterions, r,
                                                                                                                masking=masking, kmeans=kmeans, latent=latent)
        loss.backward()
        optimizer.step()

        total_loss.append(loss.item())
        seg_loss.append(s_loss.item())
        class_loss.append(c_loss.item())
        accuracy.append(get_accuracy(predicted_action, action))

        report_interval = 20
        if (batch_id + 1) % report_interval == 0:
            r_total = np.array(total_loss).mean()
            r_seg = np.array(seg_loss).mean()
            r_class = np.array(class_loss).mean()
            r_acc = np.array(accuracy).mean()
            print('%d/%d  %d/%d  %.3f  %.3f  %.3f  %.3f  %.3f'%(epoch,N_EPOCHS,batch_id + 1,steps,r_total,r_seg,r_class,cluster_loss,r_acc))
            sys.stdout.flush()
    
    r_total = np.array(total_loss).mean()
    r_seg = np.array(seg_loss).mean()
    r_class = np.array(class_loss).mean()
    r_acc = np.array(accuracy).mean()
    print('%d/%d  %d/%d  %.3f  %.3f  %.3f  %.3f  %.3f' %(epoch, N_EPOCHS, batch_id + 1, steps, r_total, r_seg, r_class,cluster_loss, r_acc))
    sys.stdout.flush()
    
    end_time = time.time()
    train_epoch_time = end_time - start_time
    print("Training time: ", train_epoch_time)
    
    if epoch > min_epoch_to_save and epoch % 2 == 0:
        torch.save(model.state_dict(), save_path)
        print('saved weights to ', save_path)
        
        if latent:
            cluster_fp = 'files/clusters-5_{}_e{}.npy'.format(m_info, epoch)
            with open(cluster_fp, 'wb') as wid:
                np.save(wid, kmeans.clusters)
            print("KMeans cluster saved at: ", cluster_fp)
    

def validate(model, val_data_loader, criterion, epoch, kmeans=[], short=False):
    steps = len(val_data_loader)
    model.eval()
    model.training = False
    total_loss = []
    accuracy = []
    seg_loss = []
    class_loss = []
    total_IOU = 0
    validiou = 0
    print('validating...')
    start_time = time.time()
    
    with torch.no_grad():
        
        for batch_id, minibatch in enumerate(val_data_loader):
            if short:
                if batch_id > 40:
                    break
            
            output, predicted_action, segmentation, action, loss, s_loss, c_loss, cluster_loss_out = model_interface(minibatch, criterion, r, 
                                                                                                    masking=False, kmeans=kmeans, latent=False)
            total_loss.append(loss.item())
            seg_loss.append(s_loss.item())
            class_loss.append(c_loss.item())
            accuracy.append(get_accuracy(predicted_action, action))

            maskout = output.cpu()
            maskout_np = maskout.data.numpy()

            # use threshold to make mask binary
            maskout_np[maskout_np > 0] = 1
            maskout_np[maskout_np < 1] = 0

            truth_np = segmentation.cpu().data.numpy()
            for a in range(minibatch['data'].shape[0]):
                iou = utils.IOU2(truth_np[a], maskout_np[a])
                if iou == iou:
                    total_IOU += iou
                    validiou += 1
                else:
                    print('bad IOU')
    
    val_epoch_time = time.time() - start_time
    print("Time taken: ", val_epoch_time)
    
    r_total = np.array(total_loss).mean()
    r_seg = np.array(seg_loss).mean()
    r_class = np.array(class_loss).mean()
    r_acc = np.array(accuracy).mean()
    r_cluster = -1
    average_IOU = total_IOU / validiou
    print('Validation %d  %.3f  %.3f  %.3f  %.3f  %.3f  IOU %.3f' %(epoch, r_total, r_seg, r_class, r_acc, r_cluster, average_IOU))
    sys.stdout.flush()

if __name__ == '__main__':

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
    beta= 1e-6 #3e-5 #0.01    
    
        
    # ------------------------------------
    
    load_file_fp = 'ucf101_annotation_cluster_5.0.pkl'
    
    m_info = percent_vids+'V'+percent + 'F_prune'+str(int(percent_vids)-5)+'V5F'
    
    training_specs = 'Hybrid_'+percent_vids+'V'+percent + 'F_prune'+str(int(percent_vids)-5)+'V5F_Cluster_beta-'+str(beta)
    
    # ------------------------------------
    

    USE_CUDA = True if torch.cuda.is_available() else False
    TRAIN_BATCH_SIZE = 6
    VAL_BATCH_SIZE = 6
    N_EPOCHS = 40
    LR = 0.00005 
    LR_step_size = 20
    LR_SCHEDULER = False     
    IS_MASKING = True       
    pretrained_load = True   
    load_previous_weights = False 
    HYBRID = True       
    vid_size = [224, 224]
    MODEL_TYPE = 'i3d'  # 'c3d' 'i3d'
    
    print("="*40)
    print("Percent: ", percent)
    print("Percent vids: ", percent_vids)
    print("TRAIN_BATCH_SIZE: ", TRAIN_BATCH_SIZE)
    print("VAL_BATCH_SIZE: ", VAL_BATCH_SIZE)
    print("N_EPOCHS: ", N_EPOCHS)
    print("Learning Rate: ", LR)
    print("Masking mode: ", IS_MASKING)
    print("pretrained_load: ", pretrained_load)
    print("load previous weights: ", load_previous_weights)
    print("Hybrid mode: ", HYBRID)
    print("-"*40)
    
    trainset = UCF101DataLoader('train', vid_size, TRAIN_BATCH_SIZE, load_file_fp=load_file_fp, percent=percent, percent_vids=percent_vids, use_random_start_frame=False)
    validationset = UCF101DataLoader('validation',vid_size, VAL_BATCH_SIZE, use_random_start_frame=False)
    
    train_data_loader = DataLoader(
        dataset=trainset,
        batch_size=TRAIN_BATCH_SIZE,
        num_workers=5,
        shuffle=True
    )

    val_data_loader = DataLoader(
        dataset=validationset,
        batch_size=VAL_BATCH_SIZE,
        num_workers=5,
        shuffle=False
    )
    
    model = CapsNet(pretrained_load=pretrained_load)
        
    if USE_CUDA:
        model = model.cuda()
    
    optimizer = optim.Adam(model.parameters(), lr=LR, betas=[0.5, 0.999], weight_decay=0, eps=1e-6)
    if LR_SCHEDULER:
        print("Scheduler on for step on ", LR_step_size)
        scheduler = StepLR(optimizer, step_size=LR_step_size, gamma=0.1)
    
    criterion = SpreadLoss(num_class=24, m_min=0.2, m_max=0.9)
    criterion2 = DiceLoss()
    criterion3 = 'None' # Not used in this mode
    
    criterion4 = ClusterLoss(beta)
    print("Cluster loss beta: ", beta)
    
    
    save_root = os.path.join('trained', 'active_learning', 'checkpoints_ucf101_capsules_i3d')
    print("Save for :", training_specs)
    
    kmeans = batch_KMeans(args)
    print("KMeans object prepared")

    # Prepare initial cluster from pretraining
    model.eval()
    model.training = False 
    model.cluster_init = True
    batch_X = []
    vid_names = []
    for i in range(2):
        for batch_idx, minibatch in enumerate(train_data_loader):
            data = minibatch['data']
            data = data.type(torch.cuda.FloatTensor)
            action = minibatch['action']
            action = action.cuda()
            _, _, latent_X = model(data, action, latent=True)
            batch_X.append(latent_X.detach().cpu().numpy())
            vid_names.extend(minibatch['vname'])
            
    model.cluster_init = False    
    batch_X = np.vstack(batch_X)
    kmeans.init_cluster(batch_X)
    predictions = kmeans.model.predict(batch_X)
    for i in range(len(vid_names)):
        vn = vid_names[i]
        predict_x = predictions[i]
        if vn in list(vid_name_cat.keys()):
            vid_name_cat[vn].append(predict_x)
        else:
            vid_name_cat[vn] = [predict_x]
    
    for k,v in vid_name_cat.items():
        vid_name_cat[k] = np.bincount(v).argmax()
        #print(k,":",vid_name_cat[k])
    print(kmeans.model)
    print("KMeans cluster init done")
    
    cluster_fp = 'files/clusters-5_{}_init.npy'.format(m_info)
    with open(cluster_fp, 'wb') as wid:
        np.save(wid, kmeans.clusters)
    print("KMeans cluster saved at: ", cluster_fp)
    
    r = 0
    m_delta = (0.7/40)  # Reaches max r in 40 epochs 
    num_epoch_to_eval = 2
    min_epoch_to_save = 9    
    
    for e in range(1, N_EPOCHS + 1):
        name_prefix = '{}-ucf101-{}_8_ADAM_capsules_{}_RGB_Spread_BCE_epoch_{:03d}.pth'.format(MODEL_TYPE, LR, training_specs, e)
        save_path = os.path.join(save_root, name_prefix)
        train(model, train_data_loader, optimizer, [criterion,criterion2,criterion3,criterion4], e, r, 
                save_path, kmeans=kmeans, short=False, masking=IS_MASKING, 
                m_info=m_info, min_epoch_to_save=min_epoch_to_save, latent=True)
        
        if e>min_epoch_to_save and e % num_epoch_to_eval == 0:
            validate(model, val_data_loader, [criterion, criterion2, criterion3, criterion4], e, kmeans=kmeans, short=False)
        
        if LR_SCHEDULER:
            scheduler.step()


