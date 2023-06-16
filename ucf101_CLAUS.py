import os
import time
import numpy as np
import random
from threading import Thread
from scipy.io import loadmat
from skvideo.io import vread
import pdb
import sys 

import torch
import torch.nn as nn
from torch.utils.data import Dataset
from torchvision import datasets, transforms
from capsules_ucf101 import CapsNet
import torch.nn.functional as F
import cv2
import pickle
from scipy.stats import norm

from sklearn.decomposition import PCA


def select_frames(vid_scores_annotated, frames_per_instance):
    from scipy.stats import norm
    dist_norm_mask = norm.pdf(np.arange(9)-4, 0, 8/3.) / norm.pdf(0, 0, 8/3.)

    num_frames = len(vid_scores_annotated)
    selected_frames = []

    for k in range(frames_per_instance):
        sorted_idx = np.argsort(vid_scores_annotated)[::-1]  # Highest to lowest
        selected_frames.append(sorted_idx[0])
        vid_scores_annotated[sorted_idx[0]] = -100.
        range_low, range_high = max(sorted_idx[0]-4, 0), min(sorted_idx[0]+5, num_frames)
        range_center = sorted_idx[0]
        vid_scores_annotated[range_low:range_high] -= dist_norm_mask[4 -(range_center-range_low):4+(range_high-range_center)]

    return selected_frames


def get_cluster_scores(cluster_load_fp = 'none'):
    if cluster_load_fp == 'none':
        fp = 'files/clusterextract.pkl'
    else:
        fp = cluster_load_fp
    cluster_data = pickle.load(open(fp,'rb'))
    latent_val = []
    y_pred = []
    vid_names_l = []
    for k,v in cluster_data.items():
        latent_val.append(v['latent'])
        y_pred.append(v['pred'])
        vid_names_l.append(k)
    y_pred = np.array(y_pred)
    latent_val = np.array(latent_val).reshape(-1,408)
    pca_tool = PCA(n_components=2)
    pca_tool = pca_tool.fit(latent_val)
    reduced_data = pca_tool.transform(latent_val)

    for i in range(len(reduced_data)):
        cluster_data[vid_names_l[i]]['reduced_data'] = reduced_data[i]

    return reduced_data, y_pred, cluster_data


def find_vid_idx(vid_name, full_data):
    for i in range(len(full_data)):
        if vid_name == full_data[i][0]:
            return i
    return -1


class UCF101DataLoader(Dataset):
    def __init__(self, name, clip_shape, batch_size, use_random_start_frame=False):
      self._dataset_dir = 'Datasets/UCF101'
      
      if name == 'train':
          self.vid_files = self.get_det_annots_prepared()
          print("TRAINING EVAL MODE !!!!")
      else:
          print("Should not run in test mode!")
          exit()

      self._height = clip_shape[0]
      self._width = clip_shape[1]
      self._batch_size = batch_size
      self._size = len(self.vid_files)
      self.indexes = np.arange(self._size)
    
    def on_epoch_end(self):
        'Updates indexes after each epoch'
        np.random.shuffle(self.vid_files)
        self.indexes = np.arange(self._size)
        np.random.shuffle(self.indexes)
        

    def get_det_annots_prepared(self):
        import pickle     
        training_annot_file = 'training_annots.pkl'
        with open(training_annot_file, 'rb') as tr_rid:
            training_annotations = pickle.load(tr_rid)
        print("Training samples from :", training_annot_file)
        return training_annotations
            
            
    def __len__(self):
        'Denotes the number of videos per epoch'
        return int(self._size)


    def __getitem__(self, index):
        v_name, anns = self.vid_files[index]
        clip, bbox_clip, label, annots = self.load_video(v_name, anns)
        if clip is None:
            print("Video none ", v_name)
            return None, None, None, None, None, None

        frames, h, w, ch = clip.shape
        margin_h = h - 224
        h_crop_start = int(margin_h/2)
        margin_w = w - 296
        w_crop_start = int(margin_w/2)

        clip = clip[:, h_crop_start:h_crop_start +
                    224, w_crop_start:w_crop_start+296, :]
        bbox_clip = bbox_clip[:, h_crop_start:h_crop_start +
                              224, w_crop_start:w_crop_start+296, :]
        clip_resize = np.zeros((frames, self._height, self._width, ch))
        bbox_resize = np.zeros(
            (frames, self._height, self._width, 1), dtype=np.uint8)
        for i in range(frames):
            img = clip[i]
            img = cv2.resize(img, (224, 224), interpolation=cv2.INTER_LINEAR)
            clip_resize[i] = img / 255.

            bbox = bbox_clip[i, :, :, :]
            bbox = cv2.resize(bbox, (224, 224),
                              interpolation=cv2.INTER_NEAREST)
            bbox_resize[i, bbox > 0, :] = 1

        return v_name, anns, clip_resize, bbox_resize, label, annots


    def load_video(self, video_name, annotations):
        video_dir = os.path.join(self._dataset_dir, 'UCF101_Videos/%s.avi' % video_name)
        try:
            video = vread(str(video_dir)) # Reads in the video into shape (F, H, W, 3)
        except:
            return None, None, None, None

        # creates the bounding box annotation at each frame
        n_frames, h, w, ch = video.shape
        bbox = np.zeros((n_frames, h, w, 1), dtype=np.uint8)
        label = -1
        
        multi_frame_annot = []
        for ann in annotations:
            start_frame, end_frame, label = ann[0], ann[1], ann[2]      # Label is from 0 in annotations
            multi_frame_annot.extend(ann[4])
            for f in range(start_frame, min(n_frames, end_frame+1)):
                try:
                    x, y, w, h = ann[3][f-start_frame]
                    bbox[f, y:y+h, x:x+w, :] = 1
                except:
                    print('ERROR LOADING ANNOTATIONS')
                    print(start_frame, end_frame)
                    print(video_dir)
                    exit()
        multi_frame_annot = list(set(multi_frame_annot))
        
        return video, bbox, label, multi_frame_annot


def get_thresholded_arr(i_arr, threshold = 0.5):
    # b x 1 x h x w x 1  (FG)
    # b x 1 x h x w x 25 (CLS)
    arr = np.copy(i_arr)
    if arr.shape[-1] > 1:
        arr_max = (arr == np.max(arr,-1,keepdims=True)).astype(float)
        arr *= arr_max
        arr[arr>0] = 1. 
    else:
        arr[arr>threshold] = 1.
        arr[arr<=threshold] = 0.  
    return arr


def get_uncertainty_logx(frame):
    
    frame_th = get_thresholded_arr(frame, threshold = 0.4)
    if frame_th.sum() == 0:
        return -10.
    frame_th = frame_th.astype(np.bool)
    frame[frame == 0] = 1e-8
    frame = -np.log(frame)
    uncertainty = frame[frame_th].sum() / frame_th.sum()
    return uncertainty


def get_annotated_list(load_file_fp):
    import pickle     
    training_annot_file = load_file_fp
    
    with open(training_annot_file, 'rb') as tr_rid:
        training_annotations = pickle.load(tr_rid)
    print("Annotated samples from :", training_annot_file)
    
    vid_names = []
    frame_annots = []
    
    for i in range(len(training_annotations)):
        vid_name, annotations = training_annotations[i]
        vid_names.append(vid_name)
    
        multi_frame_annot = []
        for ann in annotations:
            multi_frame_annot.extend(ann[4])            
        multi_frame_annot = list(set(multi_frame_annot))

        frame_annots.append(multi_frame_annot)

    return vid_names, frame_annots


def debug(prelim_cluster_score):
    
    # -------------------------------------------
    
    select_new_vids = 0.05
    
    save_file = 'ucf101_annotation_cluster_6.0.pkl'
    fp_prior_load = 'ucf101_annotation_cluster_5.0.pkl'
    
    cluster_load_fp = 'files/clusterextract.pkl'
    
    # -------------------------------------------
    
    data = pickle.load(open(prelim_cluster_score,'rb'))

    reduced_data, cluster_pred, cluster_data = get_cluster_scores(cluster_load_fp)

    raw_vid_scores = data['raw_scores_per_vid']
        
    avg_raw_scores = []
    avg_length = []
    vid_names = []
    for i in range(len(raw_vid_scores)):
        avg_raw_scores.append(raw_vid_scores[i]['overall_selected_score'] / raw_vid_scores[i]['selected_length'])
        avg_length.append(raw_vid_scores[i]['selected_length'])
        vid_names.append(raw_vid_scores[i]['vid_name'])

    scores_per_vid = data['scores']
    sorted_idx = np.argsort(scores_per_vid)[::-1]
    select_vid_num = int(select_new_vids * len(scores_per_vid))    #int(0.05 * len(scores_per_vid))
    print("Selecting vids: ", select_vid_num)
    
    for i in range(len(scores_per_vid)):
        if scores_per_vid[i] == -100000:
            avg_raw_scores[i] = -100000

    avg_sorted_idx = np.argsort(avg_raw_scores)[::-1]

    ta = 0
    ta_len = 0
    old_ta = 0
    
    classes = np.zeros((24))
    cluster_classes = np.zeros((5))
    max_samples_per_class = 38
    max_budget = int(403282 * select_new_vids * 0.05)      #int(403282 * 0.05 * 0.05)
    total_selected_samples = 0
    total_selected_annots = 0

    temp_training_annotations = data['temp_training_annots']

    new_training_annotations = pickle.load(open(fp_prior_load, 'rb'))

    fp = 'training_annots.pkl'
    full_data = pickle.load(open(fp,'rb'))

    selected_rd = []

    overall_cluster_rep = np.zeros((5))
    
    for i in range(5):
        overall_cluster_rep[i] = np.sum(cluster_pred==i)
    print("Overall cluster rep: ", overall_cluster_rep)
    
    prior_vnames = []
    for i in range(len(new_training_annotations)):
        classes[new_training_annotations[i][1][0][2]] += 1
        for j in range(len(new_training_annotations[i][1])):
            total_selected_annots += len(new_training_annotations[i][1][j][4])
        
        
        vname = new_training_annotations[i][0]
        cluster_classes[cluster_data[vname]['pred']] += 1
        selected_rd.append(cluster_data[vname]['reduced_data'])
        prior_vnames.append(vname)

    pre_anns = total_selected_annots
    print("Pre existing samples: ", np.sum(classes))
    print("Pre existing annots: ", pre_anns)
    print(classes)
    print("Pre existing cluster:", cluster_classes)
    
    sample_training_annotations = []
    selected_indices = []
    separate_rd = []
    
    total_new_vids = np.sum(classes) + select_vid_num
    new_classes = np.copy(classes)
    new_classes_cluster = np.copy(classes)
    cluster_classes_2 = np.copy(cluster_classes)
    cluster_limit_percent = np.zeros((5))
    cluster_limit_percent = overall_cluster_rep * 0.54
    print("Cluster limit percent: ", cluster_limit_percent)

    for i in range(len(avg_sorted_idx)):
        vname = vid_names[avg_sorted_idx[i]]
        if not vname in list(cluster_data.keys()):
            continue
        if vname in prior_vnames:
            continue
        cluster_predicted_class = cluster_data[vname]['pred']
        if cluster_classes[cluster_predicted_class] <= int(cluster_limit_percent[cluster_predicted_class]) and avg_raw_scores[avg_sorted_idx[i]]>-100000:
            sample_training_annotations.append(temp_training_annotations[avg_sorted_idx[i]])
            selected_indices.append(avg_sorted_idx[i])
            
            if cluster_classes[cluster_predicted_class] > (total_new_vids/5.): # and avg_sorted_idx[i] < 2280:
                separate_rd.append(cluster_data[vname]['reduced_data']) # (reduced_data[avg_sorted_idx[i]])

            cluster_classes[cluster_predicted_class] += 1
            selected_rd.append(cluster_data[vname]['reduced_data'])

            total_selected_samples += 1
            for j in range(len(temp_training_annotations[avg_sorted_idx[i]][1])):
                total_selected_annots += len(
                    temp_training_annotations[avg_sorted_idx[i]][1][j][4])
            
            ta_len += avg_length[avg_sorted_idx[i]]
            new_classes[temp_training_annotations[avg_sorted_idx[i]][1][0][2]] += 1

        classes[temp_training_annotations[avg_sorted_idx[i]][1][0][2]] += 1

        if total_selected_samples >= select_vid_num:
            break

    cluster_select_indices = []
    selected_rd_cluster = []
    total_selected_samples_2 = 0
    for i in range(len(avg_sorted_idx)): 
        vname = vid_names[avg_sorted_idx[i]]
        if not vname in list(cluster_data.keys()):
            continue
        cluster_predicted_class = cluster_data[vname]['pred']
        if new_classes_cluster[temp_training_annotations[avg_sorted_idx[i]][1][0][2]] <= max_samples_per_class and cluster_classes_2[cluster_predicted_class] <= (total_new_vids/5.):
            cluster_select_indices.append(avg_sorted_idx[i])
            
            cluster_classes_2[cluster_predicted_class] += 1
                
            total_selected_samples_2 += 1
            
        new_classes_cluster[temp_training_annotations[avg_sorted_idx[i]][1][0][2]] += 1

        if total_selected_samples_2 >= select_vid_num:
            break

    unique_cluster = list(set(selected_indices) - set(cluster_select_indices))
    print("Cluster total: ", len(cluster_select_indices))
    print("Cluster only: ", len(unique_cluster))
    cluster_selected_rd=[]
    for i in range(len(unique_cluster)):
        idx = unique_cluster[i]
        vname = vid_names[idx]
        cluster_selected_rd.append(cluster_data[vname]['reduced_data']) # (reduced_data[idx])


    print("Samples per cluster")
    print(cluster_classes)
    print("New classes")
    print(new_classes)
    print("Selected samples len: ", len(sample_training_annotations))

    selected_rd = np.array(selected_rd)
    separate_rd = np.array(separate_rd)
    cluster_selected_rd = np.array(cluster_selected_rd)

    # reselect frames
    frames_per_vid = int(np.round(float(max_budget) / select_vid_num))
    print("Select frames per vid: ", frames_per_vid)
    selected_frames = 0
    selected_frames_filtered = 0
    dup_count = 0
    for i in range(len(sample_training_annotations)):
        sample_index = selected_indices[i]
        raw_score = raw_vid_scores[sample_index]['vid_score']
        selected_annotations = sample_training_annotations[i]
        if selected_annotations[0] in prior_vnames:
            dup_count += 1
            print("Dup ", selected_annotations[0])
        combined_annots = select_frames(raw_score, frames_per_vid)
        selected_frames += len(combined_annots)
        new_annotations = []
        for ann in selected_annotations[1]:
            sf, ef, label, bboxes = ann[0], ann[1], ann[2], ann[3]
            new_frame_annots = []
            for j in range(len(combined_annots)):
                if sf <= combined_annots[j] <= ef:
                    new_frame_annots.append(combined_annots[j])
            new_annotations.append((sf, ef, label, bboxes, new_frame_annots))
            selected_frames_filtered += len(new_frame_annots)
        new_training_annotations.append((selected_annotations[0], new_annotations))
    print("Selected frames (unique): ", selected_frames)
    print("Selected frames multi annot: ", selected_frames_filtered)
    print("Dup count: ", dup_count)
    
    prior_ta = 0
    prior_ta_unique = 0
    cls_selected = np.zeros((24))
    for i in range(len(new_training_annotations)):
        cls_selected[new_training_annotations[i][1][0][2]] += 1
        uni_annot = []
        for j in range(len(new_training_annotations[i][1])):
            prior_ta += len(new_training_annotations[i][1][j][4])
            uni_annot.extend(new_training_annotations[i][1][j][4])
            if len(new_training_annotations[i][1][j][4]) == 0:
                print("Missing: ", new_training_annotations[i][0])
        prior_ta_unique += len(set(uni_annot))
        
    print("Avg ann length: ", ta_len/len(sample_training_annotations))
    print("Total new annots: ", prior_ta)
    print("Total new annots (unique): ", prior_ta_unique)
    print("Selected annots: ", prior_ta - pre_anns)
    print("Selected samples: ", total_selected_samples)

    print("Classes selected")
    print(cls_selected)
    print("Total vids: ", np.sum(cls_selected))
    
    with open(save_file, 'wb') as wid:
        pickle.dump(new_training_annotations, wid, pickle.HIGHEST_PROTOCOL)

    print("saved to ", save_file)


if __name__ == '__main__':
    
    # ----------------------------------
    
    load_file_fp = 'ucf101_annotation_cluster_5.0.pkl'  # Previous round's annotation list
    
    model_file_path = './trained/active_learning/checkpoints_ucf101_capsules_i3d/TRAINED_MODEL_FOR_CURRENT_STEP.pth'
    
    save_file_fp = 'files/prelim_scores_cluster_6.0.pkl'  # Video scores for next round
    
    # ----------------------------------
    
    name='train'
    clip_shape=[224,224]
    channels=3
    batch_size = 1
    select_vid_percent = 0.05
    
    vid_idx_start = 0
    
    model = CapsNet()
    model.load_previous_weights(model_file_path)
    print("Model loaded from: ", model_file_path)
    model = model.to('cuda')
    model.eval()
    model.training = False
    
    # Enable dropout in eval mode 
    for m in model.modules():
        if m.__class__.__name__.startswith("Dropout"):
            m.train()
    
    dataloader = UCF101DataLoader(name, clip_shape, batch_size)
    
    iou_thresh = np.arange(0, 1, 0.1)
    frame_tp = np.zeros((24, iou_thresh.shape[0]))
    frame_fp = np.zeros((24, 1))
    
    clip_span = 16
    num_vids = len(dataloader)
    # hybrid with video needs global done together for now
    clip_batch_size = 14
    num_forward_passes = 10
    uncertainty_thresh = -np.log(0.6)
    
    print("Total vids: ", num_vids)
    new_training_annotations = []
    temp_training_annotations = []
    scores_per_vid = []
    vid_names, frame_annots = get_annotated_list(load_file_fp)
    raw_scores_per_vid = []
        
    with torch.no_grad():
        for i in range(num_vids):
            v_name, anns, video, bbox, label, _ = dataloader.__getitem__(i)
            
            select_percent = 0.05   # % of total annots to add to previous file (so 15% added to 5% will be 20% total)
            
            if v_name in vid_names:
                select_percent = 0.05   # Cause it already has some annotations, we just add partially to this
                vid_idx = vid_names.index(v_name)
                annots = frame_annots[vid_idx]
            else:
                annots = []
            
            if video is None:
                print("Video invalid. None.")
                continue
                
            num_frames = video.shape[0]
            if num_frames == 0:
                print("Video invalid. No frames.")
                continue
            
            vid_scores = np.zeros((num_frames))
            # prepare batches of this video, get results from model, stack np arrays for results 
            batches = 0
            bbox_pred_fg = np.zeros((num_frames, clip_shape[0], clip_shape[1], 1))
            while True:
                batch_frames = np.zeros((1,8,224,224,3))
                for j in range(8):
                    ind = (batches * clip_span) + (j * 2)
                    if ind >= num_frames:
                        if j > 0:
                            batch_frames[0,j] = batch_frames[0,j-1]
                    else:
                        batch_frames[0,j] = video[ind]
                
                data = np.transpose(np.array(batch_frames), [0, 4, 1, 2, 3])
                data = torch.from_numpy(data).type(torch.cuda.FloatTensor)
                action_tensor = np.ones((len(batch_frames),1), np.int) * label
                action_tensor = torch.from_numpy(action_tensor).cuda()
                
                segmentation_np = np.zeros((len(batch_frames), 1, 8, clip_shape[0], clip_shape[1]))
                for j in range(num_forward_passes):
                    segmentation, pred = model(data, action_tensor)
                    segmentation = F.sigmoid(segmentation)
                    segmentation_np += segmentation.cpu().data.numpy()   # B x C x F x H x W -> B x 1 x 8 x 224 x 224
                segmentation_np = segmentation_np / num_forward_passes
                segmentation_np = np.transpose(segmentation_np, [0, 2, 3, 4, 1]) 
                
                output_fg = segmentation_np[0]      # F x H x W x C
                output_fg = np.repeat(output_fg, 2, axis=0)
                
                end_idx = (batches+1) * clip_span
                if end_idx > num_frames:
                    end_idx = num_frames
                start_idx = batches * clip_span
                bbox_pred_fg[start_idx : end_idx] = output_fg[0:(end_idx - start_idx)]
                
                if end_idx >= num_frames:
                    break
                    
                batches += 1
            
            for f_idx in range(num_frames):
                if f_idx in annots:
                    vid_scores[f_idx] = -10.   # Already in annot list 
                    continue
                    
                output_fg = bbox_pred_fg[f_idx]
                if np.sum(output_fg) <= 0.1:
                    uncertainty_score = -10
                else:
                    # cost function
                    uncertainty_score = get_uncertainty_logx(output_fg)
                
                if len(annots) == 0:
                    closest_frame_dist = 0.     # No annotations yet
                else:
                    closest_frame_dist = np.min(np.abs(np.array(annots)-f_idx))

                frame_score = uncertainty_score - (norm.pdf(closest_frame_dist, 0, 8/3.) / norm.pdf(0, 0, 8/3.))
                vid_scores[f_idx] = frame_score     # Higher is better
            
            len_ann = np.count_nonzero(np.sum(bbox,(1,2,3)))
            select_frames = int(np.round(len_ann * select_percent))
            if select_frames < 1:
                select_frames = 1
            selected_frame = []
            
            start_frame = anns[0][0]
            end_frame = anns[0][1]
            vid_scores_annotated = np.ones((num_frames)) * -1000.
            vid_scores_annotated[start_frame:end_frame] = vid_scores[start_frame:end_frame]

            vid_scores_annotated_copy = np.copy(vid_scores_annotated)
            
            dist_norm_mask = norm.pdf(np.arange(9)-4, 0, 8/3.) / norm.pdf(0, 0, 8/3.)
            
            overall_vid_score = 0.
            for k in range(select_frames):
                sorted_idx = np.argsort(vid_scores_annotated)[::-1] # Highest to lowest
                selected_frame.append(sorted_idx[0])
                overall_vid_score += vid_scores_annotated[sorted_idx[0]]
                vid_scores_annotated[sorted_idx[0]] = -100.
                range_low, range_high = max(sorted_idx[0]-4,0), min(sorted_idx[0]+5, num_frames)
                range_center = sorted_idx[0]
                vid_scores_annotated[range_low:range_high] -= dist_norm_mask[4-(range_center-range_low):4+(range_high-range_center)]
            combined_annots = selected_frame[0:select_frames]
            
            raw_scores_per_vid.append({'vid_name':v_name, 'vid_score': vid_scores_annotated_copy,
                                      'overall_selected_score': overall_vid_score, 'selected_length': len(combined_annots)})

            combined_annots.extend(annots)
            new_annotations = []
            for ann in anns:
                sf, ef, label, bboxes = ann[0], ann[1], ann[2], ann[3]
                new_frame_annots = []
                for j in range(len(combined_annots)):
                    if sf <= combined_annots[j] <= ef:
                        new_frame_annots.append(combined_annots[j])
                new_annotations.append((sf, ef, label, bboxes, new_frame_annots))
            temp_training_annotations.append((v_name, new_annotations))
            
            if v_name in vid_names:
                scores_per_vid.append(-100000)
                new_training_annotations.append((v_name, new_annotations))
            else:    
                scores_per_vid.append(overall_vid_score)
            
    state_dict = {'scores': scores_per_vid,
                'new_training_annots': new_training_annotations,
                'temp_training_annots': temp_training_annotations,
                'raw_scores_per_vid': raw_scores_per_vid}
    fp_sd = 'state_dict_debug_'+ save_file_fp
    with open(fp_sd, 'wb') as wsid:
        pickle.dump(state_dict, wsid, protocol=4)
    print("State dict saved in: ", fp_sd)
    
    debug(fp_sd)
        
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
         
      
      
      
      
