# Hybrid Active Learning via Deep Clustering for Video Action Detection

Per frame annotation of labels for video action detection is an expensive task that increases labeling cost as the dataset increases. In this work, we expand on prior work and focus on hybrid labeling approach - rank videos and select sparse (high utility) frames from those videos using active learning to minimize labeling cost. We propose Clustering-Aware Uncertainty Scoring (CLAUS) approach, a novel hybrid active learning strategy for labeling data to be used in video action detection task.   

## Project page

Visit the project page [HERE](https://sites.google.com/view/activesparselabeling/home/claus) for more details.

## Description

This is an implementation for the CVPR 2023 paper titled: Hybrid Active Learning via Deep Clustering for Video Action Detection. 

## Pre-requisites
- python >= 3.6
- pytorch >= 1.6
- numpy   >= 1.19
- scipy   >= 1.5
- opencv  >= 3.4
- scikit-image >= 0.17
- scikit-learn >= 0.23
- tensorboard >= 2.3

We developed our code base on Ubuntu 18.04 using anaconda3. 
We suggest to clone our anaconda environment using the following code:  

``$ conda create --name <env> --file spec-file.txt``

## Folder structure

The code expects UCF101 dataset in Datasets/UCF101 directory (same format as direct download from source).

To use pretrained weights, please download charades pretrained i3d weights into weights directory from given link: [https://github.com/piergiaj/pytorch-i3d/blob/master/models/rgb_charades.pt](https://github.com/piergiaj/pytorch-i3d/blob/master/models/rgb_charades.pt)

The trained models will be saved under `trained/active_learning/checkpoints_ucf101_capsules_i3d` directory. 

Saved auxilliary files will be inside `files` directory.

The labels/annotations for ucf101 is saved as pickle files for easier processing. 


## Training step

To train, place the data and weights in appropriate folder. Then follow these:  
    `python3 train_ucf101_capsules.py <percent>`

1. Run `ucf101_cluster.py` to train model for current annotation set.

2. Once trained, get the final model file name and put that in `cluster_extract.py` and `ucf101_CLAUS.py` and save the files.

3. Run `cluster_extract.py` first. This will save the cluster values for the entire training set.

4. Run `ucf101_CAU.py` after that. This will score each video using cluster and proximity and select videos and frames for annotation.
