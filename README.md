# Are All Frames Equal? Active Sparse Labeling for Video Action Detection

Video action detection requires annotations at every frame, which drastically increases the labeling cost. In this work, we focus on efficient labeling of videos for
action detection to minimize this cost. We propose active sparse labeling (ASL), a novel active learning strategy for video action detection.   

## Project page

Visit the project page [HERE](https://sites.google.com/view/activesparselabeling/home) for more details.

## Description

This is an implementation for the NeurIPS 2022 paper titled: Are All Frames Equal? Active Sparse Labeling for Video Action Detection. 

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

The code expects UCF101 dataset in data/UCF101 folder (same format as direct download from source).

To use pretrained weights, please download charades pretrained i3d weights into weights folder from given link: [https://github.com/piergiaj/pytorch-i3d/blob/master/models/rgb_charades.pt](https://github.com/piergiaj/pytorch-i3d/blob/master/models/rgb_charades.pt)

The trained models will be saved under `trained/active_learning/checkpoints_ucf101_capsules_i3d` folder 

The labels/annotations for ucf101 is saved as pickle files for easier processing. 


## Training step

To train, place the data and weights in appropriate folder. Then run as  
    `python3 train_ucf101_capsules.py <percent>`

## APU step 

This will use the APU algorithm to select frames and create new annotation pickle file. Run as:   
`python3 APU.py`
