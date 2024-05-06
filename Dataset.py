#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Saturday April 20 2024
Requires ImageHoof dataset to be downloaded, extracted and stored in program 
directory for access. Link is provided below.
The dataset should be divided into a train_dir and val_dir for training and testing
respectively. 
 Image dimensions to be 224 x 224.
@author: dinesh
"""

import torchvision
import torchvision as tv
import torch.utils.data as datas
import torch.utils.data as data
import os

#%% ----------ImageHoof --------------------------------------
#source - https://github.com/fastai/imagenette
#https://auth0.com/blog/image-processing-in-python-with-pillow/
#https://www.analyticsvidhya.com/blog/2018/03/comprehensive-collection-deep-learning-datasets/
#https://towardsdatascience.com/how-to-scrape-the-imagenet-f309e02de1f4
    
def LoadImageHoof_Torch():
 
    #indicate your directory where imagehoof train dataset is
    traindir = 'data' + os.sep + 'imagewoof2' +  os.sep + 'train_dir'
    
    #indicate your directory where imagehoof test dataset is
    valdir = 'data' + os.sep + 'imagewoof2' + os.sep + 'val_dir'

    #for our experiment on Imagehoof dataset, no normalisation was applied 
    train_transform = tv.transforms.Compose([
            tv.transforms.ToTensor(),
        ])
    
    test_transform = tv.transforms.Compose([
            tv.transforms.ToTensor(),
        ])  

    trainset = torchvision.datasets.ImageFolder(root=traindir, transform=train_transform)
    trainloader = datas.DataLoader(trainset, batch_size=4, shuffle=True,  num_workers=4)
    testset = torchvision.datasets.ImageFolder(root=valdir, transform=test_transform)
    testloader  = data.DataLoader(testset, batch_size=4, shuffle=True, num_workers=4) 
    
    return trainset, trainloader, testset, testloader
    