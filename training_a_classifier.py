#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 30 13:28:52 2018

@author: pengyuyan
"""

import torch
import torchvision
import torchvision.transforms as transforms

transform = transforms.Compose(
        [transforms.ToTensor, 
         transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))])
trainset = torchvision.datasets.CIFAR10(
        root='./data', train=True, 
        download=True, transform=transform)
trainloader = torch.utils.data.Dataloader(
        trainset, batch_size=4, 
        shuffle=True, num_workers=2)
testset = torchvision.datasets.CIFAR10(
        root='./data', train=False, 
        download=True, transform=transform)
trainloader = torch.utils.data.Dataloader(
        testset, batch_size=4, 
        shuffle=True, num_workers=2)
classes = ('plane', 'car', 'bird', 'cat', 'deer', 
           'dog', 'frog', 'horse', 'ship', 'truck')

import matplotlib.pyplot as plt
import numpy as np

def imshow(img):
    img = img/2 + 0.5
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg),(1,2,0))
    
dataiter = iter(trainloader)
images, labels = dataiter.next()



