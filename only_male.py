from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections

import numpy as np
import nibabel as nib
import scipy as sp
import scipy.ndimage

import torch
import torch.nn as nn
import torch.optim as optim
from dataloading import MRI_Dataset
from datasplitting import Specific_MRI_Dataset
import math
from torch.utils.data.dataset import ConcatDataset

from torch.utils.data import DataLoader
from torch.utils.data import RandomSampler

import torchvision
import copy
import gc
from sampler import SuperSampler,MixedSampler
from class_sampler import ClassMixedSampler
import dcor
import pickle


device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
test_data = MRI_Dataset(fold = 0 , stage= 'original_test')
class net(nn.Module):
    def __init__(self):
        super(net, self).__init__()
        self.features_conv = nn.Sequential(
            nn.Sequential(nn.Conv3d(in_channels=1, out_channels=16, kernel_size=3, padding=1),
                          nn.ReLU(),
                          nn.MaxPool3d(kernel_size=2)),
            nn.Sequential(nn.Conv3d(in_channels=16, out_channels=32, kernel_size=3, padding=1),
                          nn.ReLU(),
                          nn.MaxPool3d(kernel_size=2)),
            nn.Sequential(nn.Conv3d(in_channels=32, out_channels=64, kernel_size=3, padding=1),
                          nn.ReLU(),
                          nn.MaxPool3d(kernel_size=2)),
            nn.Sequential(nn.Conv3d(in_channels=64, out_channels=32, kernel_size=3, padding=1),
                          nn.ReLU(),
                          nn.MaxPool3d(kernel_size=2)),
            nn.Flatten(start_dim=1)).to(device)
        self.classifier = nn.Sequential(
            nn.Linear(in_features=2048, out_features=2),
            nn.Sigmoid()
        ).to(device)
        
        model = torch.load('npz_model.pth')
        encode_dict = collections.OrderedDict([(k[7:],v) for k,v in model.items() if 'linear' not in k])
        linear_dict = collections.OrderedDict([(k[7:],v) for k,v in model.items() if 'linear' in k])	
        self.features_conv.load_state_dict(encode_dict)
        self.classifier.load_state_dict(linear_dict)	
    def forward(self, x):
        x = self.features_conv(x)
        x = self.classifier(x)
        return x

#initialize the model
network = net()
network.eval()
correct_male = 0
all_male = 0

test_loader = DataLoader(dataset=test_data ,
                          batch_size=1,#64
                          shuffle=False,
                          pin_memory=True,
                          num_workers=3)
 
for (images, labels, actual_labels, datasets, ids, ages, genders,npzs)  in test_loader:
    if genders != 0:
       continue
    all_male += 1
    pred = network(images.view(1, 1, 64, 64, 64).to(device).float())[0]
    if pred[0] < 0.5 and pred[1] < 0.5 and actual_labels == 0: #CTRL
         correct_male += 1
    elif pred[0] > 0.5 and pred[1] < 0.5 and actual_labels == 1: #MCI
         correct_male += 1
    elif pred[0] < 0.5 and pred[1] > 0.5 and actual_labels == 2: #HIV
         correct_male += 1
    elif pred[0] > 0.5 and pred[1] > 0.5 and actual_labels == 3: #HAND
         correct_male += 1
    else:
         pass

correct_female = 0
all_female = 0

for (images, labels, actual_labels, datasets, ids, ages, genders,npzs)  in test_loader:
    if genders != 1:
       continue
    all_female += 1
    pred = network(images.view(1, 1, 64, 64, 64).to(device).float())[0]
    if pred[0] < 0.5 and pred[1] < 0.5 and actual_labels == 0: #CTRL
         correct_female += 1
    elif pred[0] > 0.5 and pred[1] < 0.5 and actual_labels == 1: #MCI
         correct_female += 1
    elif pred[0] < 0.5 and pred[1] > 0.5 and actual_labels == 2: #HIV
         correct_female += 1
    elif pred[0] > 0.5 and pred[1] > 0.5 and actual_labels == 3: #HAND
         correct_female += 1
    else:
         pass
print("Test accuracy on (" + str(all_male) + ") males only is: " + str(correct_male/all_male))
print("Test accuracy on (" + str(all_female) + ") females only is: " + str(correct_female/all_female))
