from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections

import numpy as np
import nibabel as nib
import scipy as sp
import scipy.ndimage
from model import fe
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
from transformation import super_transformation

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
transformation = super_transformation()
#test_data = MRI_Dataset(fold = 1, stage= 'original_train',transform = transformation)
test_data = MRI_Dataset(fold = 3, stage= 'original_test',transform = transformation)

#FELIPE'S CODE

#from sklearn.utils import shuffle
#seed=2022
# Read in the data
#train_fold = list()
#test_fold = list()
#label_train_fold = list()
#label_test_fold = list()
#gender_train_fold = list()
#gender_test_fold = list()
#for fold in range(5):
#   with open(f'/scratch/users/jmanasse/threehead/original_train_data_{fold}.pickle', 'rb') as handle:
#      train_fold.append(pickle.load(handle))
#   with open(f'/scratch/users/jmanasse/threehead/original_test_data_{fold}.pickle', 'rb') as handle:
#      test_fold.append(pickle.load(handle))
#   with open(f'/scratch/users/jmanasse/threehead/original_train_label_{fold}.pickle', 'rb') as handle:
#      label_train_fold.append(pickle.load(handle))
#   with open(f'/scratch/users/jmanasse/threehead/original_test_label_{fold}.pickle', 'rb') as handle:
#      label_test_fold.append(pickle.load(handle))
#   with open(f'/scratch/users/jmanasse/threehead/train_gender_{fold}.pickle', 'rb') as handle:
#      gender_train_fold.append(pickle.load(handle))
#   with open(f'/scratch/users/jmanasse/threehead/test_gender_{fold}.pickle', 'rb') as handle:
#      gender_test_fold.append(pickle.load(handle))
#train_data = np.concatenate(tuple(train_fold))
#test_data = np.concatenate(tuple(test_fold))
#label_train = np.concatenate(tuple(label_train_fold))
#label_test = np.concatenate(tuple(label_test_fold))
#gender_train = np.concatenate(tuple(gender_train_fold))
#gender_test = np.concatenate(tuple(gender_test_fold))

# Apply deterministic shuffle in the list (meaning all lists are shuffled in a same random order)
#train_data = shuffle(train_data, random_state=seed)        
#test_data = shuffle(test_data, random_state=seed)
#label_train = shuffle(label_train, random_state=seed)
#label_test = shuffle(label_test, random_state=seed)
#gender_train = shuffle(gender_train, random_state=seed)
#gender_test = shuffle(gender_test, random_state=seed)
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
        #self.features_conv = fe(trainset_size = len(test_data), in_num_ch=1, img_size=(64, 64, 64), inter_num_ch=16,
        #                   fc_num_ch=16, kernel_size=3, conv_act='LeakyReLU',
        #                   fe_arch= 'fe1', dropout=0.2,
        #                   fc_dropout = 0.2, batch_size = 1).to(device)
        #self.classifier = nn.Sequential(
        #            nn.Linear(2048, 128),
        #            nn.LeakyReLU(),
        #            nn.Linear(128,16),
        #            nn.LeakyReLU(),
        #             nn.Linear(16, 2),
        #        ).to(device)
        
        model = torch.load('npz_model.pth')
        encode_dict = collections.OrderedDict([(k[7:],v) for k,v in model.items() if 'linear' not in k])
        linear_dict = collections.OrderedDict([(k[7:],v) for k,v in model.items() if 'linear' in k])	
        self.features_conv.load_state_dict(encode_dict)
        self.classifier.load_state_dict(linear_dict)	
        
        #self.features_conv.load_state_dict(torch.load('fe_weights.pt'))
        #self.classifier.load_state_dict(torch.load('class_weights.pt'))
    def forward(self, x):
        x = self.features_conv(x)
        x = self.classifier(x)
        return x

#initialize the model
network = net()
network.eval()
correct_male = 0
all_male = 0
correct_female = 0
all_female = 0
test_loader = DataLoader(dataset=test_data ,
                          batch_size=1,#64
                          shuffle=False,
                          pin_memory=True,
                          num_workers=3)
 
count_ctrl = count_hiv = count_mci = count_hand = 0

#for i, images in enumerate(test_data):
for (images, labels, actual_labels, datasets, ids, ages, genders,npzs)  in test_loader:
    if genders == 0:
    #if gender_test[i] != 0:
       all_male += 1
    else:
       all_female += 1
    pred = network(images.view(1, 1, 64, 64, 64).to(device).float())[0]
    #pred = network(torch.tensor(images).view(1, 1, 64, 64, 64).to(device).float())[0]
    #print(pred)
    pred_cd = round(float(pred[0]))
    pred_hiv = round(float(pred[1]))
    #if pred_cd==float(label_test[i][0]) and pred_hiv==float(label_test[i][1]):
    if pred_cd==float(labels[0][0]) and pred_hiv==float(labels[0][1]):
       if genders == 0:
       #if gender_test[i] != 0:
          correct_male += 1
       else:
          correct_female += 1

print("Test accuracy on (" + str(all_male+all_female) + ") samples is: " + str((correct_male+correct_female)/(all_male+all_female)))
print("Test accuracy on (" + str(all_male) + ") males only is: " + str(correct_male/all_male))
print("Test accuracy on (" + str(all_female) + ") females only is: " + str(correct_female/all_female))
