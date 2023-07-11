import os
import shutil
import time
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import nibabel as nib
import sklearn.metrics
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import confusion_matrix, balanced_accuracy_score
import scipy.ndimage
import scipy.stats
import scipy.misc as sci
import matplotlib.pyplot as plt
import matplotlib as mpl
import skimage.color
import scipy as sp
from sklearn.metrics import mean_squared_error, r2_score
import pickle




class MRI_Dataset(Dataset):
    """4 classes MRI dataset."""

    def __init__(self,  fold, stage,  transform=None):
        self.transform = transform

        print('fold:',fold)
        if stage == 'original_train':
            pickle_name = '/scratch/users/jmanasse/threehead/original_train_data_'+str(fold)+'.pickle'
            with open(pickle_name, 'rb') as handle:
                original_train_data = pickle.load(handle)

            pickle_name = '/scratch/users/jmanasse/threehead/original_train_label_'+str(fold)+'.pickle'
            with open(pickle_name, 'rb') as handle:
                original_train_label = pickle.load(handle)

            pickle_name = '/scratch/users/jmanasse/threehead/original_train_actual_label_'+str(fold)+'.pickle'
            with open(pickle_name, 'rb') as handle:
                original_actual_train_dx = pickle.load(handle)



            pickle_name = '/scratch/users/jmanasse/threehead/original_train_dataset_'+str(fold)+'.pickle'
            with open(pickle_name, 'rb') as handle:
                original_train_dataset = pickle.load(handle)

            pickle_name = '/scratch/users/jmanasse/threehead/train_id_'+str(fold)+'.pickle'
            with open(pickle_name, 'rb') as handle:
                train_id = pickle.load(handle)

            pickle_name = '/scratch/users/jmanasse/threehead/train_age_'+str(fold)+'.pickle'
            with open(pickle_name, 'rb') as handle:
                train_ages = pickle.load(handle)

            pickle_name = '/scratch/users/jmanasse/threehead/train_gender_'+str(fold)+'.pickle'
            with open(pickle_name, 'rb') as handle:
                train_genders = pickle.load(handle)

            pickle_name = '/scratch/users/jmanasse/threehead/train_npz_'+str(fold)+'.pickle'
            with open(pickle_name, 'rb') as handle:
                train_npzs = pickle.load(handle)

            ucsf_indices = [i for i,d in enumerate(original_train_dataset) if d == 'ucsf']

            self.data = original_train_data
            self.dx = original_train_label
            self.dataset = original_train_dataset
            print('ucsf:'+ str(sum([1 for d in self.dataset if d == 'ucsf'])))
            self.actual_dx = original_actual_train_dx
            self.ids = train_id
            self.ages = train_ages
            self.genders = train_genders
            self.npzs = train_npzs
            print('original_train_data num ', self.data.shape[0])
            print('actual label ', self.actual_dx.shape)



        elif stage == 'original_test':
            pickle_name = '/scratch/users/jmanasse/threehead/original_test_data_'+str(fold)+'.pickle'
            with open(pickle_name, 'rb') as handle:
                test_data = pickle.load(handle)

            pickle_name = '/scratch/users/jmanasse/threehead/original_test_label_'+str(fold)+'.pickle'
            with open(pickle_name, 'rb') as handle:
                test_dx = pickle.load(handle)

            pickle_name = '/scratch/users/jmanasse/threehead/original_test_actual_label_'+str(fold)+'.pickle'
            with open(pickle_name, 'rb') as handle:
                actual_test_dx = pickle.load(handle)

            pickle_name = '/scratch/users/jmanasse/threehead/original_test_dataset_'+str(fold)+'.pickle'
            with open(pickle_name, 'rb') as handle:
                test_dataset = pickle.load(handle)

            pickle_name = '/scratch/users/jmanasse/threehead/test_id_'+str(fold)+'.pickle'
            with open(pickle_name, 'rb') as handle:
                test_id = pickle.load(handle)

            pickle_name = '/scratch/users/jmanasse/threehead/test_age_'+str(fold)+'.pickle'
            with open(pickle_name, 'rb') as handle:
                test_ages = pickle.load(handle)

            pickle_name = '/scratch/users/jmanasse/threehead/test_gender_'+str(fold)+'.pickle'
            with open(pickle_name, 'rb') as handle:
                test_genders = pickle.load(handle)

            pickle_name = '/scratch/users/jmanasse/threehead/test_npz_'+str(fold)+'.pickle'
            with open(pickle_name, 'rb') as handle:
                test_npzs = pickle.load(handle)

            self.data = test_data
            self.dx = test_dx
            self.dataset = test_dataset
            print('ucsf:' + str(sum([1 for d in self.dataset if d == 'ucsf'])))
            self.actual_dx = actual_test_dx
            self.ids = test_id
            self.ages = test_ages
            self.genders = test_genders
            self.npzs = test_npzs
            print('original_test num ', self.data.shape[0])


        self.subject_num = self.data.shape[0]
    def __len__(self):
        return self.subject_num

    def __getitem__(self, idx):
        if self.transform != None:

            #if type(idx) != int:

             #  image = (self.transform(self.data[idx[0]]),self.transform(self.data[idx[1]]))

            #else:
            image = self.transform(self.data[idx])
        else:
            if type(idx) != int:
               image = (self.data[idx[0]],self.data[idx[1]])
            else:
               image = self.data[idx]


        #if type(idx) != int:
        #   return image, (self.dx[idx[0]], self.dx[idx[1]]),(self.actual_dx[idx[0]],self.actual_dx[idx[1]]),(self.dataset[idx[0]],self.dataset[idx[1]]),(self.ids[idx[0]],self.ids[idx[1]]),(self.ages[idx[0]],self.ages[idx[1]]),(self.genders[idx[0]],self.genders[idx[1]]),(self.npzs[idx[0]],self.npzs[idx[1]])
        #else:
        return image, self.dx[idx],self.actual_dx[idx],self.dataset[idx],self.ids[idx],self.ages[idx],self.genders[idx],self.npzs[idx]

