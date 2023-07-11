import numpy as np
import pandas as pd
import nibabel as nib
import matplotlib.pyplot as plt
import torch.optim as optim
import torch
import torch. nn as nn
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data.sampler import BatchSampler
import pickle
import torch
import torch.nn as nn
import torch.optim as optim
from scipy import stats
from torch.utils.data import DataLoader
import torch.nn.functional as F
from torch.utils.data.dataset import ConcatDataset
import math
import copy

def get_index(index_lists, output):
    if len(index_lists)>0:
        output.append(index_lists.pop(0))
        return False

    else:
        return True
# class SuperSampler(torch.utils.data.sampler.Sampler):
#     """
#     BatchSampler - from a MNIST-like dataset, samples n_classes and within these classes samples n_samples.
#     Returns batches of size n_classes * n_samples
#     """
#
#     def __init__(self, dataset,  batch_size, status = None):
#         self.status = status
#         self.batch_size = batch_size
#
#         loader = DataLoader(dataset)
#         labels_list = []
#         datasets_list = []
#         ids_ = []
#         for images, labels, actual_labels, datasets,ids in loader:
#             labels_list.append(actual_labels)
#
#             datasets_list.append(datasets[0])
#             ids_.append(ids)
#
#         self.num_batches = math.ceil( len(labels_list)/batch_size)
#
#         labels_list=np.array(labels_list)
#         datasets_list=np.array(datasets_list)
#         ids_=np.array(ids_)
#
#         #ucsf
#         temp = copy.deepcopy(labels_list)
#         temp[datasets_list!='ucsf'] = -99
#         ucsf_labels_list=temp
#
#         self.ucsf_label_to_indices = {label: np.where(ucsf_labels_list == label)[0]
#                                  for label in [0,1,2,3]}
#
#
#         #lab
#         temp = copy.deepcopy(labels_list)
#         temp[datasets_list!='lab'] = -99
#         lab_labels_list=temp
#         self.lab_label_to_indices = {label: np.where(lab_labels_list == label)[0]
#                                  for label in [0,2]}
#         #adni
#         temp = copy.deepcopy(labels_list)
#         temp[datasets_list!='adni'] = -99
#         adni_labels_list=temp
#         self.adni_label_to_indices = {label: np.where(adni_labels_list == label)[0]
#                                  for label in [0,1]}
#
#         # ucsf
#
#         self.all_ucsf_index = []
#         for l in [0,1,2,3]:
#             self.all_ucsf_index.extend(self.ucsf_label_to_indices[l])
#         # lab
#         self.all_lab_index = []
#         for l in [0,2]:
#             self.all_lab_index.extend(self.lab_label_to_indices[l])
#
#         # adni
#         self.all_adni_index = []
#         for l in [0,1]:
#             self.all_adni_index.extend(self.adni_label_to_indices[l])
#
#     def __iter__(self):
#         ucsf_0 = list(np.copy(self.ucsf_label_to_indices[0]))
#         ucsf_1 = list(np.copy(self.ucsf_label_to_indices[1]))
#         ucsf_2 = list(np.copy(self.ucsf_label_to_indices[2]))
#         ucsf_3 = list(np.copy(self.ucsf_label_to_indices[3]))
#         lab_0 = list(np.copy(self.lab_label_to_indices[0]))
#         lab_2 = list(np.copy(self.lab_label_to_indices[2]))
#         adni_0 = list(np.copy(self.adni_label_to_indices[0]))
#         adni_1 = list(np.copy(self.adni_label_to_indices[1]))
#
#         all_index = {}
#         all_index[0] = ucsf_0
#         all_index[1] = ucsf_1
#         all_index[2] = ucsf_2
#         all_index[3] = ucsf_3
#         all_index[4] = lab_0
#         all_index[5] = lab_2
#         all_index[6] = adni_0
#         all_index[7] = adni_1
#         for i in range(0,8):
#             np.random.shuffle(all_index[i])
#
#         indices = []
#
#         output = {}
#         for i in range(0,self.num_batches):
#             output[i] = []
#         reverse = False
#         current_p = 0
#
#         old_i = 0
#         old_count = 0
#         for i in range(0,8):
#             selected_list = all_index[i]
#             while len(selected_list)>0:
#                 if reverse == False:
#                     for j in range(current_p,self.num_batches):
#                         none_left = get_index(selected_list,output[current_p])
#
#                         if none_left:
#                             break
#                         else:
#                             if current_p == self.num_batches-1:
#                                 current_p = 0
#                             else:
#                                 current_p+=1
#
#
#         for i in range(0,self.num_batches):
#             indices.extend(output[i])
#             print(i,len(output[i]))
#
#         return iter(indices)
#     def __len__(self):
#         return 1
#

class ClassMixedSampler(torch.utils.data.sampler.Sampler):
    """
    BatchSampler - from a MNIST-like dataset, samples n_classes and within these classes samples n_samples.
    Returns batches of size n_classes * n_samples
    """

    def __init__(self, dataset,  batch_size, num_batches = 27,status = None):
        self.status = status
        self.batch_size = batch_size
        #self.dataset = ['ucsf','lab']
        loader = DataLoader(dataset)
        labels_list = []
        datasets_list = []
        for images, labels, actual_labels, datasets,ids,ages,genders in loader:
            labels_list.append(actual_labels)

            datasets_list.append(datasets[0])

        self.num_batches = num_batches #math.ceil( len(labels_list)/batch_size)

        labels_list=np.array(labels_list)
        datasets_list=np.array(datasets_list)
        dataset_type = datasets_list[0]
        #ucsf
        temp = np.copy(labels_list)
#         print(temp)
#         print(datasets_list)
        # temp[datasets_list!='ucsf'] = -99
        # ucsf_labels_list=temp
        # self.ucsf_label_to_indices = {label: np.where(ucsf_labels_list == label)[0]
        #                          for label in [0,1,2,3]}
        # #lab
        # temp = np.copy(labels_list)
        # temp[datasets_list!='lab'] = -99
        # lab_labels_list=temp
        # self.lab_label_to_indices = {label: np.where(lab_labels_list == label)[0]
        #                          for label in [0,2]}

        #adni
        #temp = copy.deepcopy(labels_list)
        # self.label_choices = None
        # if dataset_type == 'ucsf':
        self.label_choices = [0,1,2,3]
        # elif dataset_type == 'lab':
        #     self.label_choices = [0,2]
        # else:
        #     self.label_choices = [0,1]

        self.label_to_indices = {label: np.where(labels_list == label)[0]
                                 for label in self.label_choices}

#         ucsf
        self.all_index = []
        for l in self.label_choices:
            self.all_index.extend(self.label_to_indices[l])
        # lab
        # self.all_lab_index = []
        # for l in [0,2]:
        #     self.all_lab_index.extend(self.lab_label_to_indices[l])
        #
        # # adni
        # self.all_adni_index = []
        # for l in [0,1]:
        #     self.all_adni_index.extend(self.adni_label_to_indices[l])


    def __iter__(self):
        labels_idx = [list(np.copy(self.label_to_indices[l])) for l in self.label_choices]
        # ucsf_0 = list(np.copy(self.ucsf_label_to_indices[0]))
        # ucsf_1 = list(np.copy(self.ucsf_label_to_indices[1]))
        # ucsf_2 = list(np.copy(self.ucsf_label_to_indices[2]))
        # ucsf_3 = list(np.copy(self.ucsf_label_to_indices[3]))
        # lab_0 = list(np.copy(self.lab_label_to_indices[0]))
        # lab_2 = list(np.copy(self.lab_label_to_indices[2]))
        # adni_0 = list(np.copy(self.adni_label_to_indices[0]))
        # adni_1 = list(np.copy(self.adni_label_to_indices[1]))

        all_index = {}
        for idx, el in enumerate(labels_idx):
            all_index[idx] = el
        # all_index[0] = ucsf_0
        # all_index[1] = ucsf_1
        # all_index[2] = ucsf_2
        # all_index[3] = ucsf_3
        # all_index[4] = lab_0
        # all_index[5] = lab_2
        # all_index[6] = adni_0
        # all_index[7] = adni_1
        for i in range(0,len(self.label_choices)):
            np.random.shuffle(all_index[i])
        all_index_copy = copy.deepcopy(all_index)
        indices = []
        output = {}
        for i in range(0,self.num_batches):
            output[i]=[]
            self.get_index(all_index, output[i],all_index_copy)

        for i in range(0,self.num_batches):
            indices.extend(output[i])


        return iter(indices)

    def __len__(self):
        return 1


    def get_index(self, all_index, output, all_index_copy):

        class_size = 1
        for i in range(0,class_size):

            for j in range(0,len(self.label_choices)):
#                 if (i>=10) and(j ==0 or j == 2):
#                     if j == 0:
#                         index = 4
#                     else:
#                         index = 5

#                     if len(all_index[index])>0:
#                         output.append(all_index[index].pop(0))
#                     else:
#                         all_index[index] = list(np.copy(all_index_copy[index]))
#                         np.random.shuffle(all_index[index])

#                         output.append(all_index[index].pop(0))

#                 else:
                if len(all_index[j])>0:
                    output.append(all_index[j].pop(0))
                else:
                    print(j)
                    all_index[j] = list(np.copy(all_index_copy[j]))
                    np.random.shuffle(all_index[j])

                    output.append(all_index[j].pop(0))
