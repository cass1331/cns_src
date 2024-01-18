import os
import shutil
import time
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import nibabel as nib
import copy
import pickle
import gc

###### This code splits the data into train/val/test
###### train/val includes CTRL/MCI/HIV, test is HAND exclusive

seed = 72

np.random.seed(seed)

def get_fold_index(data,labels,datasets,ids,ages,genders,npzs,fold):
    #shuffle data
    indices = np.arange(datasets.shape[0])
    np.random.shuffle(indices)
    data = data[indices]
    labels = labels[indices]
    datasets = datasets[indices]
    ids = ids[indices]
    ages = ages[indices]
    genders = genders[indices]
    npzs = npzs[indices]
    # set fold
    fold_idx = []

    for i in range(0,len(labels)):
        fold_idx.append(i%5)

    fold_idx = np.array(fold_idx)


    train_idx = (fold_idx != fold)
    val_idx = (fold_idx == fold)

    train_data = data[train_idx]
    train_label = labels[train_idx]
    train_dataset = datasets[train_idx]
    train_ids = ids[train_idx]
    train_ages = ages[train_idx]
    train_genders = genders[train_idx]
    train_npzs = npzs[train_idx]


    val_data = data[val_idx]
    val_label = labels[val_idx]
    val_dataset = datasets[val_idx]
    val_ids = ids[val_idx]
    val_ages = ages[val_idx]
    val_genders = genders[val_idx]
    val_npzs = npzs[val_idx]


    return train_data, train_label, train_dataset, train_ids, train_ages, train_genders, train_npzs, val_data, val_label, val_dataset, val_ids, val_ages, val_genders, val_npzs

def get_CE_label(actual_labels, name):
    actual_labels = np.array(actual_labels)
    if name == 'ucsf':
        actual_label_0 = actual_labels[actual_labels==0]
        actual_label_1 = actual_labels[actual_labels==1]
        actual_label_2 = actual_labels[actual_labels==2]
        actual_label_3 = actual_labels[actual_labels==3]
        CE_label = np.zeros((len(actual_labels),2))
        CE_label[actual_labels==1,0] = 1
        CE_label[actual_labels==2,1] = 1
        CE_label[actual_labels==3] = 1
    return CE_label

def get_data(data,labels,dataset,ids,ages,genders,npzs, name,fold):
    # set train, val
    train_data, train_actual_label, train_dataset, train_ids, train_ages, train_genders,  train_npzs, val_data, val_actual_label, val_dataset, val_ids, val_ages, val_genders, val_npzs = get_fold_index(data,labels,dataset,ids,ages,genders,npzs,fold)

    train_CE_label = get_CE_label(train_actual_label, name)
    val_CE_label = get_CE_label(val_actual_label, name)

    return train_data, train_actual_label, train_CE_label, train_dataset, train_ids,  train_ages, train_genders, train_npzs, val_data, val_actual_label, val_CE_label, val_dataset, val_ids, val_ages, val_genders, val_npzs



def process_data(data_ucsf,labels_ucsf,datasets_ucsf,ids_ucsf,ages_ucsf,genders_ucsf, npzs_ucsf, fold):
    #dump cross-val data
    name = 'ucsf'
    train_data_ucsf, train_actual_label_ucsf, train_CE_label_ucsf, train_dataset_ucsf, train_ids_ucsf, train_ages_ucsf, train_genders_ucsf, train_npzs_ucsf, val_data_ucsf, val_actual_label_ucsf, val_CE_label_ucsf, val_dataset_ucsf, val_ids_ucsf, val_ages_ucsf, val_genders_ucsf, val_npzs_ucsf = get_data(data_ucsf,labels_ucsf,datasets_ucsf,ids_ucsf,ages_ucsf,genders_ucsf, npzs_ucsf, name,fold)

    print('length of train data:',len(train_data_ucsf ))
    print('length of train label:',len(train_actual_label_ucsf ))

    print('length of val data:',len(val_data_ucsf ))
    print('length of val label:',len(val_actual_label_ucsf ))

    # train data
    pickle_name = '/scratch/users/jmanasse/threehead/original_train_data_'+str(fold)+'.pickle'
    with open(pickle_name, 'wb') as handle:
        pickle.dump(train_data_ucsf, handle, protocol=pickle.HIGHEST_PROTOCOL)
    pickle_name = '/scratch/users/jmanasse/threehead/original_train_label_'+str(fold)+'.pickle'
    with open(pickle_name, 'wb') as handle:
        pickle.dump(train_CE_label_ucsf, handle, protocol=pickle.HIGHEST_PROTOCOL)

    pickle_name = '/scratch/users/jmanasse/threehead/original_train_actual_label_'+str(fold)+'.pickle'
    with open(pickle_name, 'wb') as handle:
        pickle.dump(train_actual_label_ucsf, handle, protocol=pickle.HIGHEST_PROTOCOL)

    pickle_name = '/scratch/users/jmanasse/threehead/original_train_dataset_'+str(fold)+'.pickle'
    with open(pickle_name, 'wb') as handle:
        pickle.dump(train_dataset_ucsf, handle, protocol=pickle.HIGHEST_PROTOCOL)

    pickle_name = '/scratch/users/jmanasse/threehead/train_id_'+str(fold)+'.pickle'
    with open(pickle_name, 'wb') as handle:
        pickle.dump(train_ids_ucsf, handle, protocol=pickle.HIGHEST_PROTOCOL)

    pickle_name = '/scratch/users/jmanasse/threehead/train_age_'+str(fold)+'.pickle'
    with open(pickle_name, 'wb') as handle:
        pickle.dump(train_ages_ucsf, handle, protocol=pickle.HIGHEST_PROTOCOL)

    pickle_name = '/scratch/users/jmanasse/threehead/train_gender_'+str(fold)+'.pickle'
    with open(pickle_name, 'wb') as handle:
        pickle.dump(train_genders_ucsf, handle, protocol=pickle.HIGHEST_PROTOCOL)
    
    pickle_name = '/scratch/users/jmanasse/threehead/train_npz_'+str(fold)+'.pickle'
    with open(pickle_name, 'wb') as handle:
        pickle.dump(train_npzs_ucsf, handle, protocol=pickle.HIGHEST_PROTOCOL)

    # test data
    pickle_name = '/scratch/users/jmanasse/threehead/original_val_data_'+str(fold)+'.pickle'
    with open(pickle_name, 'wb') as handle:
        pickle.dump(val_data_ucsf, handle, protocol=pickle.HIGHEST_PROTOCOL)
    pickle_name = '/scratch/users/jmanasse/threehead/original_val_label_'+str(fold)+'.pickle'
    with open(pickle_name, 'wb') as handle:
        pickle.dump(val_CE_label_ucsf, handle, protocol=pickle.HIGHEST_PROTOCOL)

    pickle_name = '/scratch/users/jmanasse/threehead/original_val_actual_label_'+str(fold)+'.pickle'
    with open(pickle_name, 'wb') as handle:
        pickle.dump(val_actual_label_ucsf, handle, protocol=pickle.HIGHEST_PROTOCOL)

    pickle_name = '/scratch/users/jmanasse/threehead/original_val_dataset_'+str(fold)+'.pickle'
    with open(pickle_name, 'wb') as handle:
        pickle.dump(val_dataset_ucsf, handle, protocol=pickle.HIGHEST_PROTOCOL)

    pickle_name = '/scratch/users/jmanasse/threehead/val_id_'+str(fold)+'.pickle'
    with open(pickle_name, 'wb') as handle:
        pickle.dump(val_ids_ucsf, handle, protocol=pickle.HIGHEST_PROTOCOL)

    pickle_name = '/scratch/users/jmanasse/threehead/val_age_'+str(fold)+'.pickle'
    with open(pickle_name, 'wb') as handle:
        pickle.dump(val_ages_ucsf, handle, protocol=pickle.HIGHEST_PROTOCOL)

    pickle_name = '/scratch/users/jmanasse/threehead/val_gender_'+str(fold)+'.pickle'
    with open(pickle_name, 'wb') as handle:
        pickle.dump(val_genders_ucsf, handle, protocol=pickle.HIGHEST_PROTOCOL)

    pickle_name = '/scratch/users/jmanasse/threehead/val_npz_'+str(fold)+'.pickle'
    with open(pickle_name, 'wb') as handle:
        pickle.dump(val_npzs_ucsf, handle, protocol=pickle.HIGHEST_PROTOCOL)

def process_test_data(test_data_ucsf,test_labels_ucsf,test_datasets_ucsf,test_ids_ucsf,test_ages_ucsf,test_genders_ucsf,test_npzs_ucsf):
    #dump cross-val data
    name = 'ucsf'
    test_actual_label_ucsf = test_labels_ucsf
    test_CE_label_ucsf = get_CE_label(test_labels_ucsf,'ucsf')
    print('length of test data:',len(test_data_ucsf ))
    print('length of test label:',len(test_actual_label_ucsf ))

    # train data
    pickle_name = '/scratch/users/jmanasse/threehead/original_test_data_'+str(fold)+'.pickle'
    with open(pickle_name, 'wb') as handle:
        pickle.dump(test_data_ucsf, handle, protocol=pickle.HIGHEST_PROTOCOL)
    pickle_name = '/scratch/users/jmanasse/threehead/original_test_label_'+str(fold)+'.pickle'
    with open(pickle_name, 'wb') as handle:
        pickle.dump(test_CE_label_ucsf, handle, protocol=pickle.HIGHEST_PROTOCOL)

    pickle_name = '/scratch/users/jmanasse/threehead/original_test_actual_label_'+str(fold)+'.pickle'
    with open(pickle_name, 'wb') as handle:
        pickle.dump(test_actual_label_ucsf, handle, protocol=pickle.HIGHEST_PROTOCOL)

    pickle_name = '/scratch/users/jmanasse/threehead/original_test_dataset_'+str(fold)+'.pickle'
    with open(pickle_name, 'wb') as handle:
        pickle.dump(test_datasets_ucsf, handle, protocol=pickle.HIGHEST_PROTOCOL)

    pickle_name = '/scratch/users/jmanasse/threehead/test_id_'+str(fold)+'.pickle'
    with open(pickle_name, 'wb') as handle:
        pickle.dump(test_ids_ucsf, handle, protocol=pickle.HIGHEST_PROTOCOL)

    pickle_name = '/scratch/users/jmanasse/threehead/test_age_'+str(fold)+'.pickle'
    with open(pickle_name, 'wb') as handle:
        pickle.dump(test_ages_ucsf, handle, protocol=pickle.HIGHEST_PROTOCOL)

    pickle_name = '/scratch/users/jmanasse/threehead/test_gender_'+str(fold)+'.pickle'
    with open(pickle_name, 'wb') as handle:
        pickle.dump(test_genders_ucsf, handle, protocol=pickle.HIGHEST_PROTOCOL)
    
    pickle_name = '/scratch/users/jmanasse/threehead/test_npz_'+str(fold)+'.pickle'
    with open(pickle_name, 'wb') as handle:
        pickle.dump(test_npzs_ucsf, handle, protocol=pickle.HIGHEST_PROTOCOL)
patch_x = 64
patch_y = 64
patch_z = 64

###LOAD IN PICKLE FILE WITH ALL UCSF DATA##
with open('../data/all_ucsf.pickle', 'rb') as handle:
    all_data_ucsf = pickle.load(handle)

# ------------------------------------------------------------------------------------------------------------------------

def collect_files(all_data,name):
    ''' collect all the loaded data and separate out HAND samples'''
    filenames = []
    labels = []
    datasets= []
    cor_datasets= []
    ids = []
    ages = []
    genders = []
    npzs = []
    for i in range(0,len(all_data)):
        filenames.append(all_data[i][0])
        labels.append(all_data[i][1])
        datasets.append(all_data[i][2])
        cor_datasets.append(all_data[i][3])
        ids.append(all_data[i][4])
        ages.append(all_data[i][5])
        genders.append(all_data[i][6])
        npzs.append(all_data[i][7])

    #make a copy of the labels to use as an indexer
    labels_copy = np.array(copy.deepcopy(labels)) 
    #print(labels_copy)
    filenames = np.array(filenames)
    labels = np.array(labels)
    datasets=np.array(datasets)
    cor_datasets=np.array(cor_datasets)
    ids = np.array(ids)
    ages = np.array(ages)
    genders = np.array(genders)
    npzs = np.array(npzs)

    train_filenames = filenames[np.where(labels_copy != 3)]
    train_labels = labels[np.where(labels_copy != 3)]
    train_datasets = datasets[np.where(labels_copy != 3)]
    train_cor_datasets = cor_datasets[np.where(labels_copy != 3)]
    train_ids = ids[np.where(labels_copy != 3)]
    train_ages = ages[np.where(labels_copy != 3)]
    train_genders = genders[np.where(labels_copy != 3)]
    train_npzs = npzs[np.where(labels_copy != 3)]

    test_filenames = filenames[np.where(labels_copy == 3)]
    test_labels = labels[np.where(labels_copy == 3)]
    test_datasets=datasets[np.where(labels_copy == 3)]
    test_cor_datasets=cor_datasets[np.where(labels_copy == 3)]
    test_ids = ids[np.where(labels_copy == 3)]
    test_ages = ages[np.where(labels_copy == 3)]
    test_genders = genders[np.where(labels_copy == 3)]
    test_npzs = npzs[np.where(labels_copy == 3)]

    train_num = len(train_filenames) 
    test_num = len(test_filenames)
    subject_num = train_num + test_num
    data = np.zeros((train_num, 1, patch_x, patch_y, patch_z))
    test_data = np.zeros((test_num, 1, patch_x, patch_y, patch_z))
    i = 0
    for filename in train_filenames:
        img = nib.load(filename)
        img_data = img.get_fdata()

        data[i,0,:,:,:] = img_data[0:patch_x, 0:patch_y, 0:patch_z]
        data[i,0,:,:,:] = (data[i,0,:,:,:] - np.mean(data[i,0,:,:,:])) / np.std(data[i,0,:,:,:])
        i += 1
    i = 0
    for filename in test_filenames:
        img = nib.load(filename)
        img_data = img.get_fdata()

        test_data[i,0,:,:,:] = img_data[0:patch_x, 0:patch_y, 0:patch_z]
        test_data[i,0,:,:,:] = (test_data[i,0,:,:,:] - np.mean(test_data[i,0,:,:,:])) / np.std(test_data[i,0,:,:,:])
        i += 1

    print('total number of '+name+' data:', subject_num )
    return data,train_labels,train_datasets,train_ids,train_ages,train_genders,train_npzs, test_data, test_labels, test_datasets, test_ids,test_ages,test_genders,test_npzs

# ------------------------------------------------------------------------------------------------------------------------

data_ucsf,labels_ucsf,datasets_ucsf,ids_ucsf,ages_ucsf,genders_ucsf,npzs_ucsf,test_data_ucsf,test_labels_ucsf,test_datasets_ucsf,test_ids_ucsf,test_ages_ucsf,test_genders_ucsf,test_npzs_ucsf = collect_files(all_data_ucsf,name='ucsf')

for fold in range(5):
    process_data(data_ucsf,labels_ucsf,datasets_ucsf,ids_ucsf,ages_ucsf,genders_ucsf, npzs_ucsf,
                 fold)

process_test_data(test_data_ucsf,test_labels_ucsf,test_datasets_ucsf,test_ids_ucsf,test_ages_ucsf,test_genders_ucsf,test_npzs_ucsf)
