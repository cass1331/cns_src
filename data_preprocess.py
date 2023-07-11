import os
import shutil
import time
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import nibabel as nib

import pickle
import gc


def get_fold_index(data,labels,datasets,ids,ages,genders,fold):
        # set fold  
    fold_idx = []
    
    for i in range(0,len(labels)):
        fold_idx.append(i%5)

    fold_idx = np.array(fold_idx)
    
    
    train_idx = (fold_idx != fold)
    test_idx = (fold_idx == fold)
    
    train_data = data[train_idx]
    train_label = labels[train_idx]
    train_dataset = datasets[train_idx]
    train_ids = ids[train_idx]
    train_ages = ages[train_idx]
    train_genders = genders[train_idx]

    
    test_data = data[test_idx]
    test_label = labels[test_idx]
    test_dataset = datasets[test_idx]
    test_ids = ids[test_idx]
    test_ages = ages[test_idx]
    test_genders = genders[test_idx]

    
    return train_data, train_label, train_dataset, train_ids, train_ages, train_genders, test_data, test_label, test_dataset, test_ids, test_ages, test_genders

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


    elif name == 'lab':
        actual_label_0 = actual_labels[actual_labels==0] 
        actual_label_2 = actual_labels[actual_labels==2]

        CE_label = np.zeros((len(actual_labels),2))

        CE_label[actual_labels==2,1] = 1 
    elif name == 'adni':
        actual_label_0 = actual_labels[actual_labels==0] 
        actual_label_1 = actual_labels[actual_labels==1]

        CE_label = np.zeros((len(actual_labels),2))

        CE_label[actual_labels==1,0] = 1
    
    return CE_label
    
def get_data(data,labels,dataset,ids,ages,genders,name,fold):


    
    # set train, val, test
    train_data, train_actual_label, train_dataset, train_ids, train_ages, train_genders, test_data, test_actual_label, test_dataset, test_ids, test_ages, test_genders = get_fold_index(data,labels,dataset,ids,ages,genders,fold)
    
    train_CE_label = get_CE_label(train_actual_label, name)
    test_CE_label = get_CE_label(test_actual_label, name)

    
   
    return train_data, train_actual_label, train_CE_label, train_dataset, train_ids,  train_ages, train_genders, test_data, test_actual_label,test_CE_label, test_dataset, test_ids, test_ages, test_genders



    
def process_data(data_ucsf,labels_ucsf,datasets_ucsf,ids_ucsf,ages_ucsf,genders_ucsf,data_lab,labels_lab,datasets_lab,
                 ids_lab,ages_lab,genders_lab,data_adni,labels_adni,datasets_adni,ids_adni,ages_adni,genders_adni,fold):
    name = 'ucsf'
    train_data_ucsf, train_actual_label_ucsf, train_CE_label_ucsf, train_dataset_ucsf, train_ids_ucsf, train_ages_ucsf, train_genders_ucsf, test_data_ucsf, test_actual_label_ucsf, test_CE_label_ucsf, test_dataset_ucsf, test_ids_ucsf, test_ages_ucsf, test_genders_ucsf = get_data(data_ucsf,labels_ucsf,datasets_ucsf,ids_ucsf,ages_ucsf,genders_ucsf, name,fold)
    name = 'lab'
    train_data_lab, train_actual_label_lab, train_CE_label_lab, train_dataset_lab, train_ids_lab, train_ages_lab, train_genders_lab, test_data_lab, test_actual_label_lab, test_CE_label_lab, test_dataset_lab, test_ids_lab, test_ages_lab, test_genders_lab = get_data(data_lab,labels_lab,datasets_lab,ids_lab,ages_lab,genders_lab, name,fold)
    name = 'adni'
    train_data_adni, train_actual_label_adni, train_CE_label_adni, train_dataset_adni, train_ids_adni, train_ages_adni, train_genders_adni, test_data_adni, test_actual_label_adni, test_CE_label_adni, test_dataset_adni, test_ids_adni, test_ages_adni, test_genders_adni = get_data(data_adni,labels_adni,datasets_adni,ids_adni,ages_adni,genders_adni, name, fold)

    train_data = np.concatenate((train_data_ucsf,
                                 train_data_lab,
                                train_data_adni), axis=0)
    train_actual_label = np.concatenate((train_actual_label_ucsf,
                                         train_actual_label_lab,
                                        train_actual_label_adni), axis=0)
    train_CE_label = np.concatenate((train_CE_label_ucsf,
                                     train_CE_label_lab,
                                    train_CE_label_adni), axis=0)
    train_dataset = np.concatenate((train_dataset_ucsf,
                                 train_dataset_lab,
                                   train_dataset_adni), axis=0)
    train_ids = np.concatenate((train_ids_ucsf,
                                 train_ids_lab,
                               train_ids_adni), axis=0)
    train_ages = np.concatenate((train_ages_ucsf,
                                 train_ages_lab,
                               train_ages_adni), axis=0)
    train_genders = np.concatenate((train_genders_ucsf,
                                 train_genders_lab,
                               train_genders_adni), axis=0)
                               
    test_data = np.concatenate((test_data_ucsf,
                                 test_data_lab,
                               test_data_adni), axis=0)
    test_actual_label = np.concatenate((test_actual_label_ucsf,
                                         test_actual_label_lab,
                                       test_actual_label_adni), axis=0)
    test_CE_label = np.concatenate((test_CE_label_ucsf,
                                     test_CE_label_lab,
                                   test_CE_label_adni), axis=0)
    test_dataset = np.concatenate((test_dataset_ucsf,
                                 test_dataset_lab,
                                  test_dataset_adni), axis=0)
    test_ids = np.concatenate((test_ids_ucsf,
                                 test_ids_lab,
                              test_ids_adni), axis=0)     
    test_ages = np.concatenate((test_ages_ucsf,
                                 test_ages_lab,
                              test_ages_adni), axis=0)  
    test_genders = np.concatenate((test_genders_ucsf,
                             test_genders_lab,
                          test_genders_adni), axis=0)  

    print('length of train data:',len(train_data ))
    print('length of train label:',len(train_actual_label ))                      

    print('length of test data:',len(test_data ))
    print('length of test label:',len(test_actual_label ))    

    # train data
    pickle_name = '/scratch/users/jmanasse/threehead/original_train_data_'+str(fold)+'.pickle'
    with open(pickle_name, 'wb') as handle:
        pickle.dump(train_data, handle, protocol=pickle.HIGHEST_PROTOCOL)
    pickle_name = '/scratch/users/jmanasse/threehead/original_train_label_'+str(fold)+'.pickle'
    with open(pickle_name, 'wb') as handle:
        pickle.dump(train_CE_label, handle, protocol=pickle.HIGHEST_PROTOCOL)    

    pickle_name = '/scratch/users/jmanasse/threehead/original_train_actual_label_'+str(fold)+'.pickle'
    with open(pickle_name, 'wb') as handle:
        pickle.dump(train_actual_label, handle, protocol=pickle.HIGHEST_PROTOCOL) 

    pickle_name = '/scratch/users/jmanasse/threehead/original_train_dataset_'+str(fold)+'.pickle'
    with open(pickle_name, 'wb') as handle:
        pickle.dump(train_dataset, handle, protocol=pickle.HIGHEST_PROTOCOL) 

    pickle_name = '/scratch/users/jmanasse/threehead/train_id_'+str(fold)+'.pickle'
    with open(pickle_name, 'wb') as handle:
        pickle.dump(train_ids, handle, protocol=pickle.HIGHEST_PROTOCOL) 
    
    pickle_name = '/scratch/users/jmanasse/threehead/train_age_'+str(fold)+'.pickle'
    with open(pickle_name, 'wb') as handle:
        pickle.dump(train_ages, handle, protocol=pickle.HIGHEST_PROTOCOL) 
        
    pickle_name = '/scratch/users/jmanasse/threehead/train_gender_'+str(fold)+'.pickle'
    with open(pickle_name, 'wb') as handle:
        pickle.dump(train_genders, handle, protocol=pickle.HIGHEST_PROTOCOL) 
        
    # test data    
    pickle_name = '/scratch/users/jmanasse/threehead/original_test_data_'+str(fold)+'.pickle'
    with open(pickle_name, 'wb') as handle:
        pickle.dump(test_data, handle, protocol=pickle.HIGHEST_PROTOCOL)
    pickle_name = '/scratch/users/jmanasse/threehead/original_test_label_'+str(fold)+'.pickle'
    with open(pickle_name, 'wb') as handle:
        pickle.dump(test_CE_label, handle, protocol=pickle.HIGHEST_PROTOCOL)    

    pickle_name = '/scratch/users/jmanasse/threehead/original_test_actual_label_'+str(fold)+'.pickle'
    with open(pickle_name, 'wb') as handle:
        pickle.dump(test_actual_label, handle, protocol=pickle.HIGHEST_PROTOCOL) 

    pickle_name = '/scratch/users/jmanasse/threehead/original_test_dataset_'+str(fold)+'.pickle'
    with open(pickle_name, 'wb') as handle:
        pickle.dump(test_dataset, handle, protocol=pickle.HIGHEST_PROTOCOL) 

    pickle_name = '/scratch/users/jmanasse/threehead/test_id_'+str(fold)+'.pickle'
    with open(pickle_name, 'wb') as handle:
        pickle.dump(test_ids, handle, protocol=pickle.HIGHEST_PROTOCOL)
    
    pickle_name = '/scratch/users/jmanasse/threehead/test_age_'+str(fold)+'.pickle'
    with open(pickle_name, 'wb') as handle:
        pickle.dump(test_ages, handle, protocol=pickle.HIGHEST_PROTOCOL) 
        
    pickle_name = '/scratch/users/jmanasse/threehead/test_gender_'+str(fold)+'.pickle'
    with open(pickle_name, 'wb') as handle:
        pickle.dump(test_genders, handle, protocol=pickle.HIGHEST_PROTOCOL) 

patch_x = 64
patch_y = 64
patch_z = 64

with open('../data/all_ucsf.pickle', 'rb') as handle:
    all_data_ucsf = pickle.load(handle)
with open('../data/all_lab.pickle', 'rb') as handle:
    all_data_lab = pickle.load(handle)
with open('../data/all_adni.pickle', 'rb') as handle:
    all_data_adni = pickle.load(handle) 

# ------------------------------------------------------------------------------------------------------------------------ 

def collect_files(all_data,name):
    filenames = []
    labels = []
    datasets= []
    cor_datasets= []
    ids = []
    ages = []
    genders = []
    for i in range(0,len(all_data)):
        filenames.append(all_data[i][0])
        labels.append(all_data[i][1])
        datasets.append(all_data[i][2])
        cor_datasets.append(all_data[i][3])
        ids.append(all_data[i][4])
        ages.append(all_data[i][5])
        genders.append(all_data[i][6])

    filenames = np.array(filenames)
    labels = np.array(labels)
    datasets=np.array(datasets)
    cor_datasets=np.array(cor_datasets)
    ids = np.array(ids)
    ages = np.array(ages)
    genders = np.array(genders)

    subject_num = len(filenames)
    data = np.zeros((subject_num, 1, patch_x, patch_y, patch_z))
    i = 0
    for filename in filenames:
        img = nib.load(filename)
        img_data = img.get_fdata()

        data[i,0,:,:,:] = img_data[0:patch_x, 0:patch_y, 0:patch_z] 
        data[i,0,:,:,:] = (data[i,0,:,:,:] - np.mean(data[i,0,:,:,:])) / np.std(data[i,0,:,:,:])
        i += 1
        
    print('total number of '+name+' data:', subject_num )
    return data,labels,datasets,ids,ages,genders

    
    
    

# ------------------------------------------------------------------------------------------------------------------------ 

data_ucsf,labels_ucsf,datasets_ucsf,ids_ucsf,ages_ucsf,genders_ucsf = collect_files(all_data_ucsf,name='ucsf')
data_lab,labels_lab,datasets_lab,ids_lab,ages_lab,genders_lab = collect_files(all_data_lab,name='lab')
data_adni,labels_adni,datasets_adni,ids_adni,ages_adni,genders_adni = collect_files(all_data_adni,name='adni')

for fold in range(5):
    process_data(data_ucsf,labels_ucsf,datasets_ucsf,ids_ucsf,ages_ucsf,genders_ucsf,
                 data_lab,labels_lab,datasets_lab,ids_lab,ages_lab,genders_lab,
                 data_adni,labels_adni,datasets_adni,ids_adni,ages_adni,genders_adni,
                 fold)
    
    
    
