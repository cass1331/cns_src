from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from sklearn.model_selection import StratifiedKFold
import numpy as np
import nibabel as nib
import scipy as sp
import scipy.ndimage
from sklearn.metrics import mean_squared_error, r2_score

import sys
import argparse
import os
import glob
import csv

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

from tqdm import tqdm
import torchvision
from torch.optim.lr_scheduler import CosineAnnealingLR
from misc import CSVLogger
import copy
import gc
from transformation import super_transformation
from sampler import SuperSampler,MixedSampler
from class_sampler import ClassMixedSampler
from metadatanorm2 import MetadataNorm
import dcor
import pickle

from test import test, train, get_cf_kernel
from test import FocalLoss

import wandb



parser = argparse.ArgumentParser(description='ADNI')
parser.add_argument('--batch_size', type=int, default=64,
                    help='input batch size for training (default: 128)')
parser.add_argument('--epochs', type=int, default=100,
                    help='number of epochs to train (default: 100)')
parser.add_argument('--lr', type=float, default=0.0002,
                    help='learning rate')
parser.add_argument('--L2_lambda', type=float, default=0,
                    help='lambda')
parser.add_argument('--L1_lambda', type=float, default=0,
                    help='lambda')
parser.add_argument('--name', type=str, default='debug',
                    help='name of this run')
parser.add_argument('--fe_arch', type=str, default='baseline',
                    help='FeatureExtractor')
parser.add_argument('--dropout', type=float, default=0.1,
                    help='dropout in conv3d')
parser.add_argument('--fc_dropout', type=float, default=0.1,
                    help='dropout for fc')
parser.add_argument('--wd', type=float, default=0.01,
                    help='weight decay for adam')
parser.add_argument('--dyn_drop',action='store_true', default=False,
                    help='apply dynamic drop out ')
parser.add_argument('--alpha', type=float, nargs='+', default=0.5,
                    help='alpha for focal loss')
parser.add_argument('--gamma', type=float, default=2.0,
                    help='gamma for focal loss')
parser.add_argument('--seed', type=int, default=1,
                    help='seed')
args = parser.parse_args()


device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


seed = args.seed
torch.manual_seed(seed)
np.random.seed(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
if args.dyn_drop:

    args.name = args.name + '_dyn_drop_'+ '_fearch_' + args.fe_arch + '_bz_' + str(args.batch_size) +'_epoch_' + str(args.epochs) + '_lr_' + str(args.lr) + '_wd_' + str(args.wd) + '_alpha_' + str(args.alpha)+'_seed_'+str(seed)
else:
    args.name = args.name + '_fearch_' + args.fe_arch + '_bz_' + str(args.batch_size) + '_epoch_' + str(args.epochs) + '_lr_' + str(args.lr) + '_do_'+str(args.dropout) +'_fcdo_' + str(args.fc_dropout) + '_wd_' + str(args.wd) + '_alpha_'+str(args.alpha)+'_seed_'+str(seed)


wandb.init(project="train_ucsf", entity="jmanasse", config = {
  "learning_rate": args.lr,
  "epochs": args.epochs,
  "batch_size": args.batch_size,
  "fc dropout": args.fc_dropout,
  'dropout': args.dropout,
  "weight decay": args.wd,
  'loss type': 'focal loss'
})

print("device:",device)

L1_lambda = args.L1_lambda
L2_lambda = args.L2_lambda

alpha1 = args.alpha[0]
alpha2 = args.alpha[1]
ucsf_criterion_cd = FocalLoss(alpha = alpha1, gamma = args.gamma, dataset = 'ucsf')
ucsf_criterion_hiv = FocalLoss(alpha = alpha2, gamma = args.gamma, dataset = 'ucsf')
adni_criterion_cd = FocalLoss(alpha = alpha2, gamma = args.gamma, dataset = 'adni')

if __name__ == '__main__':
    log_path = '/scratch/users/jmanasse/mri_proj/logs/'
    filename = log_path + args.name +'.csv'
    os.makedirs(log_path, exist_ok=True)
    # csv_logger_sets =  CSVLogger( args,fieldnames=['ids','datasets'],filename='idss.csv')
    csv_logger = CSVLogger( args, fieldnames=['epoch', 'train_acc',  'test_acc',
                                              'train_loss','test_loss',
                                              'ucsf_train_acc',#'lab_train_acc','adni_train_acc',

                                              'ucsf_test_acc',#'lab_test_acc', 'adni_test_acc',
                                              'correlation_ctrl_train', 'correlation_hiv_train',
                                              #
                                              'correlation_ctrl_test', 'correlation_hiv_test',
                                              'train_ucsf_ctrl','train_ucsf_mci', 'train_ucsf_hiv','train_ucsf_mnd',
                                              # 'train_lab_ctrl', 'train_lab_hiv', 'train_adni_ctrl', 'train_adni_mci',

                                              'test_ucsf_ctrl','test_ucsf_mci','test_ucsf_hiv','test_ucsf_mnd',
                                              # 'test_lab_ctrl', 'test_lab_hiv','test_adni_ctrl', 'test_adni_mci',
                                             'train_distance','test_distance'],
                           filename=filename)

    filename2 = log_path + 'predictions/'+ args.name +'.csv'
    os.makedirs(log_path + 'predictions/', exist_ok=True)
    csv_logger_pred = CSVLogger( args, fieldnames=['epoch', 'id', 'dataset', 'CD_pred', 'HIV_pred', 'fold','CD_label', 'HIV_label'], filename=filename2)
    filename3 = log_path + 'predictions/' + args.name + 'corrs.csv'
    # csv_logger_corr = CSVLogger( args, fieldnames=['epoch', 'train', 'fold', 'final_corr0_age', 'final_corr1_age', 'final_corr0_gender', 'final_corr1_gender', 'intermediate_age', 'intermediate_gender'], filename=filename3)

    ## cross-validation
    best_accuracy_list = [0,0,0,0,0]
    best_epoch_list = [0,0,0,0,0]
    final_accuracy_list = [0,0,0,0,0]
    ave_valid_acc_50_list=[0,0,0,0,0]
    best_model_dict = {}

    acc_each_class_list = []

    #first load adni hiv cases

    #acc_each_dataset_list = []
    for fold in range(0,5):

        row = {'epoch': 'fold', 'train_acc': str(fold)}
        csv_logger.writerow(row)
        transformation = super_transformation()
        train_data = MRI_Dataset(fold = fold , stage= 'original_train',transform = transformation)
        test_data = MRI_Dataset(fold = fold , stage= 'original_test')

        train_loader = DataLoader(dataset=train_data,
                                  batch_size=None,
                                  sampler=MixedSampler(dataset=train_data,
                                                       batch_size=args.batch_size),
                                  shuffle=False,
                                  pin_memory=True,
                                  num_workers=3)
        test_loader = DataLoader(dataset=test_data ,
                                  batch_size=None,# to include all test images
                                  shuffle=True,
                                  pin_memory=True,
                                  num_workers=3)
        final_test_loader = DataLoader(dataset=test_data ,
                          batch_size=1,#args.batch_size,
                          shuffle=False,
                          pin_memory=True,
                          num_workers=3)
        print("Begin training fold ",fold)


        adni_data = []
        adni_test_data = []
        for i,(images, labels, actual_labels, datasets, ids, ages, genders) in enumerate(train_loader):

            #id_tracker.append(ids)
            if datasets == 'adni':
                # csv_logger_sets.writerow({'ids':ids, 'datasets':'train'})
                adni_data.append((images, datasets, ids, actual_labels, labels, ages, genders))

        for i,(images, labels, actual_labels, datasets, ids, ages, genders) in enumerate(test_loader):

            if datasets == 'adni':
                adni_test_data.append((images, datasets, ids, actual_labels, labels, ages, genders))


        adni_dataset = Specific_MRI_Dataset(adni_data)
        adni_test_dataset = Specific_MRI_Dataset(adni_test_data)

        adni_train_loader = DataLoader(dataset=adni_dataset,
                                  batch_size=args.batch_size,
                                  # sampler=ClassMixedSampler(dataset=ucsf_data, batch_size=args.batch_size),

                                  shuffle=False,
                                  pin_memory=True,
                                  num_workers=3)
        adni_test_loader = DataLoader(dataset=adni_test_dataset ,
                                  batch_size=args.batch_size,# to include all test images
                                  shuffle=True,
                                  pin_memory=True,
                                  num_workers=3)
        adni_final_test_loader = DataLoader(dataset=adni_test_dataset ,
                          batch_size=1,#args.batch_size,
                          shuffle=False,
                          pin_memory=True,
                          num_workers=3)

        cf_kernel  = get_cf_kernel(adni_train_loader)

        feature_extractor = fe(cf_kernel, trainset_size = len(adni_data), in_num_ch=1, img_size=(64, 64, 64), inter_num_ch=16,
                           fc_num_ch=16, kernel_size=3, conv_act='relu',
                           fe_arch=args.fe_arch, dropout=args.dropout,
                           fc_dropout = args.fc_dropout, batch_size = args.batch_size).to(device)

        classifier_adni = nn.Sequential(
                     # MetadataNorm(batch_size=args.batch_size, cf_kernel=cf_kernel, num_features = 2048, trainset_size = len(train_data)),
                    nn.Linear(2048, 128),
                    nn.LeakyReLU(),
                     # nn.BatchNorm1d(128),
                    MetadataNorm(batch_size=args.batch_size, cf_kernel=cf_kernel, num_features = 128, trainset_size = len(adni_data)),
                    nn.Linear(128,16),
                    nn.LeakyReLU(),
                     # nn.BatchNorm1d(16),
                   MetadataNorm(batch_size=args.batch_size, cf_kernel=cf_kernel, num_features = 16, trainset_size = len(adni_data)),
                     nn.Linear(16, 1),
                     MetadataNorm(batch_size=args.batch_size, cf_kernel=cf_kernel, num_features = 1, trainset_size = len(adni_data)),
                ).to(device)

        test_acc, test_accuracy_class, test_accuracy_dataset, best_accuracy, best_epoch, best_models = train(feature_extractor,  classifier_adni, adni_train_loader, adni_test_loader,adni_final_test_loader, cf_kernel=cf_kernel, fold = fold, dset='adni')

        feature_extractor, classifier_adni = best_models
        best_accuracy_list[fold] = best_accuracy
        final_accuracy_list[fold] = test_acc
        best_epoch_list[fold] = best_epoch


        test_acc, test_loss,test_accuracy_class, test_accuracy_dataset, test_distance = test(feature_extractor, classifier_adni, adni_train_loader, adni_criterion_cd, fold =fold, epoch = None, train = True, dset='adni')
        acc_each_class_list.append( test_accuracy_class)
        # acc_each_dataset_list.append( test_accuracy_dataset)
        row = {'epoch': 'fold', 'train_acc': str(fold)}
        csv_logger.writerow(row)
        model_path = '/scratch/users/jmanasse/mri_ckpts/'
        folder_name = args.name + '/'
        fold = 'fold_' + str(fold)
        new_dir = model_path + folder_name + fold +'/'
        print("Woohoo", new_dir)
        os.makedirs(new_dir, exist_ok=True)
        torch.save(feature_extractor.state_dict(), new_dir + 'feature_extractor.pt')
        torch.save(classifier_adni.state_dict(), new_dir + 'classifier_adni.pt')

    for fold in range(0,5):

        row = {'epoch': 'fold', 'train_acc': str(fold)}
        csv_logger.writerow(row)
        transformation = super_transformation()
        train_data = MRI_Dataset(fold = fold , stage= 'original_train',transform = transformation)
        test_data = MRI_Dataset(fold = fold , stage= 'original_test')

        train_loader = DataLoader(dataset=train_data,
                                  batch_size=None,
                                  sampler=MixedSampler(dataset=train_data,
                                                       batch_size=args.batch_size),
                                  shuffle=False,
                                  pin_memory=True,
                                  num_workers=3)
        test_loader = DataLoader(dataset=test_data ,
                                  batch_size=None,# to include all test images
                                  shuffle=True,
                                  pin_memory=True,
                                  num_workers=3)
        final_test_loader = DataLoader(dataset=test_data ,
                          batch_size=1,#args.batch_size,
                          shuffle=False,
                          pin_memory=True,
                          num_workers=3)
        print("Begin training fold ",fold)

#
        ucsf_data = []
        adni_data = []
        for i,(images, labels, actual_labels, datasets, ids, ages, genders) in enumerate(train_loader):

            #id_tracker.append(ids)
            if datasets == 'ucsf':
                # csv_logger_sets.writerow({'ids':ids, 'datasets':'train'})
                ucsf_data.append((images, datasets, ids, actual_labels, labels, ages, genders))

        ucsf_test_data = []
        for i,(images, labels, actual_labels, datasets, ids, ages, genders) in enumerate(test_loader):

            if datasets == 'ucsf':
                ucsf_test_data.append((images, datasets, ids, actual_labels, labels, ages, genders))

        # print(len(ucsf_data), len(ucsf_test_data))
        # all_data = ucsf_data + ucsf_test_data
        # print(len(all_data))
        ucsf_dataset = Specific_MRI_Dataset(ucsf_data)
        ucsf_test_dataset = Specific_MRI_Dataset(ucsf_test_data)


        cf_kernel  = get_cf_kernel(ucsf_train_loader)

        fex = fe(cf_kernel, trainset_size = len(ucsf_data), in_num_ch=1, img_size=(64, 64, 64), inter_num_ch=16,
                           fc_num_ch=16, kernel_size=3, conv_act='relu',
                           fe_arch=args.fe_arch, dropout=args.dropout,
                           fc_dropout = args.fc_dropout, batch_size = args.batch_size).to(device)

        # model_path = '/scratch/users/jmanasse/mri_ckpts/'
        # folder_name = args.name + '/'
        # fold = 'fold_' + str(fold)
        # new_dir = model_path + folder_name + fold +'/'
        fex.load_state_dict(best_models[0].state_dict())

        classifier_ucsf = nn.Sequential(
                     # MetadataNorm(batch_size=args.batch_size, cf_kernel=cf_kernel, num_features = 2048, trainset_size = len(train_data)),
                    nn.Linear(2048, 128),
                    nn.LeakyReLU(),
                     # nn.BatchNorm1d(128),
                    MetadataNorm(batch_size=args.batch_size, cf_kernel=cf_kernel, num_features = 128, trainset_size = len(ucsf_data)),
                    nn.Linear(128,16),
                    nn.LeakyReLU(),
                     # nn.BatchNorm1d(16),
                   MetadataNorm(batch_size=args.batch_size, cf_kernel=cf_kernel, num_features = 16, trainset_size = len(ucsf_data)),
                     nn.Linear(16, 2),
                     MetadataNorm(batch_size=args.batch_size, cf_kernel=cf_kernel, num_features = 2, trainset_size = len(ucsf_data)),
                ).to(device)

        new = classifier_ucsf.state_dict()
        old = best_models[1].state_dict()

        for weight_ in old:
            if weight_.startswtih('6') or weight_.startswtih('7'):
                break
            else:
                new[weight_] = old[weight_]

        ucsf_train_loader = DataLoader(dataset=ucsf_dataset,
                                  batch_size=args.batch_size,
                                  # sampler=ClassMixedSampler(dataset=ucsf_data, batch_size=args.batch_size),

                                  shuffle=False,
                                  pin_memory=True,
                                  num_workers=3)
        ucsf_test_loader = DataLoader(dataset=ucsf_test_dataset ,
                                  batch_size=args.batch_size,# to include all test images
                                  shuffle=True,
                                  pin_memory=True,
                                  num_workers=3)
        ucsf_final_test_loader = DataLoader(dataset=ucsf_test_dataset ,
                          batch_size=1,#args.batch_size,
                          shuffle=False,
                          pin_memory=True,
                          num_workers=3)



        cf_kernel  = get_cf_kernel(ucsf_train_loader)

        test_acc, test_accuracy_class, test_accuracy_dataset, best_accuracy, best_epoch, best_things = train(fex,  classifier_ucsf, ucsf_train_loader, ucsf_test_loader,ucsf_final_test_loader, cf_kernel=cf_kernel, fold = fold, dset='ucsf')


        fex, classifier_ucsf = best_things
        best_accuracy_list[fold] = best_accuracy
        final_accuracy_list[fold] = test_acc
        best_epoch_list[fold] = best_epoch


        test_acc, test_loss,test_accuracy_class, test_accuracy_dataset, test_distance = test(fex, classifier_ucsf, ucsf_train_loader, [ucsf_criterion_cd,ucsf_criterion_hiv], fold =fold, epoch = None, train = True, dset='ucsf')
        acc_each_class_list.append( test_accuracy_class)
        acc_each_dataset_list.append( test_accuracy_dataset)
        row = {'epoch': 'fold', 'train_acc': str(fold)}
        csv_logger.writerow(row)
        model_path = '/scratch/users/jmanasse/mri_ckpts/'
        folder_name = args.name + '/'
        fold = 'fold_' + str(fold)
        new_dir = model_path + folder_name + fold +'/'
        print("Woohoo", new_dir)
        os.makedirs(new_dir, exist_ok=True)
        torch.save(fex.state_dict(), new_dir + 'fex.pt')
        torch.save(classifier_ucsf.state_dict(), new_dir + 'classifier_ucsf.pt')





    print('best_accuracy', best_accuracy_list)
    print('final_accuracy',final_accuracy_list)
    print('best_epoch', best_epoch_list)
    print('ave_valid_acc_50',ave_valid_acc_50_list)
    print('acc_each_class',acc_each_class_list)
    #print('acc_each_dataset',acc_each_dataset_list)

    ave_acc_each_class_list, ave_acc_each_dataset_list = average_results(acc_each_class_list,acc_each_dataset_list)
    print(ave_acc_each_class_list)
#    print(ave_acc_each_dataset_list)

    # finish the loggers
    csv_logger.final(best_accuracy_list,final_accuracy_list,
                     best_epoch_list,ave_valid_acc_50_list,
                     acc_each_class_list)#acc_each_dataset_list)
    csv_logger.close()
    csv_logger_pred.close()
    csv_logger_corr.close()
