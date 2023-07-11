from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from sklearn.model_selection import StratifiedKFold
import numpy as np
import nibabel as nib
import pandas as pd
import scipy as sp
import random
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
import math
from torch.utils.data.dataset import ConcatDataset

from torch.utils.data import DataLoader

from tqdm import tqdm
import torchvision
from torch.optim.lr_scheduler import CosineAnnealingLR
from misc import CSVLogger
import copy
import gc
from transformation import super_transformation
from sampler import SuperSampler,MixedSampler
from metadatanorm2 import MetadataNorm
import dcor
import pickle


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




print("device:",device)



class FocalLoss(nn.Module):
    def __init__(self, alpha=0.1, gamma=2, dataset = 'ucsf'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.criterion = torch.nn.BCEWithLogitsLoss(reduction= 'none').to(device)
#         self.criterion = torch.nn.BCEWithLogitsLoss().to(device)
        self.dataset = dataset

    def forward(self, inputs, targets):
        BCE_loss = self.criterion(inputs, targets)

        if inputs.shape[0] == 1:
            return BCE_loss

        pt = torch.exp(-BCE_loss)
        F_loss= 0


        F_loss_pos = self.alpha * (1-pt[targets==1])**self.gamma * BCE_loss[targets==1]
        F_loss_neg = (1-self.alpha) * (1-pt[targets==0])**self.gamma * BCE_loss[targets==0]

        if inputs.shape[0] == 1:
            if F_loss_pos.nelement() > 0:
                return F_loss_pos
            else:
                return F_loss_neg

        F_loss += (torch.mean(F_loss_pos)+torch.mean(F_loss_neg))/2

        return F_loss


L1_lambda = args.L1_lambda
L2_lambda = args.L2_lambda

alpha1 = args.alpha[0]
alpha2 = args.alpha[1]
ucsf_criterion_cd = FocalLoss(alpha = alpha1, gamma = args.gamma, dataset = 'ucsf')
ucsf_criterion_hiv = FocalLoss(alpha = alpha2, gamma = args.gamma, dataset = 'ucsf')

#set a learning rate for tau
tau_learning_rate = 0.01

#paired loss helper functions

#extract ids and cdr scores

adnimerge = pd.read_csv('/home/users/jmanasse/ADNIMERGE.csv')
adnimerge = adnimerge.dropna(subset=['CDRSB'])
cdr_ids = {}
for i,j in zip(adnimerge['PTID'], adnimerge['CDRSB']):
    cdr_ids[i] = j

cdrs = pd.read_excel('/home/users/jmanasse/2020.05.13 Demographic and CDR data.xlsx', engine = 'openpyxl', sheet_name = 'CDR')
cdrs = cdrs.dropna(subset=['CDR Sum'])
for i, j in zip(cdrs['PIDN'], cdrs['CDR Sum']):
    if j > 0:
        cdr_ids[i] = j


def pair_maker(cdr_ids, batch_data):#, features):
    #unpack values
    images, labels, actual_labels, datasets, ids, ages, genders = batch_data
    has_cdrs = []
    no_cdrs = []
    for i, id in enumerate(ids):
        if id in cdr_ids:
            has_cdrs.append(i)
            #has_cdrs.append((features[i], images[i], labels[i], actual_labels[i], datasets[i], ids[i], ages[i], genders[i]))
            #has_cdrs.append(features[i])
        else:
            no_cdrs.append(i)
            #no_cdrs.append((features[i],images[i], labels[i], actual_labels[i], datasets[i], ids[i], ages[i], genders[i]))
            #no_cdrs.append(features[i])
    np.random.shuffle(has_cdrs)
    np.random.shuffle(no_cdrs)
    has_pairs = [has_cdrs[i:i + 2] for i in range(0, len(has_cdrs), 2)]
    no_pairs = [no_cdrs[i:i + 2] for i in range(0, len(no_cdrs), 2)]
    if len(has_pairs[-1]) == 1:
        has_pairs = has_pairs[:-1]
    if len(no_pairs[-1]) == 1:
        no_pairs = no_pairs[:-1]
    return has_pairs, no_pairs

#tau = np.array([random.rand() for i in range(len(features))])

def pair_loss(cdr_pairs, features, ids, tau, scale_down=100):
    loss_ = 0
    T = tau.detach().numpy()
    for i, pair in enumerate(cdr_pairs):

        #LENGTH of projection of paired vector onto tau
        #delta_i = np.transpose(feature_extractor(pair[0][0]))*tau - np.transpose(feature_extractor(pair[1][0]))*tau
        T_norm = np.sqrt(sum(T**2))
        #print(pair[0][0])
        f1 = features[pair[0]].detach().cpu()
        f2 = features[pair[1]].detach().cpu()
        # f1 = pair[0][0].detach().cpu()
        # f2 =  pair[1][0].detach().cpu()

        delta_i = np.linalg.norm((np.dot(f1-f2,T)/T_norm**2)*T)


        id1 = ids[pair[0]]
        id2 = ids[pair[1]]

        #actual difference in cdr scores
        gamma_i = cdr_ids[id1] - cdr_ids[id2]

        loss_ += (delta_i - gamma_i)**2
    loss_ = torch.tensor(loss_, requires_grad=True)

    #scale down loss to 'forgive' poor initial tau?
    loss_ = loss_/torch.tensor(scale_down)
    return loss_

def cosine_loss(cdr_pairs, features, tau):
    #Compute the angular distance between the
    loss_ = 0
    T = tau.detach().numpy()
    for i, pair in enumerate(cdr_pairs):

        #paired vector in feature space
        f1 = features[pair[0]].detach().cpu()
        f2 = features[pair[1]].detach().cpu()
        # delta_i = np.transpose(feature_extractor(pair[0][0])) - np.transpose(feature_extractor(pair[1][0]))
        delta_i = np.transpose(f1) - np.transpose(f2)



        loss_ += np.dot(delta_i, T)/(np.linalg.norm(delta_i)*np.linalg.norm(T))

    #get angular distance from cosine similarity
    loss_ = torch.arccos(torch.tensor(loss_, requires_grad=True))/torch.tensor(math.pi)
    return loss_

# def tau_error(cdr_pairs, tau):
#     error = 0
#     T = tau.detach().numpy()
#     for pair in cdr_pairs:
#         delta_i = np.transpose(pair[0][0].detach().cpu()) - np.transpose(pair[1][0].detach().cpu())
#         error += T - delta_i
#     return error


@torch.no_grad()
def test(feature_extractor, classifier_ucsf, loader,  ucsf_criterion_cd, ucsf_criterion_hiv, tau, fold=None, epoch=None, train = False):
    feature_extractor.eval()
    classifier_ucsf.eval()

    xentropy_loss_avg = 0.
    paired_loss_avg = 0.
    cos_loss_avg = 0.
    correct = 0.
    total = 0.
    toprint = 0

    overall_accuracy = 0.
    correlation_ctrl = torch.tensor(0.)
    correlation_hiv = torch.tensor(0.)


    accuracy_class = {}
    accuracy_class['ucsf'] = {}
    accuracy_class['lab'] = {}
    accuracy_class['adni'] = {}
    accuracy_class['ucsf']['CTRL']=0
    accuracy_class['ucsf']['MCI']=0
    accuracy_class['ucsf']['HIV']=0
    accuracy_class['ucsf']['MND']=0
    accuracy_class['lab']['CTRL']=0
    accuracy_class['lab']['HIV']=0
    accuracy_class['adni']['CTRL']=0
    accuracy_class['adni']['MCI']=0

    accuracy_dataset = {}
    accuracy_dataset['lab']=0
    accuracy_dataset['ucsf']=0
    accuracy_dataset['adni']=0


    total_0_ucsf = 0.0
    total_1_ucsf = 0.0
    total_2_ucsf = 0.0
    total_3_ucsf = 0.0
    total_0_lab = 0.0
    total_2_lab = 0.0
    total_0_adni = 0.0
    total_1_adni = 0.0

    total_ucsf = 0
    total_lab = 0
    total_adni = 0

    feature_list = []
    all_datasets = []
    all_ids = []

    all_preds = []
    all_genders = []
    all_ages = []
    all_label_cd = []
    all_label_hiv = []

    num_batches = 0
    for i, _ in enumerate(loader):
        num_batches += 1

    for i,(images, labels,actual_labels,datasets,ids, ages, genders) in enumerate(loader):
        datasets = np.array(datasets)
        ids = np.array(ids)
        actual_labels = np.array(actual_labels)
        images = images.to(device).float()
        labels = labels.to(device).float()

        if i == num_batches - 1:
            number_needed = int(args.batch_size) - len(images)
            images0, labels0, actual_labels0, datasets0, ids0, ages0, genders0 = first_batch_data
            images = torch.cat((images, images0[:number_needed,]),dim=0)
            labels = torch.cat((labels, labels0[:number_needed,]),dim=0)
            actual_labels = np.concatenate((actual_labels, actual_labels0[:number_needed,]),axis=0)
            datasets = np.concatenate((datasets, datasets0[:number_needed]),axis=0)

            ids = np.concatenate((ids, ids0[:number_needed]),axis=0)
            ages = np.concatenate((ages, ages0[:number_needed]),axis=0)
            genders = np.concatenate((genders, genders0[:number_needed]),axis=0)

        data = (images, labels, actual_labels, datasets, ids, ages, genders)

        cfs = get_cf_kernel_batch(data)
        classifier_ucsf[2].cfs = cfs
        classifier_ucsf[5].cfs = cfs
        classifier_ucsf[7].cfs = cfs

        if i==0:
            first_batch_data = copy.deepcopy(data)

        feature = feature_extractor(images)
        pred = classifier_ucsf(feature)
        #print(pred)
        pred_cd = pred[:,0]
        pred_cd = torch.unsqueeze(pred_cd,1)
        pred_hiv = pred[:,1]
        pred_hiv = torch.unsqueeze(pred_hiv,1)

        # BELOW (TO REST OF FUNC) IS just metrics essentially
        labels_cd = labels[:,0]
        labels_cd = torch.unsqueeze(labels_cd,1)
        labels_hiv = labels[:,1]
        labels_hiv = torch.unsqueeze(labels_hiv,1)

        cdr_pairs, no_pairs = pair_maker(cdr_ids, data)

            #for pair in cdr_pairs:
            #get predictions and labels of each position in both pair list

        firsts = torch.tensor([row[0] for row in cdr_pairs] + [row[0] for row in no_pairs], dtype=torch.int32).to(device)
        seconds = torch.tensor([row[1] for row in cdr_pairs] + [row[1] for row in no_pairs], dtype=torch.int32).to(device)

        firstpred_cd1 = torch.index_select(pred_cd, 0, firsts)
        secondpred_cd1 = torch.index_select(pred_cd, 0, seconds)

        firstlabel_cd = torch.index_select(labels_cd, 0, firsts)
        secondlabel_cd = torch.index_select(labels_cd, 0, seconds)

        firstpred_hiv1 = torch.index_select(pred_hiv, 0, firsts)
        secondpred_hiv1=  torch.index_select(pred_hiv, 0, seconds)

        firstlabel_hiv =  torch.index_select(labels_hiv, 0, firsts)
        secondlabel_hiv = torch.index_select(labels_hiv, 0, seconds)

        xentropy_loss1 = ucsf_criterion_cd(firstpred_cd1, firstlabel_cd).to(device) + ucsf_criterion_hiv(firstpred_hiv1, firstlabel_hiv).to(device)
        xentropy_loss2 = ucsf_criterion_cd(secondpred_cd1, secondlabel_cd).to(device) + ucsf_criterion_hiv(secondpred_hiv1, secondlabel_hiv).to(device)

        paired_loss = pair_loss(cdr_pairs, feature, ids, tau).to(device) #+ #cos_loss(feature_extractor, cdr_pairs, tau)
        cos_loss =  cosine_loss(cdr_pairs, feature, tau).to(device)
        xentropy_loss = xentropy_loss1 + xentropy_loss2
        total_loss = xentropy_loss + paired_loss + cos_loss

        xentropy_loss_avg += xentropy_loss.item()
        paired_loss_avg += paired_loss.item()
        cos_loss_avg += cos_loss.item()
        pred_cur = copy.deepcopy(pred)
        pred_cd_copy = copy.deepcopy(pred_cd)
        pred_hiv_copy = copy.deepcopy(pred_hiv)

        pred_cd[pred_cd>0]=1
        pred_cd[pred_cd<0]=0
        pred_hiv[pred_hiv>0]=1
        pred_hiv[pred_hiv<0]=0
        # cd
        a=pred_cd == labels_cd
        # hiv
        b=pred_hiv == labels_hiv
        truth = torch.tensor([True]*len(a)).cuda()
        truth = torch.unsqueeze(truth,1)
        correct += ((a==truth)&(b==truth)).sum().item()
        total += images.size(0)

        # remove duplicate test data
        if i == num_batches - 1:
            number_actual = int(args.batch_size) - number_needed
            pred_cur = pred_cur[:number_actual,]
            pred_cd = pred_cd[:number_actual,]
            pred_hiv = pred_hiv[:number_actual,]
            labels_cd = labels_cd[:number_actual,]
            labels_hiv = labels_hiv[:number_actual,]
            datasets = datasets[:number_actual,]
            actual_labels = actual_labels[:number_actual,]
            ids= ids[:number_actual,]
            ages = ages[:number_actual,]
            genders = genders[:number_actual,]

            pred_cd_copy = pred_cd_copy[:number_actual,]
            pred_hiv_copy = pred_hiv_copy[:number_actual,]
            feature = feature[:number_actual,]


        feature_list.append(feature.cpu())
        all_datasets = np.append(all_datasets,datasets)
        all_ids = np.append(all_ids, ids)
        all_preds.extend(pred_cur.detach().cpu().numpy())
        all_genders.extend(genders)
        all_ages.extend(ages)
        all_label_cd.extend(labels_cd.squeeze().detach().cpu().numpy())
        all_label_hiv.extend(labels_hiv.squeeze().detach().cpu().numpy())

        # ucsf
        ucsf_pred_cd = pred_cd[datasets=='ucsf']
        ucsf_pred_hiv = pred_hiv[datasets=='ucsf']
        ucsf_pred_cd_copy = pred_cd_copy[datasets=='ucsf']
        ucsf_pred_hiv_copy = pred_hiv_copy[datasets=='ucsf']
        ucsf_labels_cd = labels_cd[datasets=='ucsf']
        ucsf_labels_hiv = labels_hiv[datasets=='ucsf']
        ucsf_actual_labels = actual_labels[datasets=='ucsf']
        ucsf_ids = ids[datasets=='ucsf']
        for j in range(0,len(ucsf_pred_cd)):
            total_ucsf += 1
            if train == False:

                row = {'epoch':epoch, 'id':ucsf_ids[j], 'dataset':'UCSF', 'CD_pred':torch.sigmoid(ucsf_pred_cd_copy[j]).item(), 'HIV_pred':torch.sigmoid(ucsf_pred_hiv_copy[j]).item(), 'fold': fold,'CD_label':ucsf_labels_cd[j].item(), 'HIV_label':ucsf_labels_hiv[j].item()}
                csv_logger_pred.writerow(row)

            if ucsf_pred_cd[j] == 0 and ucsf_pred_hiv[j] == 0 :
                actual_pred = 0
            elif ucsf_pred_cd[j] == 1 and ucsf_pred_hiv[j] == 0 :
                actual_pred = 1
            elif ucsf_pred_cd[j] == 0 and ucsf_pred_hiv[j] == 1 :
                actual_pred = 2
            elif ucsf_pred_cd[j] == 1 and ucsf_pred_hiv[j] == 1 :
                actual_pred = 3

            if ucsf_actual_labels[j] ==  0 :
                total_0_ucsf += 1
                if actual_pred == 0   :
                    accuracy_class['ucsf']['CTRL'] += 1
                    accuracy_dataset['ucsf'] += 1
            elif ucsf_actual_labels[j] ==  1 :
                total_1_ucsf += 1
                if actual_pred == 1  :
                    accuracy_class['ucsf']['MCI'] += 1
                    accuracy_dataset['ucsf'] += 1
            elif ucsf_actual_labels[j] ==  2 :
                total_2_ucsf += 1
                if actual_pred == 2   :
                    accuracy_class['ucsf']['HIV'] += 1
                    accuracy_dataset['ucsf'] += 1
            elif ucsf_actual_labels[j] ==  3 :
                total_3_ucsf += 1
                if actual_pred == 3 :
                    accuracy_class['ucsf']['MND'] += 1
                    accuracy_dataset['ucsf'] += 1

        # lab
        lab_pred_cd = pred_cd[datasets=='lab']
        lab_pred_hiv = pred_hiv[datasets=='lab']
        lab_pred_cd_copy = pred_cd_copy[datasets=='lab']
        lab_pred_hiv_copy = pred_hiv_copy[datasets=='lab']
        lab_labels_cd = labels_cd[datasets=='lab']
        lab_labels_hiv = labels_hiv[datasets=='lab']
        lab_actual_labels = actual_labels[datasets=='lab']
        lab_ids = ids[datasets=='lab']
        for j in range(0,len(lab_pred_cd)):
            total_lab += 1
            if train == False:
                row = {'epoch':epoch, 'id':lab_ids[j], 'dataset':'LAB', 'CD_pred':torch.sigmoid(lab_pred_cd_copy[j]).item(), 'HIV_pred':torch.sigmoid(lab_pred_hiv_copy[j]).item(), 'fold': fold,'CD_label':lab_labels_cd[j].item(), 'HIV_label':lab_labels_hiv[j].item()}
                csv_logger_pred.writerow(row)
            if lab_pred_cd[j] == 0 and lab_pred_hiv[j] == 0 :
                actual_pred = 0
            elif lab_pred_cd[j] == 1 and lab_pred_hiv[j] == 0 :
                actual_pred = 1
            elif lab_pred_cd[j] == 0 and lab_pred_hiv[j] == 1 :
                actual_pred = 2
            elif lab_pred_cd[j] == 1 and lab_pred_hiv[j] == 1 :
                actual_pred = 3

            if lab_actual_labels[j] ==  0 :
                total_0_lab += 1
                if actual_pred == 0   :
                    accuracy_class['lab']['CTRL'] += 1
                    accuracy_dataset['lab'] += 1
            elif lab_actual_labels[j] ==  2 :
                total_2_lab += 1
                if actual_pred == 2   :
                    accuracy_class['lab']['HIV'] += 1
                    accuracy_dataset['lab'] += 1
            else:
                print('LAB  actual_labels[j]:',actual_labels[j])

        # adni
        adni_pred_cd = pred_cd[datasets=='adni']
        adni_pred_hiv = pred_hiv[datasets=='adni']
        adni_pred_cd_copy = pred_cd_copy[datasets=='adni']
        adni_pred_hiv_copy = pred_hiv_copy[datasets=='adni']
        adni_labels_cd = labels_cd[datasets=='adni']
        adni_labels_hiv = labels_hiv[datasets=='adni']
        adni_actual_labels = actual_labels[datasets=='adni']
        adni_ids = ids[datasets=='adni']
        for j in range(0,len(adni_pred_cd)):
            total_adni += 1
            if train == False:
                row = {'epoch':epoch, 'id':adni_ids[j], 'dataset':'ADNI', 'CD_pred':torch.sigmoid(adni_pred_cd_copy[j]).item(), 'HIV_pred':torch.sigmoid(adni_pred_hiv_copy[j]).item(), 'fold': fold,'CD_label':adni_labels_cd[j].item(), 'HIV_label':adni_labels_hiv[j].item()}
                csv_logger_pred.writerow(row)

            if adni_pred_cd[j] == 0 and adni_pred_hiv[j] == 0 :
                actual_pred = 0
            elif adni_pred_cd[j] == 1 and adni_pred_hiv[j] == 0 :
                actual_pred = 1
            elif adni_pred_cd[j] == 0 and adni_pred_hiv[j] == 1 :
                actual_pred = 2
            elif adni_pred_cd[j] == 1 and adni_pred_hiv[j] == 1 :
                actual_pred = 3

            if adni_actual_labels[j] ==  0 :
                total_0_adni += 1
                if actual_pred == 0   :
                    accuracy_class['adni']['CTRL'] += 1
                    accuracy_dataset['adni'] += 1
            elif adni_actual_labels[j] ==  1 :
                total_1_adni += 1
                if actual_pred == 1  :
                    accuracy_class['adni']['MCI'] += 1
                    accuracy_dataset['adni'] += 1

            else:
                print('ADNI  actual_labels[j]:',actual_labels[j])

    # calculate correlations
    all_feature = np.concatenate(feature_list, axis=0)
    all_datasets_onehot = np.zeros([len(all_datasets),3])
    all_datasets_onehot[all_datasets=='ucsf'] = [1,0,0]
    all_datasets_onehot[all_datasets=='lab'] = [0,1,0]
    all_datasets_onehot[all_datasets=='adni'] = [0,0,1]
    distance = dcor.distance_correlation_sqr(all_feature, all_datasets_onehot)
    correlation0_gender, correlation1_gender, correlation0_age, correlation1_age, distance_age, distance_gender = calculate_gender_age_correlation(all_preds, all_genders, all_ages, all_label_cd, all_label_hiv, all_feature)
    print("corr0_g:", correlation0_gender, "corr1_g:", correlation1_gender, "corr0_a:", correlation0_age, "corr1_a", correlation1_age)
    row = {'epoch':epoch, 'train':train, 'fold':fold, 'final_corr0_age':correlation0_age, 'final_corr1_age':correlation1_age, 'final_corr0_gender':correlation0_gender, 'final_corr1_gender':correlation1_gender, 'intermediate_age':distance_age, 'intermediate_gender':distance_gender}
    csv_logger_corr.writerow(row)


    accuracy_class['ucsf']['CTRL'] = round(accuracy_class['ucsf']['CTRL'] / total_0_ucsf,3)
    accuracy_class['ucsf']['MCI'] = round(accuracy_class['ucsf']['MCI'] / total_1_ucsf,3)
    accuracy_class['ucsf']['HIV'] = round(accuracy_class['ucsf']['HIV'] / total_2_ucsf,3)
    accuracy_class['ucsf']['MND']= round(accuracy_class['ucsf']['MND'] / total_3_ucsf,3)
    accuracy_class['lab']['CTRL'] = round(accuracy_class['lab']['CTRL'] / total_0_lab,3)
    accuracy_class['lab']['HIV'] = round(accuracy_class['lab']['HIV'] / total_2_lab,3)
    accuracy_class['adni']['CTRL'] = round(accuracy_class['adni']['CTRL'] / total_0_adni,3)
    accuracy_class['adni']['MCI'] = round(accuracy_class['adni']['MCI'] / total_1_adni,3)


    accuracy_dataset['ucsf'] = round(accuracy_dataset['ucsf'] / total_ucsf,3)
    accuracy_dataset['lab'] = round(accuracy_dataset['lab'] / total_lab,3)
    accuracy_dataset['adni'] = round(accuracy_dataset['adni'] / total_adni,3)

    # for k, key in enumerate(accuracy_class):
    #     print(accuracy_class[key], [total_0_ucsf, total_1_ucsf, total_2_ucsf, total_3_ucsf, total_0_lab, total_2_lab, total_0_adni, total_1_adni][k])
    print(accuracy_class, total_ucsf, total_adni, total_lab)
    overall_accuracy = (correct) / (total)
    overall_accuracy = round(overall_accuracy,3)


    xentropy_loss_avg = xentropy_loss_avg / (i + 1)
    paired_loss_avg = paired_loss_avg / (i + 1)
    cos_loss_avg = cos_loss_avg / (i + 1)

    return overall_accuracy, xentropy_loss_avg, paired_loss_avg, cos_loss_avg, accuracy_class, accuracy_dataset, distance

def calculate_gender_age_correlation(predictions, genders, ages, cd_labels, hiv_labels, features):
    m_gender = []
    a_ages = []
    predictions0 = []
    predictions1 = []
    predictions0_array = np.array(predictions)[:,0]
    predictions1_array = np.array(predictions)[:,1]
    features_cur = []
    for i in range(len(genders)):
        if cd_labels[i] != 0 or hiv_labels[i] != 0:
            continue
        predictions0.append(predictions0_array[i])
        predictions1.append(predictions1_array[i])
        features_cur.append(features[i])
        a_ages.append(ages[i])
        if genders[i] == 0:
            m_gender.append(1)
        elif genders[i] == 1:
            m_gender.append(0)

    features_np = np.array(features_cur)
    mean0 = np.array(predictions0).mean()
    mean1 = np.array(predictions1).mean()
    meanM = np.array(m_gender).mean()
    meanA = np.array(a_ages).mean()

    numerator0_gender = np.sum((predictions0 - mean0) * (m_gender - meanM))
    denomenator0_gender = np.sqrt(np.sum((predictions0 - mean0)**2) * np.sum((m_gender - meanM)**2))
    correlation0_gender = numerator0_gender / denomenator0_gender

    numerator1_gender = np.sum((predictions1 - mean1) * (m_gender - meanM))
    denomenator1_gender = np.sqrt(np.sum((predictions1 - mean1)**2) * np.sum((m_gender - meanM)**2))
    correlation1_gender = numerator1_gender / denomenator1_gender

    numerator0_age = np.sum((predictions0 - mean0) * (a_ages - meanA))
    denomenator0_age = np.sqrt(np.sum((predictions0 - mean0)**2) * np.sum((a_ages - meanA)**2))
    correlation0_age = numerator0_age / denomenator0_age

    numerator1_age = np.sum((predictions1 - mean1) * (a_ages - meanA))
    denomenator1_age = np.sqrt(np.sum((predictions1 - mean1)**2) * np.sum((a_ages - meanA)**2))
    correlation1_age = numerator1_age / denomenator1_age

    distance_age = dcor.distance_correlation_sqr(features_np, a_ages)
    distance_gender = dcor.distance_correlation_sqr(features_np, m_gender)

    return correlation0_gender, correlation1_gender, correlation0_age, correlation1_age, distance_age, distance_gender


def get_cf_kernel(loader):
    label_hiv = []
    label_cd = []
    dataset_lab = []
    dataset_adni = []
    dataset_ucsf = []
    ages = []
    gender_m = []

    for i,(all_images, all_labels, all_actual_labels, all_datasets, all_ids, all_ages, all_genders) in enumerate(loader):
        for j in range(0,len(all_images)):
            labels=all_labels[j]
            actual_labels=all_actual_labels[j]
            datasets =  all_datasets[j]

            if actual_labels == 0:
                label_hiv.append(0)
                label_cd.append(0)
            elif actual_labels == 1: #cd
                label_hiv.append(0)
                label_cd.append(1)
            elif actual_labels == 2: #hiv
                label_hiv.append(1)
                label_cd.append(0)
            elif actual_labels == 3: #hand
                label_hiv.append(1)
                label_cd.append(1)

            if datasets=='ucsf':
                dataset_lab.append(0)
                dataset_adni.append(0)
                dataset_ucsf.append(1)
            elif datasets=='lab':
                dataset_lab.append(1)
                dataset_adni.append(0)
                dataset_ucsf.append(0)
            elif datasets=='adni':
                dataset_lab.append(0)
                dataset_adni.append(1)
                dataset_ucsf.append(0)

            ages.append(all_ages[j])
            cur_gender = all_genders[j]
            if cur_gender == 0:
                gender_m.append(1)
            elif cur_gender == 1:
                gender_m.append(0)

    N = len(dataset_lab)
    X_shuffled = np.zeros((N,7))
    X_shuffled[:,0] = label_hiv
    X_shuffled[:,1] = label_cd
    X_shuffled[:,2] = dataset_lab
    X_shuffled[:,3] = dataset_adni
#     X_shuffled[:,4] = dataset_ucsf
    X_shuffled[:,4] = np.ones((N,))
    X_shuffled[:,5] = ages
    X_shuffled[:,6] = gender_m

    cf_kernel = nn.Parameter(torch.tensor(np.linalg.inv(np.transpose(X_shuffled).dot(X_shuffled))).float().to(device),  requires_grad=False)

    return cf_kernel

def get_cf_kernel_batch(data):
    all_images, all_labels, all_actual_labels, all_datasets, all_ids, all_ages, all_genders = data
    label_hiv = []
    label_cd = []
    dataset_lab = []
    dataset_adni = []
    dataset_ucsf = []
    ages = []
    gender_m = []
    N = all_images.shape[0]
    for j in range(0,N):
        labels=all_labels[j]
        actual_labels=all_actual_labels[j]
        datasets =  all_datasets[j]

        if actual_labels == 0:
            label_hiv.append(0)
            label_cd.append(0)
        elif actual_labels == 1: #cd
            label_hiv.append(0)
            label_cd.append(1)
        elif actual_labels == 2: #hiv
            label_hiv.append(1)
            label_cd.append(0)
        elif actual_labels == 3: #hand
            label_hiv.append(1)
            label_cd.append(1)

        if datasets=='ucsf':
            dataset_lab.append(0)
            dataset_adni.append(0)
            dataset_ucsf.append(1)
        elif datasets=='lab':
            dataset_lab.append(1)
            dataset_adni.append(0)
            dataset_ucsf.append(0)
        elif datasets=='adni':
            dataset_lab.append(0)
            dataset_adni.append(1)
            dataset_ucsf.append(0)

        ages.append(all_ages[j])
        cur_gender = all_genders[j]
        if cur_gender == 0:
            gender_m.append(1)
        elif cur_gender == 1:
            gender_m.append(0)

    cfs_batch = np.zeros((N,7))
    cfs_batch[:,0] = label_hiv
    cfs_batch[:,1] = label_cd
    cfs_batch[:,2] = dataset_lab
    cfs_batch[:,3] = dataset_adni
#     cfs_batch[:,4] = dataset_ucsf
    cfs_batch[:,4] = np.ones((N,))
    cfs_batch[:,5] = ages
    cfs_batch[:,6] = gender_m

    cfs = nn.Parameter(torch.Tensor(cfs_batch).to(device).float(), requires_grad=False)


    return cfs

def train(feature_extractor,  classifier_ucsf, train_loader, test_loader,final_test_loader, cf_kernel, fold ):

    feature_extractor.zero_grad()
    classifier_ucsf.zero_grad()

    tau = nn.Parameter(data=torch.from_numpy(np.random.rand(2048,1)))

    fe_optimizer = optim.AdamW(feature_extractor.parameters(), lr =args.lr, weight_decay=0.01) # used to be 0.01
    ucsf_optimizer = optim.AdamW(classifier_ucsf.parameters(), lr =args.lr, weight_decay=args.wd) # used to be args.wd
    tau_optimizer = optim.AdamW([tau], lr =args.lr, weight_decay=0.01)

    best_accuracy = 0
    best_epoch = 0
    best_tau = 0
    epochs = args.epochs
    ave_valid_acc_50 = 0.0
    counter = 0.0

    alpha1 = args.alpha[0]
    alpha2 = args.alpha[1]
    ucsf_criterion_cd = FocalLoss(alpha = alpha1, gamma = args.gamma, dataset = 'ucsf')
    ucsf_criterion_hiv = FocalLoss(alpha = alpha2, gamma = args.gamma, dataset = 'ucsf')

    for epoch in range(epochs):
        feature_extractor.train()
        classifier_ucsf.train()

        progress_total = 0
        num_samples = []
        for i, batch in enumerate(train_loader):
            progress_total += 1
            num_samples.append(len(batch[0]))

        progress_bar = tqdm(train_loader, total = progress_total)
        xentropy_loss_avg = 0.
        paired_loss_avg = 0.
        cos_loss_avg = 0.
        cur_loss_sum = 0
        correct = 0.
        total = 0.
        ucsf_correct = 0.
        ucsf_total = 0.
        lab_correct = 0.
        lab_total = 0.
        adni_correct = 0.
        adni_total = 0.
        total = 0.
        overall_accuracy = 0

        ###### "Training happens here! ######
        for i, (images, labels, actual_labels, datasets, ids, ages, genders) in enumerate(progress_bar):
            #print(act)
            feature_extractor.zero_grad()
            classifier_ucsf.zero_grad()

            data = (images, labels, actual_labels, datasets, ids, ages, genders)
            #print(data[1:])
            cfs = get_cf_kernel_batch(data)

            classifier_ucsf[2].cfs = cfs
            classifier_ucsf[5].cfs = cfs
            classifier_ucsf[7].cfs = cfs

            datasets = np.array(datasets)

            progress_bar.set_description('Epoch ' + str(epoch))
            images = images.to(device).float()
            labels = labels.to(device).float()

            feature = feature_extractor(images)
            pred = classifier_ucsf(feature)

            pred_cd1 = pred[:,0].unsqueeze(1)
            pred_hiv1 = pred[:,1].unsqueeze(1)
            labels_cd = labels[:,0].unsqueeze(1)
            labels_hiv = labels[:,1].unsqueeze(1)

            cdr_pairs, no_pairs = pair_maker(cdr_ids, data)


        #    print(cdr_pairs)
            #for pair in cdr_pairs:
            #get predictions and labels of each position in both pair list
            firsts = torch.tensor([row[0] for row in cdr_pairs] + [row[0] for row in no_pairs], dtype=torch.int32).to(device)
            seconds = torch.tensor([row[1] for row in cdr_pairs] + [row[1] for row in no_pairs], dtype=torch.int32).to(device)

            firstpred_cd1 = torch.index_select(pred_cd1, 0, firsts)
            secondpred_cd1 = torch.index_select(pred_cd1, 0, seconds)

            firstlabel_cd = torch.index_select(labels_cd, 0, firsts)
            secondlabel_cd = torch.index_select(labels_cd, 0, seconds)

            firstpred_hiv1 = torch.index_select(pred_hiv1, 0, firsts)
            secondpred_hiv1=  torch.index_select(pred_hiv1, 0, seconds)

            firstlabel_hiv =  torch.index_select(labels_hiv, 0, firsts)
            secondlabel_hiv = torch.index_select(labels_hiv, 0, seconds)

            xentropy_loss1 = ucsf_criterion_cd(firstpred_cd1, firstlabel_cd).to(device) + ucsf_criterion_hiv(firstpred_hiv1, firstlabel_hiv).to(device)
            xentropy_loss2 = ucsf_criterion_cd(secondpred_cd1, secondlabel_cd).to(device) + ucsf_criterion_hiv(secondpred_hiv1, secondlabel_hiv).to(device)

            paired_loss = pair_loss(cdr_pairs, feature, ids, tau).to(device) #+ #cos_loss(feature_extractor, cdr_pairs, tau)
            cos_loss =  cosine_loss(cdr_pairs, feature, tau).to(device)
            xentropy_loss = xentropy_loss1 + xentropy_loss2
            total_loss = xentropy_loss + paired_loss + cos_loss




            #print(total_loss)

            #update tau
            #adam optimizer, no backprop
            #from adam paper - alpha = 0.001, beta1 = 0.9, beta = 0.999 and epsilon = 10^-8


            total_loss.backward()


            #xentropy_loss.backward()

            fe_optimizer.step()
            ucsf_optimizer.step()
            tau_optimizer.step()
            ###### End of "training" is here! ######

            pred_cd = pred_cd1.clone()
            pred_cd[pred_cd>0]=1
            pred_cd[pred_cd<0]=0

            pred_hiv = pred_hiv1.clone()
            pred_hiv[pred_hiv>0]=1
            pred_hiv[pred_hiv<0]=0
            # cd
            a=pred_cd == labels_cd
            # hiv
            b=pred_hiv == labels_hiv
            truth = torch.tensor([True]*len(a)).cuda()
            truth = torch.unsqueeze(truth,1)
            correct += ((a==truth)&(b==truth)).sum().item()
            total += images.size(0)
            overall_accuracy= correct/total

            xentropy_loss_avg += xentropy_loss1.item()
            paired_loss_avg += paired_loss.item()
            cos_loss_avg += cos_loss.item()
            #print(correct, total)
            progress_bar.set_postfix(
                xloss='%.6f' % (xentropy_loss_avg / (i + 1)),
                ploss='%.6f' % (paired_loss_avg / (i + 1)),
                closs='%.6f' % (cos_loss_avg / (i + 1)),
                acc='%.2f' % overall_accuracy)

        test_acc, test_xloss, test_ploss, test_closs, test_accuracy_class, test_accuracy_dataset, test_distance = test(feature_extractor, classifier_ucsf, test_loader, ucsf_criterion_cd, ucsf_criterion_hiv, tau, fold =fold, epoch = epoch)

        test_ucsf_ctrl = test_accuracy_class['ucsf']['CTRL']
        test_ucsf_mci = test_accuracy_class['ucsf']['MCI']
        test_ucsf_hiv = test_accuracy_class['ucsf']['HIV']
        test_ucsf_mnd = test_accuracy_class['ucsf']['MND']
        test_lab_ctrl = test_accuracy_class['lab']['CTRL']
        test_lab_hiv = test_accuracy_class['lab']['HIV']
        test_adni_ctrl = test_accuracy_class['adni']['CTRL']
        test_adni_mci = test_accuracy_class['adni']['MCI']

        ucsf_test_acc = np.mean([test_ucsf_ctrl, test_ucsf_mci, test_ucsf_hiv, test_ucsf_mnd])
        lab_test_acc = np.mean([test_lab_ctrl, test_lab_hiv])
        adni_test_acc = np.mean([test_adni_ctrl, test_adni_mci])
        test_accuracy_dataset['ucsf'] = round(ucsf_test_acc,3)
        test_accuracy_dataset['lab'] = round(lab_test_acc,3)
        test_accuracy_dataset['adni'] = round(adni_test_acc,3)
        print('test:',test_accuracy_class['ucsf'])

        # this trainning accuracy has augmentation in it!!!!!
        # some images are sampled more than once!!!!
        train_acc, train_xloss, train_ploss,train_closs, train_accuracy_class, train_accuracy_dataset, train_distance = test(feature_extractor, classifier_ucsf, train_loader, ucsf_criterion_cd, ucsf_criterion_hiv, tau, fold =fold, epoch = epoch, train =True)

        train_ucsf_ctrl = train_accuracy_class['ucsf']['CTRL']
        train_ucsf_mci = train_accuracy_class['ucsf']['MCI']
        train_ucsf_hiv = train_accuracy_class['ucsf']['HIV']
        train_ucsf_mnd = train_accuracy_class['ucsf']['MND']
        train_lab_ctrl = train_accuracy_class['lab']['CTRL']
        train_lab_hiv = train_accuracy_class['lab']['HIV']
        train_adni_ctrl = train_accuracy_class['adni']['CTRL']
        train_adni_mci = train_accuracy_class['adni']['MCI']

        # redefine ucsf_train_acc, lab_val_acc to be the average of all classes
        ucsf_train_acc = np.mean([train_ucsf_ctrl, train_ucsf_mci, train_ucsf_hiv, train_ucsf_mnd])
        lab_train_acc = np.mean([train_lab_ctrl, train_lab_hiv])
        adni_train_acc = np.mean([train_adni_ctrl, train_adni_mci])
        train_accuracy_dataset['ucsf'] = round(ucsf_train_acc,3)
        train_accuracy_dataset['lab'] = round(lab_train_acc,3)
        train_accuracy_dataset['adni'] = round(adni_train_acc,3)

        tqdm.write('train_acc: %.2f u_train_acc: %.2f l_train_acc: %.2f a_train_acc: %.2f' % (overall_accuracy, ucsf_train_acc, lab_train_acc, adni_train_acc))
        tqdm.write('test_acc: %.2f u_test_acc: %.2f l_test_acc: %.2f a_test_acc: %.2f  test_xloss: %.2f test_ploss: %.2f test_closs: %.2f' % (test_acc, ucsf_test_acc, lab_test_acc, adni_test_acc, test_xloss, test_ploss, test_closs))
        row = {'epoch': epoch, 'train_acc': round(overall_accuracy,3), 'test_acc': test_acc, 'train_xloss':round((xentropy_loss_avg / (i + 1)),3), 'train_ploss':round((paired_loss_avg / (i + 1)),3), 'train_closs':round((cos_loss_avg / (i + 1)),3),
        'test_xloss': round(test_xloss,3), 'test_ploss': round(test_ploss,3),'test_closs': round(test_closs,3),
               'ucsf_train_acc': ucsf_train_acc,'lab_train_acc': lab_train_acc,'adni_train_acc': adni_train_acc,

               'ucsf_test_acc': ucsf_test_acc, 'lab_test_acc': lab_test_acc,'adni_test_acc': adni_test_acc,
               'correlation_ctrl_train':0,
               'correlation_hiv_train':0,

               'correlation_ctrl_test':0,
               'correlation_hiv_test':0,
               'train_ucsf_ctrl':train_ucsf_ctrl, 'train_ucsf_mci':train_ucsf_mci,
               'train_ucsf_hiv':train_ucsf_hiv, 'train_ucsf_mnd':train_ucsf_mnd,
               'train_lab_ctrl':train_lab_ctrl, 'train_lab_hiv':train_lab_hiv,
               'train_adni_ctrl':train_adni_ctrl, 'train_adni_mci':train_adni_mci,

               'test_ucsf_ctrl':test_ucsf_ctrl, 'test_ucsf_mci':test_ucsf_mci,
               'test_ucsf_hiv':test_ucsf_hiv, 'test_ucsf_mnd':test_ucsf_mnd,
               'test_lab_ctrl':test_lab_ctrl, 'test_lab_hiv':test_lab_hiv,
              'test_adni_ctrl':test_adni_ctrl, 'test_adni_mci':test_adni_mci,
              'train_distance':train_distance,'test_distance':test_distance}
        csv_logger.writerow(row)

        if test_acc > best_accuracy:
            best_accuracy = test_acc
            best_epoch = epoch
            best_tau = tau


    best_models = [feature_extractor, classifier_ucsf]

    return test_acc, test_accuracy_class, test_accuracy_dataset, best_accuracy, best_epoch, best_tau, best_models

def average_results(acc_each_class_list,acc_each_dataset_list):
    ave_acc_each_class_list = {}
    ave_acc_each_class_list['ucsf'] = {}
    ave_acc_each_class_list['lab'] = {}
    ave_acc_each_class_list['adni'] = {}
    for d in acc_each_class_list:
        for dataset in d.keys():
            for key in d[dataset].keys():
                if key not in ave_acc_each_class_list[dataset]:
                    ave_acc_each_class_list[dataset][key] = d[dataset][key]/5
                else:
                    ave_acc_each_class_list[dataset][key] += d[dataset][key]/5

    for dataset in ave_acc_each_class_list.keys():
            for key in ave_acc_each_class_list[dataset].keys():
                ave_acc_each_class_list[dataset][key] = round(ave_acc_each_class_list[dataset][key],3)

    ave_acc_each_dataset_list={}
    for d in acc_each_dataset_list:
        for key in d.keys():
            if key not in ave_acc_each_dataset_list:
                ave_acc_each_dataset_list[key] = d[key]/5
            else:
                ave_acc_each_dataset_list[key] += d[key]/5
    for key in ave_acc_each_dataset_list.keys():
        ave_acc_each_dataset_list[key] = round(ave_acc_each_dataset_list[key],3)

    return ave_acc_each_class_list, ave_acc_each_dataset_list

if __name__ == '__main__':
    log_path = '/scratch/users/jmanasse/mri_proj/logs/'
    filename = log_path + args.name +'.csv'
    os.makedirs(log_path, exist_ok=True)
    csv_logger = CSVLogger( args, fieldnames=['epoch', 'train_acc',  'test_acc',
                                              'train_xloss','train_ploss','train_closs','test_xloss','test_ploss', 'test_closs',
                                              'ucsf_train_acc','lab_train_acc','adni_train_acc',

                                              'ucsf_test_acc','lab_test_acc', 'adni_test_acc',
                                              'correlation_ctrl_train', 'correlation_hiv_train',

                                              'correlation_ctrl_test', 'correlation_hiv_test',
                                              'train_ucsf_ctrl','train_ucsf_mci', 'train_ucsf_hiv','train_ucsf_mnd',
                                              'train_lab_ctrl', 'train_lab_hiv', 'train_adni_ctrl', 'train_adni_mci',

                                              'test_ucsf_ctrl','test_ucsf_mci','test_ucsf_hiv','test_ucsf_mnd',
                                              'test_lab_ctrl', 'test_lab_hiv','test_adni_ctrl', 'test_adni_mci',
                                             'train_distance','test_distance'],
                           filename=filename)

    filename2 = log_path + 'predictions/'+ args.name +'.csv'
    os.makedirs(log_path + 'predictions/', exist_ok=True)
    csv_logger_pred = CSVLogger( args, fieldnames=['epoch', 'id', 'dataset', 'CD_pred', 'HIV_pred', 'fold','CD_label', 'HIV_label'], filename=filename2)
    filename3 = log_path + 'predictions/' + args.name + 'corrs.csv'
    csv_logger_corr = CSVLogger( args, fieldnames=['epoch', 'train', 'fold', 'final_corr0_age', 'final_corr1_age', 'final_corr0_gender', 'final_corr1_gender', 'intermediate_age', 'intermediate_gender'], filename=filename3)

    ## cross-validation
    best_accuracy_list = [0,0,0,0,0]
    best_epoch_list = [0,0,0,0,0]
    final_accuracy_list = [0,0,0,0,0]
    ave_valid_acc_50_list=[0,0,0,0,0]
    best_model_dict = {}

    acc_each_class_list = []
    acc_each_dataset_list = []
    for fold in range(0,5):

        row = {'epoch': 'fold', 'train_acc': str(fold)}
        csv_logger.writerow(row)
        transformation = super_transformation()
        train_data = MRI_Dataset(fold = fold , stage= 'original_train',transform = transformation)
        test_data = MRI_Dataset(fold = fold , stage= 'original_test')

        train_loader = DataLoader(dataset=train_data,
                                  batch_size=args.batch_size,
                                  sampler=MixedSampler(dataset=train_data,
                                                       batch_size=args.batch_size),
                                  shuffle=False,
                                  pin_memory=True,
                                  num_workers=3)
        test_loader = DataLoader(dataset=test_data ,
                                  batch_size=args.batch_size,# to include all test images
                                  shuffle=True,
                                  pin_memory=True,
                                  num_workers=3)
        final_test_loader = DataLoader(dataset=test_data ,
                          batch_size=1,#args.batch_size,
                          shuffle=False,
                          pin_memory=True,
                          num_workers=3)
        print("Begin training fold ",fold)

        cf_kernel  = get_cf_kernel(train_loader)
        feature_extractor = fe(cf_kernel, trainset_size = len(train_data), in_num_ch=1, img_size=(64, 64, 64), inter_num_ch=16,
                           fc_num_ch=16, kernel_size=3, conv_act='relu',
                           fe_arch=args.fe_arch, dropout=args.dropout,
                           fc_dropout = args.fc_dropout, batch_size = args.batch_size).to(device)

        classifier_ucsf = nn.Sequential(
#                     MetadataNorm(batch_size=args.batch_size, cf_kernel=cf_kernel, num_features = 2048, trainset_size = len(train_data)),
                    nn.Linear(2048, 128),
                    nn.ReLU(),
#                     nn.BatchNorm1d(128),
                    MetadataNorm(batch_size=args.batch_size, cf_kernel=cf_kernel, num_features = 128, trainset_size = len(train_data)),
                    nn.Linear(128, 16),
                    nn.ReLU(),
#                     nn.BatchNorm1d(16),
                   MetadataNorm(batch_size=args.batch_size, cf_kernel=cf_kernel, num_features = 16, trainset_size = len(train_data)),
                    nn.Linear(16, 2),
                    MetadataNorm(batch_size=args.batch_size, cf_kernel=cf_kernel, num_features = 2, trainset_size = len(train_data)),
                ).to(device)


        test_acc, test_accuracy_class, test_accuracy_dataset, best_accuracy, best_epoch, best_tau, best_models = train(feature_extractor,  classifier_ucsf, train_loader,  test_loader, final_test_loader, cf_kernel=cf_kernel, fold = fold)


        feature_extractor, classifier_ucsf = best_models
        best_accuracy_list[fold] = best_accuracy
        final_accuracy_list[fold] = test_acc
        best_epoch_list[fold] = best_epoch


        test_acc, test_xloss,test_ploss,test_closs,test_accuracy_class, test_accuracy_dataset, test_distance = test(feature_extractor, classifier_ucsf, test_loader, ucsf_criterion_cd, ucsf_criterion_hiv, best_tau,  fold = fold, train =True)
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
        torch.save(feature_extractor.state_dict(), new_dir + 'feature_extractor.pt')
        torch.save(classifier_ucsf.state_dict(), new_dir + 'classifier_ucsf.pt')




    print('best_accuracy', best_accuracy_list)
    print('final_accuracy',final_accuracy_list)
    print('best_epoch', best_epoch_list)
    print('ave_valid_acc_50',ave_valid_acc_50_list)
    print('acc_each_class',acc_each_class_list)
    print('acc_each_dataset',acc_each_dataset_list)

    ave_acc_each_class_list, ave_acc_each_dataset_list = average_results(acc_each_class_list,acc_each_dataset_list)
    print(ave_acc_each_class_list)
    print(ave_acc_each_dataset_list)

    # finish the loggers
    csv_logger.final(best_accuracy_list,final_accuracy_list,
                     best_epoch_list,ave_valid_acc_50_list,
                     acc_each_class_list,acc_each_dataset_list)
    csv_logger.close()
    csv_logger_pred.close()
    csv_logger_corr.close()
