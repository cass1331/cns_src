from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from sklearn.metrics import roc_curve

from sklearn.model_selection import StratifiedKFold
import numpy as np
import nibabel as nib
import scipy as sp
import scipy.ndimage
from sklearn.metrics import mean_squared_error, r2_score

import pandas as pd
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

from datasplitting import Paired_UCSF_Dataset
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

import wandb



#take lambda to zero and take model apart

#1. trains OK w.o. pair loop
#2 next! re-omplement pair loop w.o. breaking anything
#keep a row of hps vs results (like with baseline)
#hypothesis testing, latent space seperabilitys

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
parser.add_argument('--lamb', type=float, default=0.5,
                    help='multiplier for pair loss')
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

    args.name = args.name + '_dyn_drop_'+ '_fearch_' + args.fe_arch + '_bz_' + str(args.batch_size) +'_epoch_' + str(args.epochs) + '_lr_' + str(args.lr) + '_wd_' + str(args.wd) + '_lamb_' + str(args.lamb) + '_alpha_'+str(args.alpha)+ '_seed_'+str(seed)
else:
    args.name = args.name + '_fearch_' + args.fe_arch + '_bz_' + str(args.batch_size) + '_epoch_' + str(args.epochs) + '_lr_' + str(args.lr) + '_do_'+str(args.dropout) +'_fcdo_' + str(args.fc_dropout) + '_wd_' + str(args.wd) + '_lamb_' + str(args.lamb)  + '_alpha_'+str(args.alpha)+ '_seed_'+str(seed)

#UNCOMMENT TO LOG TRAINING SETTINGS

wandb.init(project="paired", entity="jmanasse", config = {
  "learning_rate": args.lr,
  "epochs": args.epochs,
  "batch_size": args.batch_size,
  "fc dropout": args.fc_dropout,
  'dropout': args.dropout,
  "weight decay": args.wd,
  "loss multiplier": args.lamb,
  'loss type': 'baseline'

})

print("device:",device)


#extract npz scores
ucsf_metadata =  pd.read_csv('/home/users/jmanasse/all_ucsf_metadata.csv')
ucsf_metadata = ucsf_metadata[ucsf_metadata['NPZ'].notna()]
#remove 99s (non-entries)

ucsf_metadata = ucsf_metadata[ucsf_metadata['NPZ'] != 99]

cdr_ids = ucsf_metadata['PIDN']

#make dictionary to look up -- patient id: NPZ score
npz = {}
for i, j in zip(ucsf_metadata['PIDN'], ucsf_metadata['NPZ']):
    npz[i] = j

def focal_loss(inputs, targets, alpha=0.1, gamma=2):
    bce = nn.BCEWithLogitsLoss(reduction= 'none')
    BCE_loss = bce(inputs, targets)
    if inputs.shape[0] == 1:
        return BCE_loss

    pt = torch.exp(-BCE_loss)
    F_loss= 0


    F_loss_pos = alpha * (1-pt[targets==1])**gamma * BCE_loss[targets==1]
    F_loss_neg = (1-alpha) * (1-pt[targets==0])**gamma * BCE_loss[targets==0]

    if inputs.shape[0] == 1:
        if F_loss_pos.nelement() > 0:
            return F_loss_pos
        else:
            return F_loss_pos
    # F_loss += (torch.mean(F_loss_pos)+torch.mean(F_loss_neg))/2
    F_loss += (torch.mean(torch.cat((F_loss_pos, F_loss_neg))))/2
    return F_loss

def pair_loss(emb_t1, emb_t2, delta_npz, pred_cd, pred_hiv, labels_cd, labels_hiv, lamb, tau):
    """
    emb_t1 (128-tensor): embedding for first image in pair
    emb_t2 (128-tensor): embedding for second image in pair
    delta_sigma (float): ground truth difference in value for factor to disentangle
    pred_1, pred_2 (tuple[float, float]): model's probability prediction for each of the labels, i.e. (0.12, 0.96)
    label_1, label_2 (tuple[int, int]): ground truth class membership for the two independent label, i.e. (0, 0), (0, 1), (1, 0), (1, 1)
    lamb: weighting factor between BCE loss and disentangle loss
    tau: current value of disentanglement vector
    """
    bce = nn.BCEWithLogitsLoss()
    # print(pred_cd,pred_hiv)
    # print(labels_cd, labels_hiv)
    bce_loss_cd = bce(pred_cd,labels_cd)
    # bce_loss_cd = focal_loss(pred_cd,labels_cd)
    bce_loss_hiv = bce(pred_hiv,labels_hiv)
    #bce_loss_hiv = focal_loss(pred_hiv,labels_hiv)
    bce_loss = bce_loss_cd + bce_loss_hiv

    proj_e1_len = torch.norm((torch.dot(emb_t1, tau) / torch.dot(tau, tau)) * tau)
    proj_e2_len = torch.norm((torch.dot(emb_t2, tau) / torch.dot(tau, tau)) * tau)
    emb_len_diff = torch.abs(proj_e2_len - proj_e1_len)
    disentangle_loss = torch.abs(emb_len_diff - delta_npz)
    # print(bce_loss.get_device())
    # print(disentangle_loss.get_device())
    return bce_loss + lamb*disentangle_loss

@torch.no_grad()
def test(feature_extractor, classifier_ucsf, loader, lamb=0.5, fold=None, epoch=None, train = False):
    feature_extractor.eval()
    classifier_ucsf.eval()

    xentropy_loss_avg = 0.
    paired_loss_avg = 0.
    correct = 0.
    total = 0.
    toprint = 0

    overall_accuracy = 0.
    correlation_ctrl = torch.tensor(0.)
    correlation_hiv = torch.tensor(0.)


    accuracy_class = {}
    accuracy_class['ucsf'] = {}

    accuracy_class['ucsf']['CTRL']=0
    accuracy_class['ucsf']['MCI']=0
    accuracy_class['ucsf']['HIV']=0
    accuracy_class['ucsf']['MND']=0
    accuracy_dataset = {}
    accuracy_dataset['ucsf']=0

    total_0_ucsf = 0.0
    total_1_ucsf = 0.0
    total_2_ucsf = 0.0
    total_3_ucsf = 0.0


    total_ucsf = 0

    feature_list = []
    all_datasets = []
    all_ids = []

    all_preds = []
    all_genders = []
    all_ages = []
    all_label_cd = []
    all_label_hiv = []


    # num_batches = 0
    # for i, _ in enumerate(loader):
    #     num_batches += 1
    num_samples = []
    for i, batch in enumerate(loader):
        num_samples.append(len(batch[0][0]))

    for i,data in enumerate(loader):
        # datasets = np.array(datasets)
        # ids = np.array(ids)
        # actual_labels = np.array(actual_labels)
        # images = images.to(device).float()
        # labels = labels.to(device).float()

        # if i == num_batches - 1:
        #     number_needed = int(args.batch_size) - len(images)
        #     images0, labels0, actual_labels0, datasets0, ids0, ages0, genders0 = first_batch_data
        #     images = torch.cat((images, images0[:number_needed,]),dim=0)
        #     labels = torch.cat((labels, labels0[:number_needed,]),dim=0)
        #     actual_labels = np.concatenate((actual_labels, actual_labels0[:number_needed,]),axis=0)
        #     datasets = np.concatenate((datasets, datasets0[:number_needed]),axis=0)
        #
        #     ids = np.concatenate((ids, ids0[:number_needed]),axis=0)
        #     ages = np.concatenate((ages, ages0[:number_needed]),axis=0)
        #     genders = np.concatenate((genders, genders0[:number_needed]),axis=0)

        images1 = data[0][0].to(device).float()
        images2 = data[0][1].to(device).float()

        labels1 = data[1][0].to(device).float()
        labels2 = data[1][1].to(device).float()

        labels = torch.cat((labels1,labels2))

        alabels1 = data[2][0].to(device).float()
        alabels2 = data[2][1].to(device).float()

        actual_labels = torch.cat((alabels1,alabels2))


        # cfs = get_cf_kernel_batch(data)
        # classifier_ucsf[2].cfs = cfs
        # classifier_ucsf[5].cfs = cfs
        # classifier_ucsf[7].cfs = cfs
        #forward pass
        images = torch.cat((images1,images2))
        features = feature_extractor(images)
        preds = classifier_ucsf(features)

        pred_cd = preds[:,0]
        pred_cd = torch.unsqueeze(pred_cd,1)
        pred_hiv = preds[:,1]
        pred_hiv = torch.unsqueeze(pred_hiv,1)

        # BELOW (TO REST OF FUNC) IS just metrics essentially
        labels_cd = labels[:,0]
        labels_cd = torch.unsqueeze(labels_cd,1)
        labels_hiv = labels[:,1]
        labels_hiv = torch.unsqueeze(labels_hiv,1)

        #get tau
        params = feature_extractor.state_dict()
        tau = params['feature_extractor.tau']

        bce = nn.BCEWithLogitsLoss()
        loss_cd = bce(pred_cd, labels_cd).to(device)
        loss_hiv = bce(pred_hiv, labels_hiv).to(device)

        paired_loss = loss_cd + loss_hiv


        # if i==0:
        #     first_batch_data = copy.deepcopy(data)

        ucsf_pred_cd = []
        ucsf_pred_hiv = []

        ucsf_labels_cd = []
        ucsf_labels_hiv = []
        ucsf_actual_labels = []
        ucsf_ids = []

        # for j in range(num_samples[i]):
        #
        #     #retrieve pair
        #     pair = [[data[k][0][j],data[k][1][j]] for k in range(7)]
        #     images, labels, actual_labels, datasets, ids, ages, genders = pair #now single tuples of size 2
        #
        #
        #     # features = feature_extractor(images)
        #     feature_1, feature_2 = features[j],features[j+num_samples[i]]
        #     # feature_1, feature_2 = feature_extractor(images[0][None, ...].to(device).float()), feature_extractor(images[1][None, ...].to(device).float())
        #
        #     pred_1, pred_2 = torch.squeeze(preds[j]),torch.squeeze(preds[j+num_samples[i]])
        #     # pred_1, pred_2 = torch.squeeze(classifier_ucsf(feature_1)),torch.squeeze(classifier_ucsf(feature_2))
        #
        #     pred = [pred_1, pred_2]
        #
        #     #get diff in npz score
        #     delta_npz = 0
        #     if ids[0] not in npz:
        #         pass
        #     else:
        #         delta_npz = npz[ids[0]] - npz[ids[1]]
        #
        #     pred_cd = torch.tensor([pred[0][0], pred[1][0]])
        #     pred_hiv = torch.tensor([pred[0][1], pred[1][1]])
        #     labels_cd = torch.tensor([labels[0][0], labels[1][0]])
        #     labels_hiv = torch.tensor([labels[0][1], labels[1][1]])
        #
        #     paired_loss = pair_loss(torch.squeeze(feature_1), torch.squeeze(feature_2), delta_npz, pred_cd, pred_hiv, labels_cd, labels_hiv, lamb, tau).to(device)
        #
        paired_loss_avg += paired_loss.item()
        #
        #     # BELOW (TO REST OF FUNC) IS just metrics essentially



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

        ucsf_pred_cd += pred_cd
        ucsf_pred_hiv += pred_hiv

        ucsf_labels_cd += labels_cd
        ucsf_labels_hiv += labels_hiv
        ucsf_actual_labels += actual_labels
        # ucsf_ids += ids


            # pred_cur = copy.deepcopy(pred)

        # remove duplicate test data
        # if i == num_batches - 1:
        #     number_actual = int(args.batch_size) - number_needed
        #     pred_cur = pred_cur[:number_actual,]
        #     pred_cd = pred_cd[:number_actual,]
        #     pred_hiv = pred_hiv[:number_actual,]
        #     labels_cd = labels_cd[:number_actual,]
        #     labels_hiv = labels_hiv[:number_actual,]
        #     datasets = datasets[:number_actual,]
        #     actual_labels = actual_labels[:number_actual,]
        #     ids= ids[:number_actual,]
        #     ages = ages[:number_actual,]
        #     genders = genders[:number_actual,]
        #
        #     pred_cd_copy = pred_cd_copy[:number_actual,]
        #     pred_hiv_copy = pred_hiv_copy[:number_actual,]
        #     feature = feature[:number_actual,]


        # feature_list.append(feature.cpu())
        # all_datasets = np.append(all_datasets,datasets)
        # all_ids = np.append(all_ids, ids)
        # all_preds.extend(pred_cur.detach().cpu().numpy())
        # all_genders.extend(genders)
        # all_ages.extend(ages)
        # all_label_cd.extend(labels_cd.squeeze().detach().cpu().numpy())
        # all_label_hiv.extend(labels_hiv.squeeze().detach().cpu().numpy())

        # ucsf


        # roc_hiv = roc_curve(np.array(ucsf_labels_hiv.cpu()), np.array(ucsf_pred_hiv.cpu()))
        # roc_cd = roc_curve(np.array(ucsf_labels_cd.cpu()), np.array(ucsf_pred_cd.cpu()))
        for j in range(0,len(ucsf_pred_cd)):
            total_ucsf += 1
            #UNCOMMENT BELOW TO LOG
            #if train == False:

                # row = {'epoch':epoch, 'id':ucsf_ids[j], 'dataset':'UCSF', 'CD_pred':torch.sigmoid(ucsf_pred_cd[j]).item(), 'HIV_pred':torch.sigmoid(ucsf_pred_hiv[j]).item(), 'fold': fold,'CD_label':ucsf_labels_cd[j].item(), 'HIV_label':ucsf_labels_hiv[j].item()}
                # csv_logger_pred.writerow(row)
            actual_pred = None
            if ucsf_pred_cd[j] == 0 and ucsf_pred_hiv[j] == 0 :
                actual_pred = 0
            elif ucsf_pred_cd[j] == 1 and ucsf_pred_hiv[j] == 0 :
                actual_pred = 1
            elif ucsf_pred_cd[j] == 0 and ucsf_pred_hiv[j] == 1 :
                actual_pred = 2
            elif ucsf_pred_cd[j] == 1 and ucsf_pred_hiv[j] == 1 :
                actual_pred = 3
            # print(actual_pred, ucsf_actual_labels[j])
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


    # all_feature = np.concatenate(feature_list, axis=0)
    # all_datasets_onehot = np.zeros([len(all_datasets),3])
    # all_datasets_onehot[all_datasets=='ucsf'] = [1,0,0]
    # all_datasets_onehot[all_datasets=='lab'] = [0,1,0]
    # all_datasets_onehot[all_datasets=='adni'] = [0,0,1]
    # distance = dcor.distance_correlation_sqr(all_feature, all_datasets_onehot)
    # correlation0_gender, correlation1_gender, correlation0_age, correlation1_age, distance_age, distance_gender = calculate_gender_age_correlation(all_preds, all_genders, all_ages, all_label_cd, all_label_hiv, all_feature)
    # print("corr0_g:", correlation0_gender, "corr1_g:", correlation1_gender, "corr0_a:", correlation0_age, "corr1_a", correlation1_age)
    # row = {'epoch':epoch, 'train':train, 'fold':fold, 'final_corr0_age':correlation0_age, 'final_corr1_age':correlation1_age, 'final_corr0_gender':correlation0_gender, 'final_corr1_gender':correlation1_gender, 'intermediate_age':distance_age, 'intermediate_gender':distance_gender}
    # csv_logger_corr.writerow(row)


    accuracy_class['ucsf']['CTRL'] = round(accuracy_class['ucsf']['CTRL'] / total_0_ucsf,3)
    accuracy_class['ucsf']['MCI'] = round(accuracy_class['ucsf']['MCI'] / total_1_ucsf,3)
    accuracy_class['ucsf']['HIV'] = round(accuracy_class['ucsf']['HIV'] / total_2_ucsf,3)
    accuracy_class['ucsf']['MND']= round(accuracy_class['ucsf']['MND'] / total_3_ucsf,3)

    accuracy_dataset['ucsf'] = round(accuracy_dataset['ucsf'] / total_ucsf,3)
    print(accuracy_class, total_ucsf)
    overall_accuracy = (correct) / (total)
    overall_accuracy = round(overall_accuracy,3)


    paired_loss_avg = paired_loss_avg / (i + 1)

    return overall_accuracy, paired_loss_avg, accuracy_class, accuracy_dataset#, distance, roc_hiv, roc_cd

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
    dataset_ucsf = []
    ages = []
    gender_m = []

    for i,(all_images, all_labels, all_actual_labels, all_datasets, all_ids, all_ages, all_genders) in enumerate(loader):
        for j in range(0,len(all_images)):
            labels=all_labels[j]
            actual_labels=all_actual_labels[j]
            datasets =  all_datasets[j]
            # print(datasets)

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
                dataset_ucsf.append(1)
            ages.append(all_ages[j])
            cur_gender = all_genders[j]
            if cur_gender == 0:
                gender_m.append(1)
            elif cur_gender == 1:
                gender_m.append(0)

    N = len(dataset_ucsf)

    X_shuffled = np.zeros((N,5))
    X_shuffled[:,0] = label_hiv
    X_shuffled[:,1] = label_cd
    X_shuffled[:,2] = np.ones((N,))
    X_shuffled[:,3] = ages
    X_shuffled[:,4] = gender_m
    # print(X_shuffled)
    cf_kernel = nn.Parameter(torch.tensor(np.linalg.inv(np.transpose(X_shuffled).dot(X_shuffled))).float().to(device),  requires_grad=False)

    return cf_kernel

def get_cf_kernel_batch(data):
    #flatten pairs
    all_images, all_labels, all_actual_labels, all_datasets, all_ids, all_ages, all_genders = ([] for _ in range(7))

    for j in range(len(data[0][0])):

        #retrieve pair
        pair = [[data[k][0][j],data[k][1][j]] for k in range(7)]
        images, labels, actual_labels, datasets, ids, ages, genders = pair #now single tuples of size 2
        all_images += images
        all_labels += labels
        all_actual_labels += actual_labels
        all_datasets += datasets
        all_ids += ids
        all_ages += ages
        all_genders += genders

    label_hiv = []
    label_cd = []
    dataset_ucsf = []
    ages = []
    gender_m = []
    N = len(all_images)
    # N = all_images.shape[0]
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
            dataset_ucsf.append(1)
        ages.append(all_ages[j])
        cur_gender = all_genders[j]
        if cur_gender == 0:
            gender_m.append(1)
        elif cur_gender == 1:
            gender_m.append(0)

    cfs_batch = np.zeros((N,5))
    # print(N)
    cfs_batch[:,0] = label_hiv
    cfs_batch[:,1] = label_cd
    cfs_batch[:,2] = np.ones((N,))
    cfs_batch[:,3] = ages
    cfs_batch[:,4] = gender_m

    cfs = nn.Parameter(torch.Tensor(cfs_batch).to(device).float(), requires_grad=False)


    return cfs

def train(feature_extractor,  classifier_ucsf, train_loader, test_loader,final_test_loader, fold, cf_kernel=None, lamb=0.5):

    feature_extractor.zero_grad()
    classifier_ucsf.zero_grad()

    fe_optimizer = optim.AdamW(feature_extractor.parameters(), lr =args.lr, weight_decay=0.01) # used to be 0.01
    ucsf_optimizer = optim.AdamW(classifier_ucsf.parameters(), lr =args.lr, weight_decay=args.wd) # used to be args.wd
    # tau_optimizer = optim.SGD([tau], lr = args.lr, weight_decay=0.01, momentum = 0.9)
    # fe_optimizer = optim.SGD(feature_extractor.parameters(), lr =args.lr, weight_decay=0.01, momentum = 0.9) # used to be 0.01
    # ucsf_optimizer = optim.SGD(classifier_ucsf.parameters(), lr =args.lr, weight_decay=args.wd, momentum = 0.9) # used to be args.wd

    best_accuracy = 0
    best_epoch = 0
    epochs = args.epochs
    ave_valid_acc_50 = 0.0
    counter = 0.0


    for epoch in range(epochs):
        feature_extractor.train()
        classifier_ucsf.train()

        progress_total = 0
        num_samples = []
        for i, batch in enumerate(train_loader):
            progress_total += 1
            num_samples.append(len(batch[0][0]))

        progress_bar = tqdm(train_loader, total = progress_total)
        # xentropy_loss_avg = 0.
        paired_loss_avg = 0.
        correct = 0.
        total = 0.
        ucsf_correct = 0.
        ucsf_total = 0.
        total = 0.
        overall_accuracy = 0

        ###### "Training happens here! ######

        #loop over batches
        for i, data in enumerate(progress_bar):

            feature_extractor.zero_grad()
            classifier_ucsf.zero_grad()

            images1 = data[0][0].to(device).float()
            images2 = data[0][1].to(device).float()


            labels1 = data[1][0].to(device).float()
            labels2 = data[1][1].to(device).float()
            # cfs = get_cf_kernel_batch(data)
            # classifier_ucsf[2].cfs = cfs
            # classifier_ucsf[5].cfs = cfs
            # classifier_ucsf[7].cfs = cfs

            images = torch.cat((images1,images2))
            labels = torch.cat((labels1,labels2))

            features = feature_extractor(images)
            preds = classifier_ucsf(features)

            pred_cd = preds[:,0]
            pred_cd = torch.unsqueeze(pred_cd,1)
            pred_hiv = preds[:,1]
            pred_hiv = torch.unsqueeze(pred_hiv,1)

            labels_cd = labels[:,0]
            labels_cd = torch.unsqueeze(labels_cd,1)
            labels_hiv = labels[:,1]
            labels_hiv = torch.unsqueeze(labels_hiv,1)

            bce = nn.BCEWithLogitsLoss()
            loss_cd = bce(pred_cd, labels_cd).to(device)
            loss_hiv = bce(pred_hiv, labels_hiv).to(device)

            paired_loss = loss_cd + loss_hiv

            params = feature_extractor.state_dict()
            tau = params['feature_extractor.tau']

            progress_bar.set_description('Epoch ' + str(epoch))

            #Look at each image pair in the batch to compute loss

            # for j in range(num_samples[i]):
            #
            #     #retrieve pair
            #     pair = [[data[k][0][j],data[k][1][j]] for k in range(7)]
            #     images, labels, actual_labels, datasets, ids, ages, genders = pair #now single tuples of size 2
            #     # print(labels, actual_labels, datasets, ids, ages, genders)
            #
            #     # features = feature_extractor(images)
            #     feature_1, feature_2 = features[j],features[j+num_samples[i]]
            #     # feature_1, feature_2 = feature_extractor(images[0][None, ...].to(device).float()), feature_extractor(images[1][None, ...].to(device).float())
            #
            #     pred_1, pred_2 = torch.squeeze(preds[j]),torch.squeeze(preds[j+num_samples[i]])
            #     # pred_1, pred_2 = torch.squeeze(classifier_ucsf(feature_1)),torch.squeeze(classifier_ucsf(feature_2))
            #     pred = [pred_1, pred_2]
            #
            #     #get diff in npz score
            #     delta_npz = 0
            #     if ids[0] not in npz:
            #         pass
            #     else:
            #         delta_npz = npz[ids[0]] - npz[ids[1]]
            #
            #     pred_cd1 = torch.tensor([pred[0][0], pred[1][0]])
            #     pred_hiv1 = torch.tensor([pred[0][1], pred[1][1]])
            #     labels_cd = torch.tensor([labels[0][0], labels[1][0]])
            #     labels_hiv = torch.tensor([labels[0][1], labels[1][1]])
            #     # print(pred_hiv1)
            #     paired_loss = pair_loss(torch.squeeze(feature_1), torch.squeeze(feature_2), delta_npz, pred_cd1, pred_hiv1, labels_cd, labels_hiv, lamb, tau).to(device)
            #     #backwards pass
            #     paired_loss.backward(retain_graph=True)

                ###### End of "training" is here! ######

                #Metrics
                # paired_loss_sum += paired_loss
            paired_loss.backward()
            paired_loss_avg += paired_loss.item()



            # pred_cd = pred_cd.clone()
            pred_cd[pred_cd>0]=1
            pred_cd[pred_cd<0]=0

            # pred_hiv = pred_hiv1.clone()
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



            # torch.nn.utils.clip_grad_norm_(feature_extractor.parameters(), 1.0)
            # torch.nn.utils.clip_grad_norm_(classifier_ucsf.parameters(), 1.0)

            fe_optimizer.step()
            ucsf_optimizer.step()
            # new_params = feature_extractor.state_dict()
            # ##reset tau to train jointly
            # new_params['feature_extractor.tau'] = torch.ones(2048).float()
            # feature_extractor.load_state_dict(new_params)


            progress_bar.set_postfix(
                loss='%.6f' % (paired_loss_avg / (i + 1)),
                acc='%.2f' % overall_accuracy)

        test_acc, test_ploss,test_accuracy_class, test_accuracy_dataset = test(feature_extractor, classifier_ucsf, test_loader, lamb=args.lamb, fold =fold, epoch = epoch)

        test_ucsf_ctrl = test_accuracy_class['ucsf']['CTRL']
        test_ucsf_mci = test_accuracy_class['ucsf']['MCI']
        test_ucsf_hiv = test_accuracy_class['ucsf']['HIV']
        test_ucsf_mnd = test_accuracy_class['ucsf']['MND']

        ucsf_test_acc = np.mean([test_ucsf_ctrl, test_ucsf_mci, test_ucsf_hiv, test_ucsf_mnd])

        test_accuracy_dataset['ucsf'] = round(ucsf_test_acc,3)

        print('test:',test_accuracy_class['ucsf'])

        # this training accuracy has augmentation in it!!!!!
        # some images are sampled more than once!!!!
        train_acc, train_ploss,train_accuracy_class, train_accuracy_dataset = test(feature_extractor, classifier_ucsf, train_loader, lamb=args.lamb, fold =fold, epoch = epoch, train =True)

        train_ucsf_ctrl = train_accuracy_class['ucsf']['CTRL']
        train_ucsf_mci = train_accuracy_class['ucsf']['MCI']
        train_ucsf_hiv = train_accuracy_class['ucsf']['HIV']
        train_ucsf_mnd = train_accuracy_class['ucsf']['MND']

        # redefine ucsf_train_acc, lab_val_acc to be the average of all classes
        ucsf_train_acc = np.mean([train_ucsf_ctrl, train_ucsf_mci, train_ucsf_hiv, train_ucsf_mnd])
        train_accuracy_dataset['ucsf'] = round(ucsf_train_acc,3)

        tqdm.write('train_acc: %.2f u_train_acc: %.2f' % (overall_accuracy, ucsf_train_acc))
        tqdm.write('test_acc: %.2f u_test_acc: %.2f test_ploss: %.2f' % (test_acc, ucsf_test_acc, test_ploss))
        #UNCOMMENT TO LOG TRAINING STATS TO FILE
        # row = {'epoch': epoch, 'train_acc': round(overall_accuracy,3), 'test_acc': test_acc, 'train_ploss':round((paired_loss_avg / (i + 1)),3), 'test_ploss': round(test_ploss,3),
        #        'ucsf_train_acc': ucsf_train_acc,
        #        'ucsf_test_acc': ucsf_test_acc,
        #        'correlation_ctrl_train':0,
        #        'correlation_hiv_train':0,
        #
        #        'correlation_ctrl_test':0,
        #        'correlation_hiv_test':0,
        #        'train_ucsf_ctrl':train_ucsf_ctrl, 'train_ucsf_mci':train_ucsf_mci,
        #        'train_ucsf_hiv':train_ucsf_hiv, 'train_ucsf_mnd':train_ucsf_mnd,
        #
        #        'test_ucsf_ctrl':test_ucsf_ctrl, 'test_ucsf_mci':test_ucsf_mci,
        #        'test_ucsf_hiv':test_ucsf_hiv, 'test_ucsf_mnd':test_ucsf_mnd,
        #       'train_distance':train_distance,'test_distance':test_distance}
        # csv_logger.writerow(row)

        #UNCOMMENT TO LOG TRAINING STATS TO WANDB
        wandb.log({ "train ploss": round((paired_loss_avg / (i + 1)),3), 'test_ploss': round(test_ploss,3), 'train_acc': round(overall_accuracy,3), 'test_acc': test_acc,'train_ucsf_ctrl':train_ucsf_ctrl, 'train_ucsf_mci':train_ucsf_mci,
        'train_ucsf_hiv':train_ucsf_hiv, 'train_ucsf_mnd':train_ucsf_mnd,

        'test_ucsf_ctrl':test_ucsf_ctrl, 'test_ucsf_mci':test_ucsf_mci,
        'test_ucsf_hiv':test_ucsf_hiv, 'test_ucsf_mnd':test_ucsf_mnd
        # 'train_fpr_hiv': np.mean(train_roc_hiv[0]), 'train_fpr_cd': np.mean(train_roc_cd[0]),'train_tpr_hiv': np.mean(train_roc_hiv[1]),'train_tpr_cd': np.mean(train_roc_cd[1]),
        # 'test_fpr_cd': np.mean(test_roc_cd[0]), 'test_fpr_hiv': np.mean(test_roc_hiv[0]),'test_tpr_cd': np.mean(test_roc_cd[1]),'test_tpr_hiv': np.mean(test_roc_hiv[1])
        })

        wandb.watch(feature_extractor)
        wandb.watch(classifier_ucsf)


        if test_acc > best_accuracy:
            best_accuracy = test_acc
            best_epoch = epoch


    best_models = [feature_extractor, classifier_ucsf]

    return test_acc, test_accuracy_class, test_accuracy_dataset, best_accuracy, best_epoch, best_models

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
    csv_logger_sets =  CSVLogger( args,fieldnames=['ids','datasets'],filename='idss.csv')
    csv_logger = CSVLogger( args, fieldnames=['epoch', 'train_acc',  'test_acc',
                                              'train_ploss','test_ploss'
                                              'ucsf_train_acc',

                                              'ucsf_test_acc',
                                              'correlation_ctrl_train', 'correlation_hiv_train',

                                              'correlation_ctrl_test', 'correlation_hiv_test',
                                              'train_ucsf_ctrl','train_ucsf_mci', 'train_ucsf_hiv','train_ucsf_mnd',

                                              'test_ucsf_ctrl','test_ucsf_mci','test_ucsf_hiv','test_ucsf_mnd',

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

#      #EXTRACT ONLY UCSF DATA

        ucsf_data = []
        for i,(images, labels, actual_labels, datasets, ids, ages, genders) in enumerate(train_loader):

            #id_tracker.append(ids)
            if datasets == 'ucsf':
                csv_logger_sets.writerow({'ids':ids, 'datasets':'train'})
                ucsf_data.append((images, labels, actual_labels, datasets, ids, ages, genders))

        ucsf_test_data = []
        for i,(images, labels, actual_labels, datasets, ids, ages, genders) in enumerate(test_loader):

            if datasets == 'ucsf':
                csv_logger_sets.writerow({'ids':ids, 'datasets':'test'})
                ucsf_test_data.append((images, labels, actual_labels, datasets, ids, ages, genders))
        #separate out non-npz and npz patients, discard last of list if it makes it odd
        train_npz = [x for x in ucsf_data if float(x[4]) in npz]
        if len(train_npz) % 2 != 0:
            train_npz.pop()
        train_not = [x for x in ucsf_data if float(x[4]) not in npz]
        if len(train_not) % 2 != 0:
            train_not.pop()
        test_npz = [x for x in ucsf_test_data if float(x[4]) in npz]
        if len(test_npz) % 2 != 0:
            test_npz.pop()
        test_not = [x for x in ucsf_test_data if float(x[4]) not in npz]
        if len(test_not) % 2 != 0:
            test_not.pop()
        print('train/test npz/not split')
        print(len(train_npz), len(train_not), len(test_npz), len(test_not))
        #make pairs
        train_npz_pairs = [[train_npz[i], train_npz[i+1]] for i in range(0, len(train_npz), 2)]
        train_not_pairs = [[train_not[i], train_not[i+1]] for i in range(0, len(train_not), 2)]
        test_npz_pairs = [[test_npz[i], test_npz[i+1]] for i in range(0, len(test_npz), 2)]
        test_not_pairs = [[test_not[i], test_not[i+1]] for i in range(0, len(test_not), 2)]

        ucsf_dataset = Paired_UCSF_Dataset(train_npz_pairs+train_not_pairs)
        kernel_dataset = Specific_MRI_Dataset(ucsf_data)
        ucsf_test_dataset = Paired_UCSF_Dataset(test_npz_pairs+test_not_pairs)
        # print(ucsf_dataset[0])

        ucsf_train_loader = DataLoader(dataset=ucsf_dataset,
                                  batch_size=args.batch_size,
                                  # sampler=ClassMixedSampler(dataset=ucsf_data, batch_size=args.batch_size),

                                  shuffle=False,
                                  pin_memory=True,
                                  num_workers=3)
        ucsf_test_loader = DataLoader(dataset=ucsf_test_dataset ,
                                  batch_size=args.batch_size,# to include all test images
                                  shuffle=False,
                                  pin_memory=True,
                                  num_workers=3)
        ucsf_final_test_loader = DataLoader(dataset=ucsf_test_dataset ,
                          batch_size=1,#args.batch_size,
                          shuffle=False,
                          pin_memory=True,
                          num_workers=3)


        kernel_loader = DataLoader(dataset=kernel_dataset,
                                  batch_size=args.batch_size,
                                  # sampler=ClassMixedSampler(dataset=ucsf_data, batch_size=args.batch_size),

                                  shuffle=False,
                                  pin_memory=True,
                                  num_workers=3)
        # cf_kernel  = get_cf_kernel(kernel_loader)
        feature_extractor = fe(trainset_size = len(ucsf_data), in_num_ch=1, img_size=(64, 64, 64), inter_num_ch=16,
                           fc_num_ch=16, kernel_size=3, conv_act='relu',
                           fe_arch=args.fe_arch, dropout=args.dropout,
                           fc_dropout = args.fc_dropout, batch_size = args.batch_size).to(device)

        classifier_ucsf = nn.Sequential(
                     # MetadataNorm(batch_size=args.batch_size, cf_kernel=cf_kernel, num_features = 2048, trainset_size = len(train_data)),
                    nn.Linear(2048, 128),
                    nn.LeakyReLU(),
                     # nn.BatchNorm1d(128),
                    # MetadataNorm(batch_size=args.batch_size*2, cf_kernel=cf_kernel, num_features = 128, trainset_size = len(ucsf_data)),
                    nn.Linear(128,16),
                    nn.LeakyReLU(),
                     # nn.BatchNorm1d(16),
                   # MetadataNorm(batch_size=args.batch_size*2, cf_kernel=cf_kernel, num_features = 16, trainset_size = len(ucsf_data)),
                     nn.Linear(16, 2),
                     # MetadataNorm(batch_size=args.batch_size*2, cf_kernel=cf_kernel, num_features = 2, trainset_size = len(ucsf_data)),
                ).to(device)

        test_acc, test_accuracy_class, test_accuracy_dataset, best_accuracy, best_epoch, best_models = train(feature_extractor,  classifier_ucsf, ucsf_train_loader,  ucsf_test_loader, ucsf_final_test_loader, fold = fold)


        feature_extractor, classifier_ucsf = best_models
        best_accuracy_list[fold] = best_accuracy
        final_accuracy_list[fold] = test_acc
        best_epoch_list[fold] = best_epoch


        test_acc, test_loss,test_accuracy_class, test_accuracy_dataset, test_distance = test(feature_extractor, classifier_ucsf, ucsf_test_loader, lamb=args.lamb, fold = fold, train =True)
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

