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
from sampler import SuperSampler,MixedSampler,UCSFSampler,PairedSampler
from metadatanorm2 import MetadataNorm
import dcor
import pickle

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

wandb.init(project="new_paired", entity="jmanasse", config = {
  "learning_rate": args.lr,
  "epochs": args.epochs,
  "batch_size": args.batch_size,
  "fc dropout": args.fc_dropout,
  'dropout': args.dropout,
  "weight decay": args.wd,
  'loss type': 'w/focal loss',
  'optimizer': 'added CLR'
})

print("device:",device)



class FocalLoss(nn.Module):
    def __init__(self, alpha=0.1, gamma=2, dataset = 'ucsf'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.criterion = torch.nn.BCEWithLogitsLoss(reduction= 'none').to(device)
        # self.criterion = torch.nn.BCEWithLogitsLoss().to(device)
        self.dataset = dataset

    def forward(self, inputs, targets):
        BCE_loss = self.criterion(inputs, targets)
        #
        # if inputs.shape[0] == 1:
        #     return BCE_loss
        #
        # pt = torch.exp(-BCE_loss)
        # F_loss= 0
        #
        #
        # F_loss_pos = self.alpha * (1-pt[targets==1])**self.gamma * BCE_loss[targets==1]
        # F_loss_neg = (1-self.alpha) * (1-pt[targets==0])**self.gamma * BCE_loss[targets==0]
        #
        # if inputs.shape[0] == 1:
        #     if F_loss_pos.nelement() > 0:
        #         return F_loss_pos
        #     else:
        #         return F_loss_neg
        #
        # F_loss += (torch.mean(F_loss_pos)+torch.mean(F_loss_neg))/2
        #
        # return F_loss
        return BCE_loss


L1_lambda = args.L1_lambda
L2_lambda = args.L2_lambda

alpha1 = args.alpha[0]
alpha2 = args.alpha[1]
ucsf_criterion_cd = FocalLoss(alpha = alpha1, gamma = args.gamma, dataset = 'ucsf')
ucsf_criterion_hiv = FocalLoss(alpha = alpha2, gamma = args.gamma, dataset = 'ucsf')

@torch.no_grad()
def test(feature_extractor, classifier_ucsf, loader,  ucsf_criterion_cd, ucsf_criterion_hiv, fold=None, epoch=None, train = False):
    feature_extractor.eval()
    classifier_ucsf.eval()

    xentropy_loss_avg = 0.
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

    num_samples = []
    for i, batch in enumerate(loader):
        num_samples.append(len(batch[0][0]))

    # print(num_samples)
    num_batches = 0
    for i, _ in enumerate(loader):
        num_batches += 1
    # print(num_batches)
    dataset_length = 115
    l = 0

    # batch accumulation parameter
    accum_iter = args.batch_size//2

    for i,(images, labels,actual_labels,datasets,ids, ages, genders, npzs) in enumerate(loader):
        ucsf_pred_cd = []
        ucsf_pred_hiv = []
        ucsf_actual_labels = []

        for j in range(num_samples[i]//2):
            #remove duplicate dataset
            if l > dataset_length and not train:
                # print('broken')
                break
            l += 2
            #get images in pair
            images1 = images[0][j].to(device).float()
            images2 = images[1][j].to(device).float()

            #forward pass
            feature1 = feature_extractor(images1)
            feature2 = feature_extractor(images2)
            pred1 = classifier_ucsf(feature1)
            pred2 = classifier_ucsf(feature2)
            pred = torch.cat((pred1, pred2))

            label = labels[0][j].repeat(2,1).to(device).float() #same for both images in pair

            #now, compute loss
            pred_cd1 = pred[:,0].unsqueeze(1)
            pred_hiv1 = pred[:,1].unsqueeze(1)
            labels_cd = label[:,0].unsqueeze(1)
            labels_hiv = label[:,1].unsqueeze(1)

            losscd = ucsf_criterion_cd(pred_cd1, labels_cd).to(device)
            losshiv = ucsf_criterion_hiv(pred_hiv1, labels_hiv).to(device)
            loss = torch.cat((losscd.unsqueeze(0), losshiv.unsqueeze(0)))
            xentropy_loss = torch.nansum(loss).mean()

            # normalize loss to account for batch accumulation
            xentropy_loss = xentropy_loss/accum_iter

            xentropy_loss_avg += xentropy_loss.item()

            #compute accuracy
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
            total += 2

            ucsf_pred_cd.extend(list(pred_cd))
            ucsf_pred_hiv.extend(list(pred_hiv))

            actual1 = actual_labels[0][j]
            actual2 = actual_labels[1][j]

            ucsf_actual_labels.extend([actual1,actual2])

        overall_accuracy = correct/total
        # print(correct,total)

        for j in range(0,len(ucsf_pred_cd)):
            total_ucsf += 1
            # if train == False:
            #
            #     row = {'epoch':epoch, 'id':ucsf_ids[j], 'dataset':'UCSF', 'CD_pred':torch.sigmoid(ucsf_pred_cd_copy[j]).item(), 'HIV_pred':torch.sigmoid(ucsf_pred_hiv_copy[j]).item(), 'fold': fold,'CD_label':ucsf_labels_cd[j].item(), 'HIV_label':ucsf_labels_hiv[j].item()}
            #     csv_logger_pred.writerow(row)

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
    # # calculate correlations
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

    # print(total_0_ucsf,total_1_ucsf,total_2_ucsf,total_3_ucsf)

    accuracy_class['ucsf']['CTRL'] = round(accuracy_class['ucsf']['CTRL'] / total_0_ucsf,3)
    accuracy_class['ucsf']['MCI'] = round(accuracy_class['ucsf']['MCI'] / total_1_ucsf,3)
    accuracy_class['ucsf']['HIV'] = round(accuracy_class['ucsf']['HIV'] / total_2_ucsf,3)
    accuracy_class['ucsf']['MND']= round(accuracy_class['ucsf']['MND'] / total_3_ucsf,3)

    accuracy_dataset['ucsf'] = round(accuracy_dataset['ucsf'] / total_ucsf,3)
    print(accuracy_class, total_ucsf)
    overall_accuracy = (correct) / (total)
    overall_accuracy = round(overall_accuracy,3)


    xentropy_loss_avg = xentropy_loss_avg / (i + 1)

    return overall_accuracy, xentropy_loss_avg,accuracy_class, accuracy_dataset#, distance

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
#     X_shuffled[:,2] = dataset_lab
#     X_shuffled[:,3] = dataset_adni
#     X_shuffled[:,4] = dataset_ucsf
    X_shuffled[:,2] = np.ones((N,))
    X_shuffled[:,3] = ages
    X_shuffled[:,4] = gender_m

    cf_kernel = nn.Parameter(torch.tensor(np.linalg.inv(np.transpose(X_shuffled).dot(X_shuffled))).float().to(device),  requires_grad=False)

    return cf_kernel

def get_cf_kernel_batch(data):
    all_images, all_labels, all_actual_labels, all_datasets, all_ids, all_ages, all_genders = data
    label_hiv = []
    label_cd = []
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
            dataset_ucsf.append(1)

        ages.append(all_ages[j])
        cur_gender = all_genders[j]
        if cur_gender == 0:
            gender_m.append(1)
        elif cur_gender == 1:
            gender_m.append(0)

    cfs_batch = np.zeros((N,5))
    cfs_batch[:,0] = label_hiv
    cfs_batch[:,1] = label_cd
    # cfs_batch[:,2] = dataset_lab
    # cfs_batch[:,3] = dataset_adni
#     cfs_batch[:,4] = dataset_ucsf
    cfs_batch[:,2] = np.ones((N,))
    cfs_batch[:,3] = ages
    cfs_batch[:,4] = gender_m

    cfs = nn.Parameter(torch.Tensor(cfs_batch).to(device).float(), requires_grad=False)


    return cfs


def train(feature_extractor,  classifier_ucsf, train_loader, test_loader, fold ):

    feature_extractor.zero_grad()
    classifier_ucsf.zero_grad()

    fe_optimizer = optim.AdamW(feature_extractor.parameters(), lr =args.lr, weight_decay=args.wd) # used to be 0.01
    ucsf_optimizer = optim.AdamW(classifier_ucsf.parameters(), lr =args.lr, weight_decay=args.wd) # used to be args.wd

    fe_scheduler = torch.optim.lr_scheduler.CyclicLR(fe_optimizer, base_lr=args.lr, max_lr=args.lr*5,cycle_momentum=False)
    ucsf_scheduler = torch.optim.lr_scheduler.CyclicLR(ucsf_optimizer, base_lr=args.lr, max_lr=args.lr*5,cycle_momentum=False)

    best_accuracy = 0
    best_epoch = 0
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
            num_samples.append(len(batch[0][0]))

        progress_bar = tqdm(train_loader, total = progress_total)
        xentropy_loss_avg = 0.
        cur_loss_sum = 0
        correct = 0.
        total = 0.
        ucsf_correct = 0.
        ucsf_total = 0.
        total = 0.
        overall_accuracy = 0
        # batch accumulation parameter
        accum_iter = args.batch_size//2
        if epoch == 99:
            torch.save(feature_extractor.state_dict(), 'fe_plain_weights.pt')
            torch.save(classifier_ucsf.state_dict(), 'class_plain_weights.pt')
        #
        ###### "Training happens here! ######

        for i, (images, labels, actual_labels, datasets, ids, ages, genders, npzs) in enumerate(progress_bar):
            progress_bar.set_description('Epoch ' + str(epoch))

            for j in range(num_samples[i]//2):
                #get images in pair
                images1 = images[0][j].to(device).float()
                images2 = images[1][j].to(device).float()

                #forward pass
                feature1 = feature_extractor(images1)
                feature2 = feature_extractor(images2)
                pred1 = classifier_ucsf(feature1)
                pred2 = classifier_ucsf(feature2)
                pred = torch.cat((pred1, pred2))

                label = labels[0][j].repeat(2,1).to(device).float() #same for both images in pair

                #now, compute loss
                pred_cd1 = pred[:,0].unsqueeze(1)
                pred_hiv1 = pred[:,1].unsqueeze(1)
                labels_cd = label[:,0].unsqueeze(1)
                labels_hiv = label[:,1].unsqueeze(1)

                losscd = ucsf_criterion_cd(pred_cd1, labels_cd).to(device)
                losshiv = ucsf_criterion_hiv(pred_hiv1, labels_hiv).to(device)
                loss = torch.cat((losscd.unsqueeze(0), losshiv.unsqueeze(0)))
                xentropy_loss = torch.nansum(loss).mean()

                # normalize loss to account for batch accumulation
                xentropy_loss = xentropy_loss/accum_iter

                xentropy_loss.backward()

                xentropy_loss_avg += xentropy_loss.item()

                if (j+1) == (num_samples[i]//2):
                    #step for each batch
                    fe_optimizer.step()
                    ucsf_optimizer.step()
                    feature_extractor.zero_grad()
                    classifier_ucsf.zero_grad()

                #compute accuracy
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
                total += 2


            overall_accuracy= correct/total

            # #step for each batch
            # fe_optimizer.step()
            # ucsf_optimizer.step()
            # feature_extractor.zero_grad()
            # classifier_ucsf.zero_grad()
            ###### End of "training" is here! ######

            progress_bar.set_postfix(
                loss='%.6f' % (xentropy_loss_avg / (i + 1)),
                acc='%.2f' % overall_accuracy)
            #UPDATE LR
            fe_scheduler.step()
            ucsf_scheduler.step()

        #evaluate model
        test_acc, test_loss,test_accuracy_class, test_accuracy_dataset = test(feature_extractor, classifier_ucsf, test_loader, ucsf_criterion_cd, ucsf_criterion_hiv,fold =fold, epoch = epoch)

        test_ucsf_ctrl = test_accuracy_class['ucsf']['CTRL']
        test_ucsf_mci = test_accuracy_class['ucsf']['MCI']
        test_ucsf_hiv = test_accuracy_class['ucsf']['HIV']
        test_ucsf_mnd = test_accuracy_class['ucsf']['MND']

        ucsf_test_acc = np.mean([test_ucsf_ctrl, test_ucsf_mci, test_ucsf_hiv, test_ucsf_mnd])

        test_accuracy_dataset['ucsf'] = round(ucsf_test_acc,3)

        print('test:',test_accuracy_class['ucsf'])

        # this trainning accuracy has augmentation in it!!!!!
        # some images are sampled more than once!!!!
        train_acc, train_loss,train_accuracy_class, train_accuracy_dataset  = test(feature_extractor, classifier_ucsf, train_loader, ucsf_criterion_cd, ucsf_criterion_hiv,fold =fold, epoch = epoch, train =True)

        train_ucsf_ctrl = train_accuracy_class['ucsf']['CTRL']
        train_ucsf_mci = train_accuracy_class['ucsf']['MCI']
        train_ucsf_hiv = train_accuracy_class['ucsf']['HIV']
        train_ucsf_mnd = train_accuracy_class['ucsf']['MND']

        # redefine ucsf_train_acc, lab_val_acc to be the average of all classes
        ucsf_train_acc = np.mean([train_ucsf_ctrl, train_ucsf_mci, train_ucsf_hiv, train_ucsf_mnd])

        train_accuracy_dataset['ucsf'] = round(ucsf_train_acc,3)

        tqdm.write('train_acc: %.2f u_train_acc: %.2f' % (overall_accuracy, ucsf_train_acc))
        tqdm.write('test_acc: %.2f u_test_acc: %.2f test_loss: %.2f' % (test_acc, ucsf_test_acc, test_loss))

        row = {'epoch': epoch, 'train_acc': round(overall_accuracy,3), 'test_acc': test_acc, 'train_loss':round((xentropy_loss_avg / (i + 1)),3),  'test_loss': round(test_loss,3),
               'ucsf_train_acc': ucsf_train_acc,
               'ucsf_test_acc': ucsf_test_acc,
               'train_ucsf_ctrl':train_ucsf_ctrl, 'train_ucsf_mci':train_ucsf_mci,
               'train_ucsf_hiv':train_ucsf_hiv, 'train_ucsf_mnd':train_ucsf_mnd,

               'test_ucsf_ctrl':test_ucsf_ctrl, 'test_ucsf_mci':test_ucsf_mci,
               'test_ucsf_hiv':test_ucsf_hiv, 'test_ucsf_mnd':test_ucsf_mnd}
        csv_logger.writerow(row)

        wandb.log({"train loss": round((xentropy_loss_avg / (i + 1)),3),"test loss":round(test_loss,3), 'train_acc': round(overall_accuracy,3), 'test_acc': test_acc,'train_ucsf_ctrl':train_ucsf_ctrl, 'train_ucsf_mci':train_ucsf_mci,
        'train_ucsf_hiv':train_ucsf_hiv, 'train_ucsf_mnd':train_ucsf_mnd,

        'test_ucsf_ctrl':test_ucsf_ctrl, 'test_ucsf_mci':test_ucsf_mci,
        'test_ucsf_hiv':test_ucsf_hiv, 'test_ucsf_mnd':test_ucsf_mnd})

# Optional
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
    csv_logger = CSVLogger( args, fieldnames=['epoch', 'train_acc',  'test_acc',
                                              'train_loss','test_loss',
                                              'ucsf_train_acc',

                                              'ucsf_test_acc',
                                              'train_ucsf_ctrl','train_ucsf_mci', 'train_ucsf_hiv','train_ucsf_mnd',

                                              'test_ucsf_ctrl','test_ucsf_mci','test_ucsf_hiv','test_ucsf_mnd'],
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
                                  sampler=PairedSampler(dataset=train_data,
                                                       batch_size=args.batch_size, num_batches=15),
                                  shuffle=False,
                                  pin_memory=True,
                                  num_workers=3)

        batch_num_wo_replacement = len(test_data)//args.batch_size + 2
        print(batch_num_wo_replacement)
        test_loader = DataLoader(dataset=test_data ,
                                      batch_size=args.batch_size,
                                      sampler=PairedSampler(dataset=test_data,  batch_size=args.batch_size,num_batches=batch_num_wo_replacement),# to include all test images
                                      shuffle=False,
                                      pin_memory=True,
                                      num_workers=3)

        print("Begin training fold ",fold)


        feature_extractor = fe(trainset_size = len(train_data), in_num_ch=1, img_size=(64, 64, 64), inter_num_ch=16,
                           fc_num_ch=16, kernel_size=3, conv_act='LeakyReLU',
                           fe_arch=args.fe_arch, dropout=args.dropout,
                           fc_dropout = args.fc_dropout, batch_size = args.batch_size).to(device)

        classifier_ucsf = nn.Sequential(
                    nn.Linear(256, 128),
                    nn.LeakyReLU(),
                    nn.Linear(128, 16),
                    nn.LeakyReLU(),
                    nn.Linear(16, 2)).to(device)


        test_acc, test_accuracy_class, test_accuracy_dataset, best_accuracy, best_epoch, best_models = train(feature_extractor,  classifier_ucsf, train_loader,  test_loader, fold = fold)


        feature_extractor, classifier_ucsf = best_models
        best_accuracy_list[fold] = best_accuracy
        final_accuracy_list[fold] = test_acc
        best_epoch_list[fold] = best_epoch


        test_acc, test_loss,test_accuracy_class, test_accuracy_dataset = test(feature_extractor, classifier_ucsf, test_loader, ucsf_criterion_cd, ucsf_criterion_hiv,  fold = fold, train =True)
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

