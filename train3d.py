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

from tqdm import tqdm
import torchvision
from torch.optim.lr_scheduler import CosineAnnealingLR
from misc import CSVLogger
import copy
import gc
from transformation import super_transformation
#from sampler import SuperSampler,MixedSampler
from class_sampler import ClassMixedSampler
from metadatanorm2 import MetadataNorm
import dcor
import pickle
#
# import torch.optim as optim
# from ray import tune
# from ray.tune.examples.mnist_pytorch import get_data_loaders, ConvNet, train, test
#



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




# print("device:",device)
#
#
# def train_mnist(config):
#     train_loader, test_loader = get_data_loaders()
#     model = ConvNet()
#     optimizer = optim.SGD(model.parameters(), lr=config["lr"])
#     for i in range(10):
#         train(model, optimizer, train_loader)
#         acc = test(model, test_loader)
#         tune.report(mean_accuracy=acc)
#
#
# analysis = tune.run(
#     train_mnist, config={"lr": tune.grid_search([0.001, 0.01, 0.1])})
#
# print("Best config: ", analysis.get_best_config(metric="mean_accuracy"))

# Get a dataframe for analyzing trial results.
#df = analysis.dataframe()

class FocalLoss(nn.Module):
    #for 'dense object detection' (small objects inside large and busy images)
    #binary cross entropy requires the model to have higher confidence when making predictions
    #focal loss is useful when there is a class imbalance in the dataset
    def __init__(self, alpha=0.1, gamma=2, dataset = 'ucsf'):
        super(FocalLoss, self).__init__()
        #alpha = compensation for class imbalance
        self.alpha = alpha
        #gamma = how forgiving you are of low confidence
        self.gamma = gamma
        self.criterion = torch.nn.BCEWithLogitsLoss(reduction= 'none').to(device)
#         self.criterion = torch.nn.BCEWithLogitsLoss().to(device)
        self.dataset = dataset

    #compute loss on a forward pass through the network!
    def forward(self, inputs, targets):
        BCE_loss = self.criterion(inputs, targets)

        if inputs.shape[0] == 1:
            return BCE_loss
        #pt = probability of the ground-truth class (confidence of the model that x is x)
        pt = torch.exp(-BCE_loss)

        #compute focal loss. see 'Focal Loss for Dense Object Detection' (https://arxiv.org/abs/1708.02002)
        F_loss= 0

        #compute for each direction (misses and hits)
        F_loss_pos = self.alpha * (1-pt[targets==1])**self.gamma * BCE_loss[targets==1]
        F_loss_neg = (1-self.alpha) * (1-pt[targets==0])**self.gamma * BCE_loss[targets==0]
        #print(F_loss_pos,F_loss_neg)

        if inputs.shape[0] == 1:
            if F_loss_pos.nelement() > 0:
                return F_loss_pos
            else:
                return F_loss_neg
        F_loss += (torch.mean(F_loss_pos)+torch.mean(F_loss_neg))/2
        # pos_mean = torch.mean(F_loss_pos) if F_loss_pos.nelement() > 0 else 0
        # neg_mean = torch.mean(F_loss_neg) if F_loss_neg.nelement() > 0 else 0
        # F_loss += (pos_mean + neg_mean)/2
        #print(F_loss)

        return F_loss

#compute losses!
L1_lambda = args.L1_lambda
L2_lambda = args.L2_lambda

alpha1 = args.alpha[0]
alpha2 = args.alpha[1]
# ucsf_criterion_cd = FocalLoss(alpha = alpha1, gamma = args.gamma, dataset = 'ucsf')
# ucsf_criterion_hiv = FocalLoss(alpha = alpha2, gamma = args.gamma, dataset = 'ucsf')
# adni_criterion_cd = FocalLoss(alpha = alpha1, gamma = args.gamma, dataset = 'adni')
# lab_criterion_hiv = FocalLoss(alpha = alpha2, gamma = args.gamma, dataset = 'lab')
# ucsf_criterion_cd = FocalLoss(alpha = 0.3, gamma = 1.5, dataset = 'ucsf')
# ucsf_criterion_hiv = FocalLoss(alpha = 0.4, gamma = 1.75, dataset = 'ucsf')
# adni_criterion_cd = FocalLoss(alpha = 0.5, gamma = 1, dataset = 'adni')
# lab_criterion_hiv = FocalLoss(alpha = 0.5, gamma = 1, dataset = 'lab')

ucsf_criterion_cd = FocalLoss(alpha = 0.55, gamma = 7, dataset = 'ucsf')
ucsf_criterion_hiv = FocalLoss(alpha = 0.5, gamma = 2.2, dataset = 'ucsf')
adni_criterion_cd = FocalLoss(alpha = 0.75, gamma = 1, dataset = 'adni')
lab_criterion_hiv = FocalLoss(alpha = 0.5, gamma = 1, dataset = 'lab')

@torch.no_grad()
def test(feature_extractor, classifier, loader, dataset, criterions, epoch=None, fold=None, train=False):
    feature_extractor.eval()
    classifier.eval()

    xentropy_loss_avg = 0.
    correct = 0.
    total = 0.
    toprint = 0

    overall_accuracy = 0.

    feature_list = []

    classes = []
    criterion = None
    correlations = None
    dimensions = 0
    idx=None

    if dataset == 'ucsf':
        classes = ['CTRL', 'HIV', 'MCI', 'HAND']
        criterion = [criterions[0], criterions[1]]
        correlations = [torch.tensor(0.),  torch.tensor(0.)]
        dimensions = 2
    elif dataset == 'adni':
        classes = ['CTRL', 'MCI']
        criterion = [criterions[0]]
        correlations = [torch.tensor(0.)]
        dimensions = 1
        idx = 0
    else:
        classes = ['CTRL', 'HIV']
        criterion = [criterions[0]]
        correlations = [torch.tensor(0.)]
        dimensions = 1
        idx = 1

    accuracy_class = {}
    accuracy_dataset = 0
    for class_ in classes:
        accuracy_class[class_] = 0

    totals = list(np.zeros(dimensions*2))
    #total = 0
    #print(accuracy_class, totals)
    num_batches = 0
    for i, _ in enumerate(loader):
        num_batches += 1

    for i,(images, labels,actual_labels,datasets,ids, ages, genders) in enumerate(loader):
        #datasets = np.array(datasets)
        ids = np.array(ids)
        actual_labels = np.array(actual_labels)
        images = images.to(device).float()
        labels = labels.to(device).float()
        number_needed = None
        #print([np.shape(x) for x in (labels,actual_labels, datasets,ids,genders)])
        if i == num_batches - 1:
            #print('entered!')
            number_needed = int(args.batch_size) - len(images)
            images0, labels0, actual_labels0, datasets0, ids0, ages0, genders0 = first_batch_data
            images = torch.cat((images, images0[:number_needed,]),dim=0)
            labels = torch.cat((labels, labels0[:number_needed,]),dim=0)
            if dataset != 'ucsf':
                actual_labels = np.concatenate((actual_labels, actual_labels0[:number_needed,]),axis=0)
            else:
                actual_labels = np.concatenate((actual_labels, actual_labels0[:number_needed,]),axis=0)

            datasets = np.concatenate((datasets, datasets0[:number_needed]),axis=0)

            ids = np.concatenate((ids, ids0[:number_needed]),axis=0)
            ages = np.concatenate((ages, ages0[:number_needed]),axis=0)
            genders = np.concatenate((genders, genders0[:number_needed]),axis=0)

        data = (images, labels, actual_labels, datasets, ids, ages, genders)

        #print('test')
        cfs = get_cf_kernel_batch(data)
        classifier[2].cfs = cfs
        classifier[5].cfs = cfs
        classifier[7].cfs = cfs

        if i==0:
            first_batch_data = copy.deepcopy(data)

        feature = feature_extractor(images)
        pred = classifier(feature)
        # print('pred', pred.shape)
        # print('label',labels.shape)
        pred_ = []
        labels_ = []
        losses = []

        #

        #print(labels)

        for j in range(dimensions):
            #pred_.append(pred[:,j].unsqueeze(1))
            if dataset != 'ucsf':
                #print(type(labels))
                # labels_.append(labels[:,j].unsqueeze(1)[idx])
                # pred_.append(pred[:,j].unsqueeze(1)[idx])
                labels_ = labels[:,idx].unsqueeze(1)
                pred_ = pred[:,j].unsqueeze(1)
                #print(labels_[0])
                losses = [criterion[j](pred_,labels_).to(device)]
                pred_ = [pred_]
                labels_ = [labels_]
            else:
                labels_.append(labels[:,j].unsqueeze(1))
                pred_.append(pred[:,j].unsqueeze(1))
                losses.append(criterion[j](pred_[j],labels_[j]).to(device))

        #print(losses)
        # print('pred', pred_[0].shape)
        # print('label',labels_[0].shape)
        #print(np.shape(labels_[0]))
        #print(losses)
        xentropy_loss = np.sum(losses)
        xentropy_loss_avg += xentropy_loss.item()

        #print(images.size(0))

        truth = None
        hits = []
        for k in range(dimensions):
            #pred_axis = pred_[k].clone()
            pred_axis = pred_[k]
            pred_axis[pred_axis>0]=1
            pred_axis[pred_axis<0]=0

            hit = pred_axis == labels_[k]
            hits.append(hit)

            #just literally makes a tensor of length == pred that says True at every index for comparison purposes

            truth = torch.tensor([True]*len(hit)).cuda()
            truth = torch.unsqueeze(truth,1)
            if dimensions == 1:
                correct += ((hits[0]==truth)).sum().item()
            if dimensions == 2 and k == 1:
                #print(len(hits), len(hits[0]))
                correct += ((hits[0]==truth)&(hits[1]==truth)).sum().item()

        total += images.size(0)
        pred_cur = copy.deepcopy(pred)
        pred_copy = [copy.deepcopy(pred_)[idx] for idx in range(dimensions)]


        # remove duplicate test data
        if i == num_batches - 1:
            #print('entered 2')
            number_actual = int(args.batch_size) - number_needed
            pred_cur = pred_cur[:number_actual,]
            for i in range(dimensions):
                pred_[i] = pred_[i][:number_actual,]
                labels_[i] = labels_[i][:number_actual,]
            #datasets = datasets[:number_actual,]
            actual_labels = actual_labels[:number_actual,]
            ids= ids[:number_actual,]
            ages = ages[:number_actual,]
            genders = genders[:number_actual,]
            for i in range(dimensions):
                pred_copy[i] = pred_copy[i][:number_actual,]
            feature = feature[:number_actual,]

        #print(np.shape(actual_labels))
        #print(actual_labels)
    #    print(pred_)
        #print(len(pred_[0]))
        for j in range(0,len(pred_[0])):
            #total += 1
            actual_pred = None
            if dataset == 'ucsf':
                # if train == False:
                #
                #     row = {'epoch':epoch, 'id':ids[j], 'dataset':'UCSF', 'CD_pred':torch.sigmoid(pred_copy[0][j]).item(), 'HIV_pred':torch.sigmoid(pred_copy[1][j]).item(), 'fold': fold,'CD_label':labels_[0][j].item(), 'HIV_label':labels_[1][j].item()}
                #     csv_logger_pred.writerow(row)


                if pred_[0][j] == 0 and pred_[1][j] == 0 :
                    actual_pred = 0
                elif pred_[0][j] == 1 and pred_[1][j] == 0:
                    actual_pred = 1
                elif pred_[0][j] == 0 and pred_[1][j] == 1 :
                    actual_pred = 2
                elif pred_[0][j] == 1 and pred_[1][j] == 1 :
                    actual_pred = 3

                for i in range(4):
                    if actual_labels[j] == i:
                        totals[i] += 1
                        if actual_pred == i:
                            accuracy_class[classes[i]] += 1
                            accuracy_dataset += 1
            else:
                actual_pred = pred_[0][j]

                if actual_labels[j] == 0:
                    totals[0] += 1
                    if actual_pred == 0:
                        accuracy_class['CTRL'] += 1
                        accuracy_dataset += 1
                else:
                    if dataset == 'lab':
                        if actual_labels[j] == 2:
                            totals[1] += 1
                            if actual_pred == 1:
                                accuracy_class['HIV'] += 1
                                accuracy_dataset += 1

                    else:
                        if actual_labels[j] == 1:
                            totals[1] += 1
                            if actual_pred == 1:
                                accuracy_class['MCI'] += 1
                                accuracy_dataset += 1





    i = 0

    for key_ in accuracy_class:
        #print(accuracy_class[key_], totals[i])
        accuracy_class[key_] = round(accuracy_class[key_] / totals[i],3)
        i += 1

    #accuracy_dataset= round(accuracy_dataset / total,3)
    #print(correct, total)
    #print(round(accuracy_dataset/total,3))
    #print(accuracy_class, totals, total)
    overall_accuracy = (correct) / (total)
    overall_accuracy = round(overall_accuracy,3)


    xentropy_loss_avg = xentropy_loss_avg / (i + 1)

    return overall_accuracy, xentropy_loss_avg,accuracy_class, accuracy_dataset

def get_cf_kernel_batch(data):
    all_images, all_labels, all_actual_labels, all_datasets, all_ids, all_ages, all_genders = data
    dataset = all_datasets[0]


    label_hiv = []
    label_cd = []
    dataset_ = []

    ages = []
    gender_m = []
    N = all_images.shape[0]
    #print([len(x) for x in data])
    #print([np.shape(x) for x in data])
    #print(N)
    #print(all_labels, all_actual_labels, all_datasets, all_ids, all_ages, all_genders)
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

        ages.append(all_ages[j])
        cur_gender = all_genders[j]
        if cur_gender == 0:
            gender_m.append(1)
        elif cur_gender == 1:
            gender_m.append(0)

    cfs_batch = None

    if dataset=='ucsf':
        cfs_batch = np.zeros((N,5))
        cfs_batch[:,0] = label_hiv
        cfs_batch[:,1] = label_cd
        #cfs_batch[:,2] = dataset
        # cfs_batch[:,3] = dataset_adni
    #     cfs_batch[:,4] = dataset_ucsf
        cfs_batch[:,2] = np.ones((N,))
        cfs_batch[:,3] = ages
        cfs_batch[:,4] = gender_m
    elif dataset=='lab':
        cfs_batch = np.zeros((N,4))
        cfs_batch[:,0] = label_hiv
        #cfs_batch[:,2] = dataset
        # cfs_batch[:,3] = dataset_adni
    #     cfs_batch[:,4] = dataset_ucsf
        cfs_batch[:,1] = np.ones((N,))
        cfs_batch[:,2] = ages
        cfs_batch[:,3] = gender_m
    else:
        cfs_batch = np.zeros((N,4))
        cfs_batch[:,0] = label_cd
        #cfs_batch[:,2] = dataset
        # cfs_batch[:,3] = dataset_adni
    #     cfs_batch[:,4] = dataset_ucsf
        cfs_batch[:,1] = np.ones((N,))
        cfs_batch[:,2] = ages
        cfs_batch[:,3] = gender_m


    cfs = nn.Parameter(torch.Tensor(cfs_batch).to(device).float(), requires_grad=False)


    return cfs

def get_cf_kernel(loader):
    label_hiv = []
    label_cd = []
    dataset = []
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
            #
            if datasets=='ucsf':
                dataset.append(1)

            elif datasets=='lab':
                dataset.append(1)

            elif datasets=='adni':
                dataset.append(1)


            ages.append(all_ages[j])
            cur_gender = all_genders[j]
            if cur_gender == 0:
                gender_m.append(1)
            elif cur_gender == 1:
                gender_m.append(0)

    N = len(dataset)
    #print(N)
    # print(len(label_hiv), len(label_cd), len(ages), len(gender_m))
    X_shuffled = None
    if datasets=='ucsf':
        X_shuffled = np.zeros((N,5))
        X_shuffled[:,0] = label_hiv
        X_shuffled[:,1] = label_cd
        # X_shuffled[:,2] = dataset
        # X_shuffled[:,3] = dataset_adni
    #     X_shuffled[:,4] = dataset_ucsf
        X_shuffled[:,2] = np.ones((N,))
        X_shuffled[:,3] = ages
        X_shuffled[:,4] = gender_m
    elif datasets=='lab':
        X_shuffled = np.zeros((N,4))
        X_shuffled[:,0] = label_hiv
        # X_shuffled[:,2] = dataset
        # X_shuffled[:,3] = dataset_adni
    #     X_shuffled[:,4] = dataset_ucsf
        X_shuffled[:,1] = np.ones((N,))
        X_shuffled[:,2] = ages
        X_shuffled[:,3] = gender_m
    else:
        X_shuffled = np.zeros((N,4))
        X_shuffled[:,0] = label_cd
        # X_shuffled[:,2] = dataset
        # X_shuffled[:,3] = dataset_adni
    #     X_shuffled[:,4] = dataset_ucsf
        X_shuffled[:,1] = np.ones((N,))
        X_shuffled[:,2] = ages
        X_shuffled[:,3] = gender_m

    cf_kernel = nn.Parameter(torch.tensor(np.linalg.inv(np.transpose(X_shuffled).dot(X_shuffled))).float().to(device),  requires_grad=False)

    return cf_kernel


#trains each classifier (ucsf, adni, lab) on its corresponding dataset (sri, adni, lab)
def train(feature_extractor, classifier, train_loader, test_loader, dataset, criterions, epochs=args.epochs, train=False):
    feature_extractor.zero_grad()
    classifier.zero_grad()

    fe_optimizer = optim.AdamW(feature_extractor.parameters(), lr =args.lr, weight_decay=0.01) # used to be 0.01
    classifier_optimizer = optim.AdamW(classifier.parameters(), lr =args.lr, weight_decay=args.wd) # used to be args.wd

    best_accuracy = 0
    best_epoch = 0
    epochs = args.epochs
    ave_valid_acc_50 = 0.0
    counter = 0.0

    classes = []
    criterion = None
    dimensions = 0
    idx = None

    if dataset == 'ucsf':
        classes = ['CTRL', 'HIV', 'MCI', 'HAND']
        criterion = [criterions[0], criterions[1]]
        dimensions = 2
    elif dataset == 'adni':
        classes = ['CTRL', 'MCI']
        criterion = [criterions[0]]
        dimensions = 1
        idx = 0
    else:
        classes = ['CTRL', 'HIV']
        criterion = [criterions[0]]
        dimensions = 1
        idx = 1

    for epoch in range(epochs):

        feature_extractor.train()
        classifier.train()

        progress_total = 0
        num_samples = []
        for i, batch in enumerate(train_loader):
            progress_total += 1
            num_samples.append(len(batch[0]))
        #print(progress_total)

        progress_bar = tqdm(train_loader, total = progress_total)
        xentropy_loss_avg = 0.
        cur_loss_sum = 0
        correct = 0.
        total = 0.
        total = 0.
        overall_accuracy = 0

        for i, (images, labels, actual_labels, datasets, ids, ages, genders) in enumerate(progress_bar):
            #print(actual_labels)
            feature_extractor.zero_grad()
            classifier.zero_grad()

            data = (images, labels, actual_labels, datasets, ids, ages, genders)
        #    print(data[1:])
            cfs = get_cf_kernel_batch(data)
            #print(datasets[0], np.shape(actual_labels))
            classifier[2].cfs = cfs
            classifier[5].cfs = cfs
            classifier[7].cfs = cfs

            datasets = np.array(datasets)

            progress_bar.set_description('Epoch ' + str(epoch))
            images = images.to(device).float()
            labels = labels.to(device).float()

            feature = feature_extractor(images)
            pred = classifier(feature)
            pred_ = []
            labels_ = []
            losses = []

            for j in range(dimensions):
                #pred_.append(pred[:,j].unsqueeze(1))
                if dataset != 'ucsf':
                    labels_ = labels[:,idx].unsqueeze(1)
                    pred_ = pred[:,j].unsqueeze(1)
                    #print(labels_[0])
                    losses = [criterion[j](pred_,labels_).to(device)]
                    pred_ = [pred_]
                    labels_ = [labels_]
                else:
                    labels_.append(labels[:,j].unsqueeze(1))
                    pred_.append(pred[:,j].unsqueeze(1))
                    losses.append(criterion[j](pred_[j],labels_[j]).to(device))


            #     #pred_.append(pred[:,i].unsqueeze(1))
            #     if dataset != 'ucsf':
            #         labels_.append(labels[:,i].unsqueeze(1)[idx])
            #         pred_.append(pred[:,i].unsqueeze(1)[idx])
            #     else:
            #         labels_.append(labels[:,i].unsqueeze(1))
            #         pred_.append(pred[:,i].unsqueeze(1))
            #     losses.append(criterion[i](pred_[i],labels_[i]).to(device))

            #print(np.shape(labels_[0]))
            xentropy_loss = np.sum(losses)

            xentropy_loss.backward()

            fe_optimizer.step()
            classifier_optimizer.step()

            truth = None
            hits = []
            for k in range(dimensions):
                pred_axis = pred_[k]
                pred_axis[pred_axis>0]=1
                pred_axis[pred_axis<0]=0

                hit = pred_axis == labels_[k]
                hits.append(hit)
                #print(np.array(hits).shape)
                #just literally makes a tensor of length == pred that says True at every index for comparison purposes

                truth = torch.tensor([True]*len(hit)).cuda()
                truth = torch.unsqueeze(truth,1)
                if dimensions == 1:
                    correct += ((hits[0]==truth)).sum().item()
                if dimensions == 2 and k == 1:
                    correct += ((hits[0]==truth)&(hits[1]==truth)).sum().item()

            total += images.size(0)

            overall_accuracy= correct/total


            xentropy_loss_avg += xentropy_loss.item()
            print(overall_accuracy)

            progress_bar.set_postfix(
                loss='%.6f' % (xentropy_loss_avg / (i + 1)),
                acc='%.2f' % overall_accuracy)

            #test
        test_acc, test_loss,test_accuracy_class, test_accuracy_dataset =test(feature_extractor, classifier, test_loader, dataset, criterions, epoch)

        row = {'epoch': epoch, 'dataset': dataset, 'train_acc': round(overall_accuracy,3), 'test_acc': test_acc, 'train_loss':round((xentropy_loss_avg / (i + 1)),3),  'test_loss': round(test_loss,3)}
        row['test_ctrl_acc'] = test_accuracy_class['CTRL']
        if dataset == 'ucsf':
            row['test_mci_acc'] = test_accuracy_class['MCI']
            row['test_hiv_acc'] = test_accuracy_class['HIV']
            row['test_hand_acc'] = test_accuracy_class['HAND']
        elif dataset == 'lab':
            row['test_mci_acc'] = 'n/a'
            row['test_hiv_acc'] = test_accuracy_class['HIV']
            row['test_hand_acc'] = 'n/a'
        else:
            row['test_mci_acc'] = test_accuracy_class['MCI']
            row['test_hiv_acc'] = 'n/a'
            row['test_hand_acc'] = 'n/a'
        train_acc, train_loss,train_accuracy_class, train_accuracy_dataset = test(feature_extractor, classifier, train_loader, dataset, criterions, epoch, train =True)
        row['train_ctrl_acc'] = train_accuracy_class['CTRL']
        if dataset == 'ucsf':
            row['train_mci_acc'] = train_accuracy_class['MCI']
            row['train_hiv_acc'] = train_accuracy_class['HIV']
            row['train_hand_acc'] = train_accuracy_class['HAND']
        elif dataset == 'lab':
            row['train_mci_acc'] = 'n/a'
            row['train_hiv_acc'] = train_accuracy_class['HIV']
            row['train_hand_acc'] = 'n/a'
        else:
            row['train_mci_acc'] = train_accuracy_class['MCI']
            row['train_hiv_acc'] = 'n/a'
            row['train_hand_acc'] = 'n/a'

        print(row)
        csv_logger.writerow(row)
        if test_acc > best_accuracy:
            best_accuracy = test_acc
            best_epoch = epoch

    best_models = [feature_extractor, classifier]

    return test_acc, test_accuracy_class, test_accuracy_dataset, best_accuracy, best_epoch, best_models

if __name__ == '__main__':
    log_path = '/scratch/users/jmanasse/mri_proj/logs/'
    filename = log_path + args.name +'.csv'
    os.makedirs(log_path, exist_ok=True)
    filename2 = log_path + 'predictions/'+ args.name +'.csv'
    os.makedirs(log_path + 'predictions/', exist_ok=True)
    csv_logger = CSVLogger( args, fieldnames=['epoch', 'dataset', 'train_acc',  'test_acc',
                                              'train_loss','test_loss', 'test_ctrl_acc', 'test_mci_acc',
                                              'test_hiv_acc', 'test_hand_acc', 'train_ctrl_acc','train_mci_acc',
                                              'train_hiv_acc','train_hand_acc'],
                           filename=filename)
    csv_logger_pred = CSVLogger( args, fieldnames=['epoch', 'id', 'dataset', 'CD_pred', 'HIV_pred', 'fold','CD_label', 'HIV_label'], filename=filename2)
    # filename_ids = log_path  +'ids.csv'
    filename_sets = log_path  +'sets.csv'
    #csv_logger_sets = CSVLogger( args, fieldnames=['ids', 'datasets'],filename = filename_sets)
    # csv_logger_ids = CSVLogger( args, fieldnames=['ids', 'datasets','mci','hiv'],filename = filename_ids)


    u_best_accuracy_list = [0,0,0,0,0]
    u_best_epoch_list = [0,0,0,0,0]
    u_final_accuracy_list = [0,0,0,0,0]
    u_ave_valid_acc_50_list=[0,0,0,0,0]

    a_best_accuracy_list = [0,0,0,0,0]
    a_best_epoch_list = [0,0,0,0,0]
    a_final_accuracy_list = [0,0,0,0,0]
    a_ave_valid_acc_50_list=[0,0,0,0,0]

    l_best_accuracy_list = [0,0,0,0,0]
    l_best_epoch_list = [0,0,0,0,0]
    l_final_accuracy_list = [0,0,0,0,0]
    l_ave_valid_acc_50_list=[0,0,0,0,0]
    l_best_model_dict = {}

    acc_each_class_list = []
    acc_each_dataset_list = []

    for fold in range(5):
        # row = {'epoch': 'fold', 'train_acc': str(fold)}
        # csv_logger.writerow(row)
        transformation = super_transformation()
        test_data = MRI_Dataset(fold = fold , stage= 'original_test')
        train_data = MRI_Dataset(fold = fold , stage= 'original_train',transform = transformation)
        train_loader = DataLoader(dataset=train_data,
                                  batch_size=None,
                                  # sampler=MixedSampler(dataset=train_data,
                                  #                      batch_size=args.batch_size),
                                  shuffle=False,
                                  pin_memory=True,
                                  num_workers=3)
        test_loader = DataLoader(dataset=test_data ,
                                      batch_size=None,# to include all test images
                                      shuffle=True,
                                      pin_memory=True,
                                      num_workers=3)
        ucsf_data = []
        adni_data = []
        lab_data = []
        id_tracker = []
        for i,(images, labels,actual_labels,datasets,ids, ages, genders) in enumerate(train_loader):
            #csv_logger_sets.writerow({'ids':ids, 'datasets':'train'})
            #id_tracker.append(ids)
            if datasets == 'ucsf':
                ucsf_data.append((images, datasets, ids, actual_labels, labels, ages, genders))
                #csv_logger_ids.writerow({'ids':ids, 'datasets':datasets, 'mci':int(labels[0]), 'hiv': int(labels[1])})
            elif datasets == 'adni':
                adni_data.append((images, datasets, ids, actual_labels, labels, ages, genders))
                #csv_logger_ids.writerow({'ids':ids, 'datasets':datasets, 'mci':int(labels[0]), 'hiv': 'n/a'})
            else:
                lab_data.append((images, datasets, ids, actual_labels, labels,ages, genders))
                #csv_logger_ids.writerow({'ids':ids, 'datasets':datasets, 'mci':'n/a', 'hiv': int(labels[1])})

        ucsf_test_data = []
        adni_test_data = []
        lab_test_data = []
        for i,(images, labels,actual_labels,datasets,ids, ages, genders) in enumerate(test_loader):
            #csv_logger_sets.writerow({'ids':ids, 'datasets':'test'})
            if datasets == 'ucsf':
            #     if ids in id_tracker:
            #         ucsf_data.append((images, datasets, ids, actual_labels, labels, ages, genders))
            #     else:
                ucsf_test_data.append((images, datasets, ids, actual_labels, labels, ages, genders))
            elif datasets == 'adni':
                # if ids in id_tracker:
                #     adni_data.append((images, datasets, ids, actual_labels, labels, ages, genders,))
                # else:
                adni_test_data.append((images, datasets, ids, actual_labels, labels, ages, genders,))
            else:
                # if ids in id_tracker:
                #     lab_data.append((images, datasets, ids, actual_labels, labels, ages, genders))
                # else:
                lab_test_data.append((images, datasets, ids, actual_labels, labels, ages, genders))

        ucsf_dataset = Specific_MRI_Dataset(ucsf_data)
        adni_dataset = Specific_MRI_Dataset(adni_data)
        lab_dataset = Specific_MRI_Dataset(lab_data)
        #
        ucsf_loader = DataLoader(dataset=ucsf_dataset,
                                  batch_size=args.batch_size,
                                  sampler=ClassMixedSampler(dataset=ucsf_dataset,
                                                       batch_size=args.batch_size),
                                  #shuffle=True,
                                  pin_memory=True,
                                  num_workers=3)

        adni_loader = DataLoader(dataset=adni_dataset,
                                  batch_size=args.batch_size,
                                  sampler=ClassMixedSampler(dataset=adni_dataset,
                                                       batch_size=args.batch_size),
                                  #shuffle=True,
                                  pin_memory=True,
                                  num_workers=3)
        lab_loader = DataLoader(dataset=lab_dataset,
                                  batch_size=args.batch_size,
                                  sampler=ClassMixedSampler(dataset=lab_dataset,
                                                       batch_size=args.batch_size),
                                  #shuffle=True,
                                  pin_memory=True,
                                  num_workers=3)



        ucsf_test_dataset = Specific_MRI_Dataset(ucsf_test_data)
        adni_test_dataset = Specific_MRI_Dataset(adni_test_data)
        lab_test_dataset = Specific_MRI_Dataset(lab_test_data)
        # print([x[1] for x in ucsf_data])
        # print([x[1] for x in adni_data])
        # print([x[1] for x in lab_data])
        # print([x[1] for x in ucsf_test_data])
        # print([x[1] for x in adni_test_data])
        # print([x[1] for x in lab_test_data])

        ucsf_test_loader = DataLoader(dataset=ucsf_test_dataset ,
                                      batch_size=args.batch_size,# to include all test images
                                      shuffle=True,
                                      pin_memory=True,
                                      num_workers=3)
        adni_test_loader = DataLoader(dataset=adni_test_dataset ,
                                      batch_size=args.batch_size,# to include all test images
                                      shuffle=True,
                                      pin_memory=True,
                                      num_workers=3)
        lab_test_loader = DataLoader(dataset=lab_test_dataset ,
                                      batch_size=args.batch_size,# to include all test images
                                      shuffle=True,
                                      pin_memory=True,
                                      num_workers=3)

        ucsf_kernel = get_cf_kernel(ucsf_loader)
        adni_kernel = get_cf_kernel(adni_loader)
        lab_kernel = get_cf_kernel(lab_loader)

        feature_extractor = fe(lab_kernel, trainset_size = len(train_data), in_num_ch=1, img_size=(64, 64, 64), inter_num_ch=16,
                           fc_num_ch=16, kernel_size=3, conv_act='relu',
                           fe_arch=args.fe_arch, dropout=args.dropout,
                           fc_dropout = args.fc_dropout, batch_size = args.batch_size).to(device)


        # l_feature_extractor = fe(lab_kernel, trainset_size = len(train_data), in_num_ch=1, img_size=(64, 64, 64), inter_num_ch=16,
        #                    fc_num_ch=16, kernel_size=3, conv_act='relu',
        #                    fe_arch=args.fe_arch, dropout=args.dropout,
        #                    fc_dropout = args.fc_dropout, batch_size = args.batch_size).to(device)

        # u_feature_extractor = fe(ucsf_kernel, trainset_size = len(train_data), in_num_ch=1, img_size=(64, 64, 64), inter_num_ch=16,
        #                    fc_num_ch=16, kernel_size=3, conv_act='relu',
        #                    fe_arch=args.fe_arch, dropout=args.dropout,
        #                    fc_dropout = args.fc_dropout, batch_size = args.batch_size).to(device)
        # a_feature_extractor = fe(adni_kernel, trainset_size = len(train_data), in_num_ch=1, img_size=(64, 64, 64), inter_num_ch=16,
        #                    fc_num_ch=16, kernel_size=3, conv_act='relu',
        #                    fe_arch=args.fe_arch, dropout=args.dropout,
        #                    fc_dropout = args.fc_dropout, batch_size = args.batch_size).to(device)

        classifier_ucsf = nn.Sequential(
#                     MetadataNorm(batch_size=args.batch_size, cf_kernel=cf_kernel, num_features = 2048, trainset_size = len(train_data)),
                    nn.Linear(2048, 128),
                    nn.ReLU(),
#                     nn.BatchNorm1d(128),
                    MetadataNorm(batch_size=args.batch_size, cf_kernel=ucsf_kernel, num_features = 128, trainset_size = len(ucsf_dataset)),
                    nn.Linear(128, 16),
                    nn.ReLU(),
#                     nn.BatchNorm1d(16),
                   MetadataNorm(batch_size=args.batch_size, cf_kernel=ucsf_kernel, num_features = 16, trainset_size = len(ucsf_dataset)),
                    nn.Linear(16, 2),
                    MetadataNorm(batch_size=args.batch_size, cf_kernel=ucsf_kernel, num_features = 2, trainset_size = len(ucsf_dataset)),
                ).to(device)
        classifier_adni = nn.Sequential(
#                     MetadataNorm(batch_size=args.batch_size, cf_kernel=cf_kernel, num_features = 2048, trainset_size = len(train_data)),
                    nn.Linear(2048, 128),
                    nn.ReLU(),
#                     nn.BatchNorm1d(128),
                    MetadataNorm(batch_size=args.batch_size, cf_kernel=adni_kernel, num_features = 128, trainset_size = len(adni_dataset)),
                    nn.Linear(128, 16),
                    nn.ReLU(),
#                     nn.BatchNorm1d(16),
                   MetadataNorm(batch_size=args.batch_size, cf_kernel=adni_kernel, num_features = 16, trainset_size = len(adni_dataset)),
                    nn.Linear(16, 1),
                    MetadataNorm(batch_size=args.batch_size, cf_kernel=adni_kernel, num_features = 1, trainset_size = len(adni_dataset)),
                ).to(device)
        classifier_lab = nn.Sequential(
#                     MetadataNorm(batch_size=args.batch_size, cf_kernel=cf_kernel, num_features = 2048, trainset_size = len(train_data)),
                    nn.Linear(2048, 128),
                    nn.ReLU(),
#                     nn.BatchNorm1d(128),
                    MetadataNorm(batch_size=args.batch_size, cf_kernel=lab_kernel, num_features = 128, trainset_size = len(lab_dataset)),
                    nn.Linear(128, 16),
                    nn.ReLU(),
#                     nn.BatchNorm1d(16),
                   MetadataNorm(batch_size=args.batch_size, cf_kernel=lab_kernel, num_features = 16, trainset_size = len(lab_dataset)),
                    nn.Linear(16, 1),
                    MetadataNorm(batch_size=args.batch_size, cf_kernel=lab_kernel, num_features = 1, trainset_size = len(lab_dataset)),
                ).to(device)


        ucsf_test_acc, ucsf_test_accuracy_class, ucsf_test_accuracy_dataset, ucsf_best_accuracy, ucsf_best_epoch, ucsf_best_models = train(feature_extractor, classifier_ucsf, ucsf_loader, ucsf_test_loader, 'ucsf', [ucsf_criterion_cd, ucsf_criterion_hiv])
        feature_extractor = ucsf_best_models[0]
        adni_test_acc, adni_test_accuracy_class, adni_test_accuracy_dataset, adni_best_accuracy, adni_best_epoch, adni_best_models = train(feature_extractor, classifier_adni, adni_loader, adni_test_loader, 'adni', [adni_criterion_cd])
        feature_extractor = adni_best_models[0]
        lab_test_acc, lab_test_accuracy_class, lab_test_accuracy_dataset, lab_best_accuracy, lab_best_epoch, lab_best_models = train(feature_extractor, classifier_lab, lab_loader, lab_test_loader, 'lab', [lab_criterion_hiv])

        feature_extractor = lab_best_models[0]
        classifer_ucsf = ucsf_best_models[1]
        classifer_adni = adni_best_models[1]
        classifer_lab = lab_best_models[1]

        ucsf_test_acc, ucsf_test_loss, ucsf_test_accuracy_class, ucsf_test_accuracy_dataset = test(feature_extractor, classifier_ucsf, ucsf_test_loader, 'ucsf', [ucsf_criterion_cd, ucsf_criterion_hiv],fold = fold,train=True)
        adni_test_acc, adni_test_loss, adni_test_accuracy_class, adni_test_accuracy_dataset= test(feature_extractor, classifier_adni, adni_test_loader, 'adni', [adni_criterion_cd], fold = fold,train=True)
        lab_test_acc, lab_test_loss, lab_test_accuracy_class, lab_test_accuracy_dataset  = test(feature_extractor, classifier_lab, lab_test_loader, 'lab', [lab_criterion_hiv], fold = fold,train=True)
        #
    csv_logger_pred.close()
        # csv_logger_ids.close()

        # feature_extractor, classifier_ucsf = ucsf_best_models
        # feature_extractor, classifier_adni = adni_best_models
        # feature_extractor, classifier_lab = lab_best_models
        #
        # u_best_accuracy_list[fold] = ucsf_best_accuracy
        # a_best_accuracy_list[fold] = adni_best_accuracy
        # l_best_accuracy_list[fold] = lab_best_accuracy
        #
        # u_final_accuracy_list[fold] = ucsf_test_acc
        # a_final_accuracy_list[fold] = adni_test_acc
        # l_final_accuracy_list[fold] = lab_test_acc
        #
        # u_best_epoch_list[fold] = ucsf_best_epoch
        # a_best_epoch_list[fold] = adni_best_epoch
        # l_best_epoch_list[fold] = lab_best_epoch
        #

        # # acc_each_class_list.append(test_accuracy_class)
        # # acc_each_dataset_list.append(test_accuracy_dataset)
        #
        # print('best_accuracy', ucsf_best_accuracy)
        # print('final_accuracy',ucsf_test_acc)
        # print('best_epoch', ucsf_best_epoch)
        #
        # print('best_accuracy', adni_best_accuracy)
        # print('final_accuracy',adni_test_acc)
        # print('best_epoch', adni_best_epoch)
        #
        # print('best_accuracy', lab_best_accuracy)
        # print('final_accuracy',lab_test_acc)
        # print('best_epoch', lab_best_epoch)
