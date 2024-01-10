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
from sampler import PairedSamplerNPZ
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
parser.add_argument('--name', type=str, default='debug_age',
                    help='name of this run')
parser.add_argument('--fe_arch', type=str, default='age',
                    help='FeatureExtractor')
parser.add_argument('--dropout', type=float, default=0.1,
                    help='dropout in conv3d')
parser.add_argument('--fc_dropout', type=float, default=0.1,
                    help='dropout for fc')
parser.add_argument('--wd', type=float, default=0.01,
                    help='weight decay for adam')
parser.add_argument('--lamb', type=float, default=0.5,
                    help='weight for disentanglement loss')
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

wandb.init(project="final npz", entity="jmanasse", config = {
  "learning_rate": args.lr,
  "epochs": args.epochs,
  "batch_size": args.batch_size,
  "fc dropout": args.fc_dropout,
  'dropout': args.dropout,
  "weight decay": args.wd,
  "lambda": args.lamb,
  'loss type': 'npz loss'
})

print("device:",device)

class XLoss(nn.Module):
    def __init__(self, alpha=0.1, gamma=2, dataset = 'ucsf'):
        super(XLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
       # self.criterion = torch.nn.BCEWithLogitsLoss(reduction= 'none').to(device)
        self.criterion = torch.nn.BCEWithLogitsLoss().to(device)
        self.dataset = dataset

    def forward(self, inputs, targets):
        BCE_loss = self.criterion(inputs, targets)
        return BCE_loss

class NPZLoss(nn.Module):
    def __init__(self):
        super(NPZLoss, self).__init__()
        self.criterion = torch.nn.BCEWithLogitsLoss(reduction= 'none').to(device)

    def forward(self, pred_cd1, labels_cd, pred_hiv1, labels_hiv, emb1, emb2, npz1, npz2, tau, downgraded=False):
        #compute BCE loss
        losscd = self.criterion(pred_cd1, labels_cd).to(device)
        losshiv = self.criterion(pred_hiv1, labels_hiv).to(device)

        loss = torch.cat((losscd.unsqueeze(0), losshiv.unsqueeze(0)))
        xentropy_loss = torch.nansum(loss).mean()

        #normalize tau as unit vector within loss computation
        tau = tau/float(torch.norm(tau))

        #compute disentanglement loss wrt to age
        proj_e1_len = torch.sum(emb1 * tau.repeat(emb1.shape[0], 1), dim=1) 
        proj_e2_len = torch.sum(emb2 * tau.repeat(emb2.shape[0], 1), dim=1) # dot-product
        # proj_e1_len = torch.norm((torch.dot(emb1.squeeze(), tau.squeeze()) / torch.dot(tau.squeeze(), tau.squeeze())) * tau.squeeze())
        # proj_e2_len = torch.norm((torch.dot(emb2.squeeze(), tau.squeeze()) / torch.dot(tau.squeeze(), tau.squeeze())) * tau.squeeze())
        if npz1 == 99.0 or math.isnan(npz1) or npz2 == 99.0 or math.isnan(npz2):  
            return (xentropy_loss.to(device), torch.tensor(0.).to(device)) 
        elif npz2>npz1:
            npz_diff = npz2 - npz1
            emb_len_diff = proj_e2_len - proj_e1_len
            #make this abs value
            npz_loss = torch.sum(torch.abs(emb_len_diff-npz_diff))

            return (xentropy_loss.to(device), npz_loss.to(device))
        else:
            npz_diff = npz1 - npz2
            emb_len_diff = proj_e1_len - proj_e2_len
            #make this abs value!
            npz_loss = torch.sum(torch.abs(emb_len_diff-npz_diff))

            return (xentropy_loss.to(device), npz_loss.to(device))

ucsf_criterion = NPZLoss()
ucsf_test_criterion = XLoss()

@torch.no_grad()
def test(feature_extractor, classifier_ucsf, data,  ucsf_test_criterion, fold=None, epoch=None, train = False):
    feature_extractor.eval()
    classifier_ucsf.eval()

    loader = DataLoader(dataset=data,
                        batch_size=1,# to include all test images
                        shuffle=True,
                        pin_memory=True,
                        num_workers=3)

    xentropy_loss_avg = 0.
    npz_loss_avg = 0.
    total_loss_avg = 0.
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
    all_npzs = []
    all_label_cd = []
    all_label_hiv = []

    num_batches = 0
    for i, _ in enumerate(loader):
        num_batches += 1
    
    for i,(images, labels,actual_labels,ids, ages, genders, npzs) in enumerate(loader):
        #datasets = np.array(datasets)
        ids = np.array(ids)
        actual_labels = np.array(actual_labels)
        images = images.to(device).float()
        labels = labels.to(device).float()

        data = (images, labels, actual_labels, ids, ages, genders, npzs)

        feature = feature_extractor(images)
        pred = classifier_ucsf(feature)
        pred_cd = pred[:,0]
        pred_cd = torch.unsqueeze(pred_cd,1)
        pred_hiv = pred[:,1]
        pred_hiv = torch.unsqueeze(pred_hiv,1)

        # BELOW (TO REST OF FUNC) IS just metrics essentially
        labels_cd = labels[:,0]
        labels_cd = torch.unsqueeze(labels_cd,1)
        labels_hiv = labels[:,1]
        labels_hiv = torch.unsqueeze(labels_hiv,1)
        
        xentropy_loss_cd = ucsf_test_criterion(pred_cd, labels_cd).to(device)
        xentropy_loss_hiv = ucsf_test_criterion(pred_hiv, labels_hiv).to(device)
        xentropy_loss = xentropy_loss_cd + xentropy_loss_hiv


        xentropy_loss_avg += xentropy_loss.item()
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
        total += 1




        feature_list.append(feature.cpu())
        #all_datasets = np.append(all_datasets,datasets)
        all_ids = np.append(all_ids, ids)
        all_preds.extend(pred_cur.detach().cpu().numpy())
        all_genders.extend(genders)
        all_npzs.extend(npzs)
        all_ages.extend(ages)
        all_label_cd.extend(labels_cd.detach().cpu().numpy())
        all_label_hiv.extend(labels_hiv.detach().cpu().numpy())

        # ucsf
        ucsf_pred_cd = pred_cd#[datasets=='ucsf']
        ucsf_pred_hiv = pred_hiv#[datasets=='ucsf']
        ucsf_pred_cd_copy = pred_cd_copy#[datasets=='ucsf']
        ucsf_pred_hiv_copy = pred_hiv_copy#[datasets=='ucsf']
        ucsf_labels_cd = labels_cd#[datasets=='ucsf']
        ucsf_labels_hiv = labels_hiv#[datasets=='ucsf']
        ucsf_actual_labels = actual_labels#[datasets=='ucsf']
        ucsf_ids = ids#[datasets=='ucsf']
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

    accuracy_class['ucsf']['CTRL'] = round(accuracy_class['ucsf']['CTRL'] / total_0_ucsf,3)
    accuracy_class['ucsf']['MCI'] = round(accuracy_class['ucsf']['MCI'] / total_1_ucsf,3)
    accuracy_class['ucsf']['HIV'] = round(accuracy_class['ucsf']['HIV'] / total_2_ucsf,3)
    accuracy_class['ucsf']['MND']= round(accuracy_class['ucsf']['MND'] / total_3_ucsf,3)
    

    accuracy_dataset['ucsf'] = round(accuracy_dataset['ucsf'] / total_ucsf,3)
    
    print(accuracy_class, total_ucsf)
    overall_accuracy = (correct) / (total)
    overall_accuracy = round(overall_accuracy,3)


    xentropy_loss_avg = xentropy_loss_avg / (i + 1)

    return overall_accuracy, xentropy_loss_avg,accuracy_class, accuracy_dataset


def train(feature_extractor,  classifier_ucsf, train_data, test_data, fold ):

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
    
    train_loader = DataLoader(dataset=train_data,
                              batch_size=args.batch_size,
                              sampler=PairedSamplerNPZ(dataset=train_data,
                                                   batch_size=args.batch_size),
                              shuffle=False,
                              pin_memory=True,
                              num_workers=3)

    ucsf_criterion = NPZLoss()

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
        npz_loss_avg = 0.
        total_loss_avg = 0.
        cur_loss_sum = 0
        correct = 0.
        total = 0.
        ucsf_correct = 0.
        ucsf_total = 0.
        total = 0.
        overall_accuracy = 0
        # batch accumulation parameter
        accum_iter = args.batch_size//2
        ###### "Training happens here! ######

        for i, (images, labels, actual_labels, ids, ages, genders,npzs) in enumerate(progress_bar):
            progress_bar.set_description('Epoch ' + str(epoch))

            for j in range(num_samples[i]//2):
                #get images in pair
                images1 = images[0][j].to(device).float()
                images2 = images[1][j].to(device).float()

                #get ages of pair
                npz1 = npzs[0][j].to(device).float()
                npz2 = npzs[1][j].to(device).float()

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

                xentropy_loss, npz_loss = ucsf_criterion(pred_cd1, labels_cd, pred_hiv1, labels_hiv, feature1, feature2, npz1, npz2, feature_extractor.get_parameter('feature_extractor.tau'))

                total_loss = xentropy_loss + npz_loss*args.lamb

                # normalize loss to account for batch accumulation
                xentropy_loss = xentropy_loss/accum_iter
                npz_loss = npz_loss/accum_iter
                total_loss = total_loss/accum_iter

                total_loss.backward()

                xentropy_loss_avg += xentropy_loss.item()
                npz_loss_avg += npz_loss.item()
                total_loss_avg += total_loss.item()

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
        test_acc, test_xentropy_loss,test_accuracy_class, test_accuracy_dataset = test(feature_extractor, classifier_ucsf, test_data, ucsf_test_criterion, fold =fold, epoch = epoch)

        test_ucsf_ctrl = test_accuracy_class['ucsf']['CTRL']
        test_ucsf_mci = test_accuracy_class['ucsf']['MCI']
        test_ucsf_hiv = test_accuracy_class['ucsf']['HIV']
        test_ucsf_mnd = test_accuracy_class['ucsf']['MND']

        ucsf_test_acc = np.mean([test_ucsf_ctrl, test_ucsf_mci, test_ucsf_hiv, test_ucsf_mnd])

        test_accuracy_dataset['ucsf'] = round(ucsf_test_acc,3)

        print('test:',test_accuracy_class['ucsf'])

        # this trainning accuracy has augmentation in it!!!!!
        # some images are sampled more than once!!!!
        train_acc, train_xentropy_loss, train_accuracy_class, train_accuracy_dataset  = test(feature_extractor, classifier_ucsf, train_data, ucsf_test_criterion, fold =fold, epoch = epoch, train =True)

        train_ucsf_ctrl = train_accuracy_class['ucsf']['CTRL']
        train_ucsf_mci = train_accuracy_class['ucsf']['MCI']
        train_ucsf_hiv = train_accuracy_class['ucsf']['HIV']
        train_ucsf_mnd = train_accuracy_class['ucsf']['MND']

        # redefine ucsf_train_acc, lab_val_acc to be the average of all classes
        ucsf_train_acc = np.mean([train_ucsf_ctrl, train_ucsf_mci, train_ucsf_hiv, train_ucsf_mnd])

        train_accuracy_dataset['ucsf'] = round(ucsf_train_acc,3)

        tqdm.write('train_acc: %.2f u_train_acc: %.2f' % (overall_accuracy, ucsf_train_acc))
        tqdm.write('test_acc: %.2f u_test_acc: %.2f test_x_loss: %.2f' % (test_acc, ucsf_test_acc, test_xentropy_loss))

        # row = {'epoch': epoch, 'train_acc': round(overall_accuracy,3), 'test_acc': test_acc, 'train_loss':round((xentropy_loss_avg / (i + 1)),3),  'test_loss': round(test_loss,3),
        #        'ucsf_train_acc': ucsf_train_acc,
        #        'ucsf_test_acc': ucsf_test_acc,
        #        'train_ucsf_ctrl':train_ucsf_ctrl, 'train_ucsf_mci':train_ucsf_mci,
        #        'train_ucsf_hiv':train_ucsf_hiv, 'train_ucsf_mnd':train_ucsf_mnd,

        #        'test_ucsf_ctrl':test_ucsf_ctrl, 'test_ucsf_mci':test_ucsf_mci,
        #        'test_ucsf_hiv':test_ucsf_hiv, 'test_ucsf_mnd':test_ucsf_mnd}
        # csv_logger.writerow(row)

        wandb.log({"xentropy loss": round((xentropy_loss_avg / (i + 1)),3),
                   "age loss": round((npz_loss_avg / (i + 1)),3),
                   "total loss": round((total_loss_avg / (i + 1)),3),

        "test xentropy loss":round(test_xentropy_loss,3),
        'train_acc': round(overall_accuracy,3), 'test_acc': test_acc,'train_ucsf_ctrl':train_ucsf_ctrl, 'train_ucsf_mci':train_ucsf_mci,
        'train_ucsf_hiv':train_ucsf_hiv, 'train_ucsf_mnd':train_ucsf_mnd,

        'test_ucsf_ctrl':test_ucsf_ctrl, 'test_ucsf_mci':test_ucsf_mci,
        'test_ucsf_hiv':test_ucsf_hiv, 'test_ucsf_mnd':test_ucsf_mnd})

# # Optional
       # wandb.watch(feature_extractor)
       # wandb.watch(classifier_ucsf)


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

        print("Begin training fold ",fold)


        feature_extractor = fe(trainset_size = len(train_data), in_num_ch=1, img_size=(64, 64, 64), inter_num_ch=16,
                           fc_num_ch=16, kernel_size=3, conv_act='LeakyReLU',
                           fe_arch=args.fe_arch, dropout=args.dropout,
                           fc_dropout = args.fc_dropout, batch_size = args.batch_size).to(device)
        
        #LOAD IN WEIGHTS FROM PRETRAINED MODEL (OPTIONAL) ENSURE THAT THE FeatureExtractor version is being used
        #feature_extractor.load_state_dict(torch.load('curr_model.pth'), strict=False)

        classifier_ucsf = nn.Sequential(
                    #nn.Linear(2048, 128),
                    nn.Linear(256, 128),
                    nn.Dropout(0.25),
                    nn.LeakyReLU(),
                    nn.Linear(128, 16),
                    nn.LeakyReLU(),    
                    nn.Linear(16, 2)).to(device)


        test_acc, test_accuracy_class, test_accuracy_dataset, best_accuracy, best_epoch, best_models = train(feature_extractor,  classifier_ucsf, train_data,  test_data, fold = fold)


        feature_extractor, classifier_ucsf = best_models
        best_accuracy_list[fold] = best_accuracy
        final_accuracy_list[fold] = test_acc
        best_epoch_list[fold] = best_epoch

        # test_acc, test_xloss, test_ageloss, test_total_loss, test_accuracy_class, test_accuracy_dataset = test(feature_extractor, classifier_ucsf, test_loader, ucsf_criterion,  fold = fold, train =True)
        # acc_each_class_list.append( test_accuracy_class)
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

    csv_logger_pred.close()


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

