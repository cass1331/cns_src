import torch
import matplotlib.pyplot as plt
from model import fe
from dataloading import MRI_Dataset
from transformation import super_transformation
from torch.utils.data import DataLoader
from torch import nn
from scipy.stats import chisquare,pearsonr,ttest_ind
import numpy as np
import math
import seaborn as sns
import pandas as pd
from metadatanorm2 import MetadataNorm
from mdn import get_cf_kernel, get_cf_kernel_batch
import collections

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print("device:",device)

transformation = super_transformation()
train_data = MRI_Dataset(fold = 0 , stage= 'original_train',transform = transformation)

feature_extractor = nn.Sequential(
            nn.Sequential(nn.Conv3d(in_channels=1, out_channels=16, kernel_size=3, padding=1),
                          nn.ReLU(),
                          nn.MaxPool3d(kernel_size=2)),
            nn.Sequential(nn.Conv3d(in_channels=16, out_channels=32, kernel_size=3, padding=1),
                          nn.ReLU(),
                          nn.MaxPool3d(kernel_size=2)),
            nn.Sequential(nn.Conv3d(in_channels=32, out_channels=64, kernel_size=3, padding=1),
                          nn.ReLU(),
                          nn.MaxPool3d(kernel_size=2)),
            nn.Sequential(nn.Conv3d(in_channels=64, out_channels=32, kernel_size=3, padding=1),
                          nn.ReLU(),
                          nn.MaxPool3d(kernel_size=2)),
            nn.Flatten(start_dim=1)).to(device)

classifier_ucsf = nn.Sequential(
            nn.Linear(in_features=2048, out_features=2),
            nn.Sigmoid()
        ).to(device)

#feature_extractor = fe(trainset_size = len(train_data), in_num_ch=1, img_size=(64, 64, 64), inter_num_ch=16,
#                           fc_num_ch=16, kernel_size=3, conv_act='LeakyReLU',
#                           fe_arch= 'fe1', dropout=0.2,
#                           fc_dropout = 0.2, batch_size = 1).to(device)

#classifier_ucsf = nn.Sequential(
#                     nn.Linear(2048, 128),
#                     nn.Linear(256, 128),
#                     nn.LeakyReLU(),
#                     nn.Linear(128, 16),
#                     nn.LeakyReLU(),
#                     nn.Linear(16, 2)).to(device)

final_test_loader = DataLoader(dataset=train_data ,
                          batch_size=1,#64
                          shuffle=False,
                          pin_memory=True,
                          num_workers=3)

#cf_kernel  = get_cf_kernel(final_test_loader)

#classifier_ucsf = nn.Sequential(
#                    nn.Linear(2048, 128),
#                    nn.ReLU(),
#                    MetadataNorm(batch_size=64, cf_kernel=cf_kernel, num_features = 128, trainset_size = len(train_data)),
#                    nn.Linear(128, 16),
#                    nn.ReLU(),
#                    MetadataNorm(batch_size=64, cf_kernel=cf_kernel, num_features = 16, trainset_size = len(train_data)),
#                    nn.Linear(16, 2),
#                    MetadataNorm(batch_size=64, cf_kernel=cf_kernel, num_features = 2, trainset_size = len(train_data)),
#                ).to(device)

model = torch.load('curr_model.pth')
encode_dict = collections.OrderedDict([(k[7:],v) for k,v in model.items() if 'linear' not in k])
linear_dict = collections.OrderedDict([(k[7:],v) for k,v in model.items() if 'linear' in k])

feature_extractor.load_state_dict(encode_dict)
classifier_ucsf.load_state_dict(linear_dict)
#feature_extractor.load_state_dict(torch.load('fe_mdn_weights.pt'))
#classifier_ucsf.load_state_dict(torch.load('class_mdn_weights.pt'))

feature_extractor.eval()
classifier_ucsf.eval()

def plot_mdn_corr(final_test_loader,feature_extractor, classifier_ucsf):
    fig, axes = plt.subplots(1,2,sharey=True)
    score_1=[]
    score_2=[]
    code = ['male','female']
    genders_1={'male':[],'female':[]}
    genders_2={'male':[],'female':[]}
    ages_=[]
    genders_=[]
    with torch.no_grad():
        for (images, labels, actual_labels, datasets, ids, ages, genders,npzs)  in final_test_loader:
            try:
               images = images.view(64, 1, 64, 64, 64).to(device).float()
            except:
               break
            feature = feature_extractor(images)
            scores= [list([float(pred) for pred in preds]) for preds in list(classifier_ucsf(feature).cpu())]
            #print(scores)
            score1 = [score[0] for score in scores]
            score2 = [score[1] for score in scores]
            #print(score1)
            #print(score2)
            score_1.extend(score1)
            score_2.extend(score2)
            for i,gender in enumerate(genders):
               genders_1[code[int(gender)]].append(score1[i])
               genders_2[code[int(gender)]].append(score2[i])
            ages_.extend([float(age) for age in ages])
            genders_.extend(['male' if gender == 0 else 'female'for gender in genders])

    gendersdf_1 = pd.DataFrame(data={'scores': score_1, 'genders': genders_})
    gendersdf_2 = pd.DataFrame(data={'scores': score_2, 'genders': genders_})
     
    age_r1,p1 = pearsonr(ages_, score_1)
    age_r2,p2 = pearsonr(ages_, score_2)
    print('r:' + str(age_r1) + " p-val: "+ str(p1))
    print('r:' + str(age_r2) + " p-val: "+ str(p2))

    age_fig, ages_axes = plt.subplots(1,2,sharey=True)
    ages_axes[0].scatter(ages_, score_1)
    ages_axes[0].legend(['r:' + str(round(age_r1,3)) + " p-val: "+ str(round(p1,3))])
    ages_axes[1].scatter(ages_, score_2)
    ages_axes[1].legend(['r:' + str(round(age_r2,3)) + " p-val: "+ str(round(p2,3))])
    ages_axes[0].set_xlabel('age (norm)')
    ages_axes[1].set_xlabel('age (norm)')
    ages_axes[0].set_ylabel('score')
    age_fig.savefig('corr_mdn.png')

    t_1,p_1 = ttest_ind(list(gendersdf_1['scores'].loc[gendersdf_1['genders'] == 'male']),list(gendersdf_1['scores'].loc[gendersdf_1['genders'] == 'female']), equal_var=False)
    t_2,p_2 = ttest_ind(list(gendersdf_2['scores'].loc[gendersdf_2['genders'] == 'male']),list(gendersdf_2['scores'].loc[gendersdf_2['genders'] == 'female']), equal_var=False)

    print('t-stat:' + str(t_1) + " p-val: "+ str(p_1))
    print('t_stat:' + str(t_2) + " p-val: "+ str(p_2))
    
    sns.stripplot(ax = axes[0], data=gendersdf_1,y='scores',x='genders', orient='v')
    sns.stripplot(ax = axes[1], data=gendersdf_2,y='scores',x='genders',orient='v')
    axes[0].set_title('score 1')
    axes[0].legend([" p-val: "+ str(p_1)])
    axes[1].set_title('score 2')
    axes[1].legend([" p-val: "+ str(p_2)])

    fig.savefig('scores1_mdn.png')
    
#plot_mdn_corr(final_test_loader,feature_extractor, classifier_ucsf) 

def plot_corr(final_test_loader,feature_extractor, classifier_ucsf):
    fig, axes = plt.subplots(1,2,sharey=True)
    score_1=[]
    score_2=[]
    pred_1=[]
    pred_2=[]
    code = ['male','female']
    genders_1={'male':[],'female':[]}
    genders_2={'male':[],'female':[]}
    ages_=[]
    genders_=[]
    with torch.no_grad():
        for (images, labels, actual_labels, datasets, ids, ages, genders,npzs)  in final_test_loader:
            images = images.view(1, 1, 64, 64, 64).to(device).float()
            feature = feature_extractor(images)
            scores= classifier_ucsf(feature).cpu()
            score1 = float(scores[0][0])
            score2 = float(scores[0][1])
            score_1.append(score1)
            score_2.append(score2)
            pred_1.append(0 if score1<0 else 1)
            pred_2.append(0 if score2<0 else 1)
            genders_1[code[int(genders)]].append(score1)
            genders_2[code[int(genders)]].append(score2)
            ages_.append(ages)
            genders_.append('male' if genders == 0 else 'female')

    gendersdf_1 = pd.DataFrame(data={'scores': score_1, 'genders': genders_})
    gendersdf_2 = pd.DataFrame(data={'scores': score_2, 'genders': genders_})
     
    age_r1,p1 = pearsonr(ages_, score_1)
    age_r2,p2 = pearsonr(ages_, score_2)
    print('r:' + str(age_r1) + " p-val: "+ str(p1))
    print('r:' + str(age_r2) + " p-val: "+ str(p2))

    age_fig, ages_axes = plt.subplots(1,2,sharey=True)
    ages_axes[0].scatter(ages_, score_1)
    ages_axes[0].legend(['r:' + str(round(age_r1,3)) + " p-val: "+ str(round(p1,3))])
    ages_axes[1].scatter(ages_, score_2)
    ages_axes[1].legend(['r:' + str(round(age_r2,3)) + " p-val: "+ str(round(p2,3))])
    ages_axes[0].set_xlabel('age (norm)')
    ages_axes[1].set_xlabel('age (norm)')
    ages_axes[0].set_ylabel('score')
    age_fig.savefig('corr.png')

    t_1,p_1 = ttest_ind(list(gendersdf_1['scores'].loc[gendersdf_1['genders'] == 'male']),list(gendersdf_1['scores'].loc[gendersdf_1['genders'] == 'female']), equal_var=False)
    t_2,p_2 = ttest_ind(list(gendersdf_2['scores'].loc[gendersdf_2['genders'] == 'male']),list(gendersdf_2['scores'].loc[gendersdf_2['genders'] == 'female']), equal_var=False)

    print('t-stat:' + str(t_1) + " p-val: "+ str(p_1))
    print('t_stat:' + str(t_2) + " p-val: "+ str(p_2))
    
    sns.stripplot(ax = axes[0], data=gendersdf_1,y='scores',x='genders', orient='v')
    sns.stripplot(ax = axes[1], data=gendersdf_2,y='scores',x='genders',orient='v')
    axes[0].set_title('score 1')
    axes[0].legend([" p-val: "+ str(p_1)])
    axes[1].set_title('score 2')
    axes[1].legend([" p-val: "+ str(p_2)])

    fig.savefig('scores1.png')
    
    
plot_corr(final_test_loader,feature_extractor, classifier_ucsf)  


def plot_score(final_test_loader,feature_extractor):
    fig, ax = plt.subplots()
    lens=[]
    ages_=[]
    tau=feature_extractor.get_parameter('feature_extractor.tau')
    tau=tau/float(torch.norm(tau))
    with torch.no_grad():
        for (images, labels, actual_labels, datasets, ids, ages, genders,npzs)  in final_test_loader:
        
            if npzs == 99.0 or math.isnan(npzs):
                continue
            #print(labels, actual_labels, datasets, ids, ages, genders)
            images = images.view(1, 1, 64, 64, 64).to(device).float()
            #print(images)
            feature = feature_extractor(images)
            len_ = torch.sum(feature * tau.repeat(feature.shape[0], 1), dim=1)
            #ax.scatter(ages,len_.cpu(), c=ages*2, cmap='Greens')
            ages_.append(float(npzs))
            lens.append(float(len_.cpu()))
        #   project onto current tau direction (before reset!)
    #lens = [torch.sum(feature * tau.repeat(feature.shape[0], 1), dim=1) for feature in features]
    #   obtain real age
    #   plot plot of proj len against real age
    #ax.scatter(ages,lens)
    # end
        # save plot to .png
    r = np.corrcoef(ages_, lens)
    print(r)

    #ax.title(str(r)) 
    ax.scatter(ages_,lens,c=ages_, cmap='viridis')
    ax.set_xlabel('npz score')
    ax.set_ylabel('proj score')
    ax.legend(['lamb = ' + str(0.05) + ', r = '+ str(round(r[1][0],3))])
    fig.savefig('NPZ.png')


