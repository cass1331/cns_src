from dataloading import MRI_Dataset
import numpy as np
from torch.utils.data import DataLoader
from datasplitting import Specific_MRI_Dataset
from tqdm import tqdm
from sampler import SuperSampler,MixedSampler
from transformation import super_transformation
import torch

transformation = super_transformation()
test_data = MRI_Dataset(fold = 4 , stage= 'original_test')
train_data = MRI_Dataset(fold = 4 , stage= 'original_train',transform = transformation)
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
for i,(images, labels,actual_labels,datasets,ids, ages, genders) in enumerate(train_loader):
    if datasets == 'ucsf':
        ucsf_data.append((images, datasets, ids, actual_labels, labels, ages, genders))
    elif datasets == 'adni':
        adni_data.append((images, datasets, ids, actual_labels, labels, ages, genders,))
    else:
        lab_data.append((images, datasets, ids, actual_labels, labels, ages, genders))
# fold = 0
# transformation = super_transformation()
# train_data = MRI_Dataset(fold = fold , stage= 'original_train',transform = transformation)
# test_data = MRI_Dataset(fold = fold , stage= 'original_test')
#
# train_loader = DataLoader(dataset=train_data,
#                           batch_size=None,
#                           sampler=MixedSampler(dataset=train_data,
#                                                batch_size=128),
#                           shuffle=False,
#                           pin_memory=True,
#                           num_workers=3)
# test_loader = DataLoader(dataset=test_data ,
#                           batch_size=None,# to include all test images
#                           shuffle=True,
#                           pin_memory=True,
#                           num_workers=3)
#
# ucsf_data = []
# adni_data = []
# lab_data = []
#
# for i,(images, labels,actual_labels,datasets,ids, ages, genders) in enumerate(test_loader):
#     if datasets == 'ucsf':
#         ucsf_data.append((datasets, ids, actual_labels, labels, ages, genders))
#     elif datasets == 'adni':
#         adni_data.append((datasets, ids, actual_labels, labels, ages, genders))
#     else:
#         lab_data.append((datasets, ids, actual_labels, labels, ages, genders))

ucsf_dataset = Specific_MRI_Dataset(ucsf_data)
adni_dataset = Specific_MRI_Dataset(adni_data)
lab_dataset = Specific_MRI_Dataset(lab_data)
#
ucsf_loader = DataLoader(dataset=ucsf_dataset,
                          batch_size=64,
                          # sampler=MixedSampler(dataset=ucsf_dataset,
                          #                      batch_size=12),
                          shuffle=False,
                          pin_memory=True,
                          num_workers=3)

adni_loader = DataLoader(dataset=adni_dataset,
                          batch_size=64,
                          # sampler=MixedSampler(dataset=adni_dataset,
                          #                      batch_size=12),
                          shuffle=False,
                          pin_memory=True,
                          num_workers=3)
lab_loader = DataLoader(dataset=lab_dataset,
                          batch_size=64,
                          # sampler=MixedSampler(dataset=lab_dataset,
                          #                      batch_size=12),
                          shuffle=False,
                          pin_memory=True,
                          num_workers=3)

# for i, (images, labels, actual_labels, datasets, ids, ages, genders) in enumerate(ucsf_loader):
#     print(images, labels, actual_labels, datasets, ids, ages, genders)
for loader in [ucsf_loader, adni_loader, lab_loader]:
    # progress_total = 0
    # num_samples = []
    # for i, batch in enumerate(loader):
    #     progress_total += 1
    #     num_samples.append(len(batch[0]))
    #
    # progress_bar = tqdm(loader, total = progress_total)
    for i, (images, labels, actual_labels, datasets, ids, ages, genders) in enumerate(loader):
        print(labels, actual_labels, datasets, ids, ages, genders)
# print(len(ucsf_data), len(adni_data), len(lab_data))
#print(ucsf_data, adni_data, lab_data)
