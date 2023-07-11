import numpy as np
import scipy.stats as st
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim

from tqdm import tqdm

import warnings
warnings.filterwarnings('ignore')


class Net(torch.nn.Module):

    def __init__(self):
        super().__init__()
        self.device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
        self.tau = torch.ones(128).double().to(self.device) # initial "neutral" tau direction, which gets learned in training        
        ## Encode to 128-vector
        self.encode = nn.Sequential(
            nn.Sequential(nn.Conv3d(in_channels=1, out_channels=16, kernel_size=3),
                          nn.ReLU(),
                          nn.MaxPool3d(kernel_size=2)),
            nn.Sequential(nn.Conv3d(in_channels=16, out_channels=32, kernel_size=3),
                          nn.ReLU(),
                          nn.MaxPool3d(kernel_size=2)),
            nn.Sequential(nn.Conv3d(in_channels=32, out_channels=64, kernel_size=3),
                          nn.ReLU(),
                          nn.MaxPool3d(kernel_size=2)),
            nn.Sequential(nn.Conv3d(in_channels=64, out_channels=16, kernel_size=3),
                          nn.ReLU(),
                          nn.MaxPool3d(kernel_size=2)),
            nn.Flatten(start_dim=0)                          
        )
        # Perform Soft-Classification
        self.linear = nn.Sequential(
            nn.Linear(in_features=128, out_features=2),
            nn.Sigmoid()
        )        
        
    def forward(self, x):
        ## Encode to R^128
        x = x.view((-1, 1, 64, 64, 64)).double()
        x = self.encode(x)
        ## Classif 
        x = self.linear(x)
        return x

