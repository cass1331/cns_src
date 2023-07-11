import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class MetadataNorm(nn.Module):
    def __init__(self, batch_size, cf_kernel, num_features, trainset_size, momentum=0.1):

        super(MetadataNorm, self).__init__()
        self.cf_kernel = cf_kernel
        self.batch_size = batch_size
        self.kernel_dim = cf_kernel.shape[0]
        self.cfs = nn.Parameter(torch.randn(batch_size, self.kernel_dim), requires_grad=False)
        self.num_features = num_features
        self.register_buffer('beta', torch.zeros(self.kernel_dim, self.num_features))
        self.momentum = momentum # If momentum is None, standard average is used
        self.trainset_size=trainset_size
        if momentum == None:
            self.momentum = 0.5

    def forward(self, x):
        Y = x
        # print(x.shape)
        N = x.shape[0]
        Y = Y.reshape(N, -1)
        X_batch = self.cfs # confounders for this batch onl
        scale = self.trainset_size / self.batch_size

        if self.training:
            XT = torch.transpose(X_batch, 0, 1)
            # print(np.shape(self.cf_kernel), np.shape(XT))
            pinv = torch.mm(self.cf_kernel, XT) # Calculate pinv = (X^TX)^(-1)X^T
            # print(np.shape(pinv), np.shape(Y))
            B = torch.mm(pinv, Y)

            with torch.no_grad():
                self.beta = (1-self.momentum)*self.beta + self.momentum*B
        else:
            B = self.beta
        # Create reconstructed y
        Y_r = torch.mm(X_batch[:, 2:], B[2:])
        residual = Y - scale * Y_r
        residual = residual.reshape(x.shape)
        return residual

