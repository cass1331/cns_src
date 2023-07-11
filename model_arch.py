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

    def pair_loss(self, emb_t1, emb_t2, delta_sigma, prob_pred, label, lamb):
        """
        emb_t1 (128-tensor): embedding for first image in pair
        emb_t2 (128-tensor): embedding for second image in pair
        delta_sigma (float): ground truth difference in value for factor to disentangle
        prob_pred (tuple[float, float]): model's probability prediction for each of the labels, i.e. (0.12, 0.96)
        label (tuple[int, int]): ground truth class membership for the two independent label, i.e. (0, 0), (0, 1), (1, 0), (1, 1)
        lamb: weighting factor between BCE loss and disentangle loss
        """
        bce = nn.BCELoss()
        prob_pred = torch.tensor(prob_pred).double()
        label = torch.tensor(label).double()
        bce_loss = bce(prob_pred, label)
        proj_e1_len = torch.norm((torch.dot(emb_t1, self.tau) / torch.dot(self.tau, self.tau)) * self.tau)
        proj_e2_len = torch.norm((torch.dot(emb_t2, self.tau) / torch.dot(self.tau, self.tau)) * self.tau)
        emb_len_diff = torch.abs(proj_e2_len - proj_e1_len)
        disentangle_loss = torch.abs(emb_len_diff - delta_sigma)
        return bce_loss + lamb*disentangle_loss   

    def train(self, train_pairs, test_pairs, train_sigmas, test_sigmas, train_labels, test_labels, alpha=0.001, epochs=100, lamb=0.5):
        """
        train_pairs (List[Tuple[64x64x64-tensor]]): list with the pairs of images to train on 
        test_pairs (List[Tuple[64x64x64-tensor]]): list with the pairs of images to test on 
        train_sigmas (List[float]): list of the **difference** in value to disentangle between the two images in each pair for training
        test_sigmas (List[float]): list of the **difference** in value to disentangle between the two images in each pair for testing
        train_labels (List[Tuple[int, int]]): ground truth class membership for the two independent labels for training
        test_labels (List[Tuple[int, int]]): ground truth class membership for the two independent labels for testing
        alpha (float): learning rate
        epochs (int): number of epochs to train for
        lamb (float): weighting factor between BCE loss and disentangle loss                     
        """
        train_loss_vals = list()
        opt = optim.Adam(self.parameters(), lr=alpha)
        train_loss_vals, train_acc, test_loss_vals, test_acc = list(), list(), list(), list() # for tracking the evolution over training
        for e in (range(epochs)):
            total_loss = 0 # incrementaly updated with total epoch loss
            correct = 0 # count of correct classifications on an epoch
            for i in range(len(train_pairs)):
                opt.zero_grad()
                img1 = train_pairs[i][0].view(1, 1, 64, 64, 64)
                img2 = train_pairs[i][1].view(1, 1, 64, 64, 64)
                pred = self.forward(img1) # classification is performed based on first image of the pair
                if torch.all(torch.eq(torch.round(pred), torch.tensor(train_labels[i]))): # "correct" only if both indep. labels match
                    correct += 1
                emb1 = self.encode(img1) # embedding for first image in the pair
                emb2 = self.encode(img2) # embedding for second image in the pair
                loss = self.pair_loss(emb_t1=emb1, emb_t2=emb2, delta_sigma=train_sigmas[i], prob_pred=pred, label=train_labels[i], lamb=lamb)
                loss.backward()
                opt.step()
                self.tau = self.encode(torch.ones((1, 1, 64, 64, 64)).double()).to(self.device) # update tau-direction (jointly learned)
                total_loss += float(loss.data)
            total_loss /= len(train_pairs) # extract avg loss per pair for direct comparision
            train_loss_vals.append(total_loss)
            train_acc.append(correct / len(train_pairs))
            total_loss = 0
            correct = 0
            # testing on held-out set (repeats the same procedure from training but without updating parameters)
            with torch.no_grad():
                for i in range(len(test_pairs)):
                    img1 = test_pairs[i][0].view(1, 1, 64, 64, 64)
                    img2 = test_pairs[i][1].view(1, 1, 64, 64, 64)
                    pred = self.forward(img1)
                    if torch.all(torch.eq(torch.round(pred), torch.tensor(train_labels[i]))):
                        correct += 1
                    loss = self.pair_loss(self.encode(img1), self.encode(img2), test_sigmas[i], self.forward(img1), test_labels[i], lamb)
                    total_loss += float(loss.data)        
                total_loss /= len(test_pairs) 
                test_loss_vals.append(total_loss)
                test_acc.append(correct / len(test_pairs))
            print(f'    Epoch {e+1} -> train avg loss = {round(train_loss_vals[-1], 4)} | test avg loss = {round(test_loss_vals[-1], 4)} |',
                    f'train acc = {round(train_acc[-1], 4)} | test acc = {round(test_acc[-1], 4)}')
        # Train/Test Loss Plot 
        plt.plot(range(1, len(train_loss_vals)+1), train_loss_vals, label='combined train loss')
        plt.plot(range(1, len(test_loss_vals)+1), test_loss_vals, label='combined test loss')
        # plt.plot([1, len(train_loss_vals)], [0, 0], label='theoretical lower bound') # assuming bce->0 and dist_len->0
        plt.legend(loc='upper right')
        plt.xlabel('epochs')
        plt.ylabel('avg {cross-entropy + disentanglement} loss')
        plt.title(f'Model Train/Test Loss - lr={alpha}, epochs={epochs}, lambda={lamb}')
        # plt.savefig(f'loss___lr{alpha}_e{epochs}_lamb{lamb}.png')
        plt.show()
        # Train/Test Accuracy Plot
        plt.plot(range(1, len(train_acc)+1), train_acc, label='train accuracy')
        plt.plot(range(1, len(test_acc)+1), test_acc, label='test accuracy')
        plt.legend(loc='lower right')
        plt.xlabel('epochs')
        plt.ylabel('accuracy')
        plt.title(f'Model Binary Classif Acc - lr={alpha}, epochs={epochs}, lambda={lamb}')
        # plt.savefig(f'acc___lr{alpha}_e{epochs}_lamb{lamb}.png')
        plt.show()             

def main():
    model = Net().double()
    TRAIN_COUNT = 16
    TEST_COUNT = 4
    ALPHA = 0.1
    EPOCHS = 5
    LAMB = 0.5
    SEED = 0
    np.random.seed(SEED)
    torch.seed()
    train_pairs = [(torch.ones((64, 64, 64)).double(), torch.ones((64, 64, 64)).double()) for _ in range(TRAIN_COUNT)]
    test_pairs = [(torch.ones((64, 64, 64)).double(), torch.ones((64, 64, 64)).double()) for _ in range(TEST_COUNT)]
    train_sigmas = [0.1 for _ in range(len(train_pairs))]
    test_sigmas = [0.1 for _ in range(len(test_pairs))]
    train_labels = [(0, 1) for _ in range(len(train_pairs))]
    test_labels = [(0, 1) for _ in range(len(test_pairs))]
    model.train(train_pairs, test_pairs, train_sigmas, test_sigmas, train_labels, test_labels, ALPHA, EPOCHS, LAMB)

# execute:
main()

