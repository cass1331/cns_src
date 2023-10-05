import numpy as np
import random
import networkx as nx
from scipy.stats import ttest_ind
import torch
from dataloading import MRI_Dataset
from transformation import super_transformation
from torch.utils.data import DataLoader
from torch import nn
import pandas as pd

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print("device:",device)

data = pd.read_csv('/home/users/jmanasse/match_frame.csv')  

def matchGroup(data, group):
    # Split data into two groups
    data1 = data[data[group] == 0.0]
    data2 = data[data[group] == 1.0]

    # Randomly permute data1
    data.sample(frac=1)

    N_etoh = len(data1.index)
    N_ctrl = len(data2.index)

    # Initialize variables for the graph
    s = []
    t = []
    weights = []
    edgeNum = 0

    # Construct the bipartite graph
    for i in range(N_etoh):
        for j in range(N_ctrl):
            if np.abs(data1.iloc[i, 4] - data2.iloc[j, 4]) <= 0.5:
                edgeNum += 1
                s.append(i)
                t.append(j + N_etoh)
                weights.append(1 + random.random() * 1e-5)

    for i in range(N_etoh):
        edgeNum += 1
        s.append(N_etoh + N_ctrl + 1)
        t.append(i)
        weights.append(1 + random.random() * 1e-5)

    for i in range(N_ctrl):
        edgeNum += 1
        s.append(N_etoh + i)
        t.append(N_etoh + N_ctrl + 2)
        weights.append(1 + random.random() * 1e-5)

    # Create a directed graph and find the max flow
    G = nx.DiGraph()
    G.add_edges_from(zip(s, t), capacity=weights)
    flow_value, flow_dict = nx.maximum_flow(G, N_etoh + N_ctrl + 1, N_etoh + N_ctrl + 2, capacity='1.0')

    # Extract selected samples
    selected_etoh = [i for i in range(N_etoh) if flow_dict[i] == 1]
    selected_ctrl = [i - N_etoh for i in range(N_etoh + N_ctrl) if flow_dict[i] == 1 and i >= N_etoh]

    # Perform a two-sample t-test on the first confounder between matched cohorts
    t_stat, p_value = ttest_ind(data1.iloc[selected_etoh, 4], data2.iloc[selected_ctrl, 4])
    print(f'2 sample t-test of the first confounder between matched cohorts p value: {p_value}')

    # Find which samples are selected in the matched groups
    selected = np.zeros(len(data), dtype=int)
    idx0 = np.where(group == 0)[0]
    selected[idx0[selected_etoh]] = 1
    idx1 = np.where(group == 1)[0]
    selected[idx1[selected_ctrl]] = 1

    print(f'{np.sum(selected)} total samples in the matched cohorts')

    return selected

# Example usage:
selected_samples = matchGroup(data, "hiv_labels")
print(selected_samples)
