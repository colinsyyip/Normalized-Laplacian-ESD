# Script to generate Table 1 regarding descriptive statistics of real-world datasets

import networkx as nx
import torch
import torch.nn as nn
import datasets
from torch_geometric.data import Data
from torch_geometric.datasets import FakeDataset,FakeHeteroDataset
import numpy as np
from torch_geometric.utils import *
import matplotlib.pyplot as plt
import math
import pandas as pd
np.set_printoptions(formatter={'float': '{:0.5f}'.format})

path = '/data'
device = 'cpu'
dataset_list = [ 'cora', 'citeseer', 'pubmed', 'computers', 'photo', 'texas','cornell', 'chameleon', 'film', 'squirrel']


def distinct_num(dataset):
    """
    Calculate the number of distinct eigenvalues for a given graph's normalized Laplacian matrix
    """
    eps = 1e-6
    count  = 0
    data = datasets.load_dataset(dataset, 'dense')
    data.edge_index = to_undirected(data.edge_index)
    data.edge_index, data.edge_attr = remove_self_loops(data.edge_index)
    A =  to_dense_adj(data.edge_index).squeeze()
    I = torch.eye(A.shape[0])
    degree = torch.diag(A.sum(-1)**(-0.5))
    degree[torch.isinf(degree)] = 0.
    L_sym = I - degree.mm(A.mm(degree))
    # e,_=torch.symeig(L_sym,False)
    e = torch.linalg.eigvalsh(L_sym, UPLO='U')
    for i in range(1,len(e.numpy())):
        if torch.abs(e[i]-e[i-1])<eps:
            count = count+1
    print('dataset','node_num','distinct_num')
    print(dataset,data.x.shape[0],data.edge_index.shape[1],data.x.shape[0]-count)

    return([dataset,data.x.shape[0],data.edge_index.shape[1],data.x.shape[0]-count])

retrieved_data = []

for dataset in dataset_list:
    retrieved_data.append(distinct_num(dataset))
    
table_df = pd.DataFrame(retrieved_data, 
                        columns=['dataset', 'n_vertices', 'n_edges', 'n_distinct_eigenvalues'])
table_df['n_edges'] = table_df['n_edges']/2
table_df['p_distinct_ev'] = table_df['n_distinct_eigenvalues']/table_df['n_vertices']
table_df.to_csv("table1.csv")
