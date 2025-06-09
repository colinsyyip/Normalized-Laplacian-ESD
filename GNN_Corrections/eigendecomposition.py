# Script to run decomposition of graph spectral properties. Runs on CPU, as Torch eigenvalue algorithms 
# set up for AMD machines yet. Slight modifications will need to be made for this to run on Nvidia GPUs

import datasets
import torch
import time
import sys
import os
from torch_geometric.utils import add_self_loops,to_undirected,to_dense_adj,is_undirected

dataset = ['pubmed','texas','cora', 'citeseer', 'computers', 'squirrel', 'photo', 'chameleon', 'film',  'cornell']

torch.set_num_threads(12)

for d in dataset:
    s = time.time()
    
    eigenvalues_path = 'data/eigenvalues/'+d
    eigenvectors_path = 'data/eigenvectors/'+d

    baseG = datasets.load_dataset(d, 'dense').to('cpu')

    print(d,baseG.num_nodes,baseG.edge_index.size())

    A =  to_dense_adj(baseG.edge_index).squeeze()
    n = baseG.num_nodes

    degree = torch.diag(A.sum(-1)**(-0.5))
    degree[torch.isinf(degree)] = 0.
    A_sym = degree.mm(A.mm(degree))

    e_val, e_vec = torch.linalg.eig(A_sym)
    print(d,time.time()-s)

    if not os.path.exists("data/eigenvectors/"):
        os.makedirs("data/eigenvectors/")
    if not os.path.exists("data/eigenvalues/"):
        os.makedirs("data/eigenvalues/")
    torch.save(e_vec, eigenvectors_path)
    torch.save(e_val,eigenvalues_path)

