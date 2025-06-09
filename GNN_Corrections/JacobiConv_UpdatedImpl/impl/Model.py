import torch
import torch.nn as nn
from torch import Tensor
from scipy.special import comb
from torch_geometric.utils import add_self_loops,to_undirected,to_dense_adj
from torch_geometric.utils import get_laplacian, degree
from torch_sparse import SparseTensor
from torch_geometric.nn import MessagePassing
import torch.nn.functional as F
import numpy as np
import time
import sys
import os
import sys
from .models import Seq, TensorMod
# from gpu_mem_track import MemTracker

# gpu_tracker = MemTracker()   

class JacobiConv_model(nn.Module):
    def __init__(self,
                 adj,
                 x,
                 dpb,
                 dpt,
                 out_channels,
                 method,
                 dataset,
                 depth: int = 3,
                 cached: bool = True,
                 alpha: float = 1.0,
                 fixed: float = False,
                 beta: float = 0.0,
                 gamma: float = 0.5):
        super().__init__()
        self.depth = depth
        self.basealpha = alpha
        self.alphas = nn.ParameterList([
            nn.Parameter(torch.tensor(float(min(1 / alpha, 1))),
                         requires_grad=not fixed) for i in range(depth + 1)
        ])
        self.cached = cached
        self.method = method
        self.dataset = dataset
        self.adj = adj
        self.beta = beta
        self.gamma = gamma
        self.comb_weight = nn.Parameter(torch.ones((1, depth+1, out_channels)))
        # self.comb_weight = nn.Parameter(torch.ones((1, depth+1)))
        self.emb = Seq([
            TensorMod(x),
            nn.Dropout(p=dpb),
            nn.Linear(x.shape[1], out_channels),
            nn.Dropout(dpt, inplace=True)
        ])
        # self.emb1 = Seq([TensorMod(baseG.x[:, image_idx].reshape(-1, 1))])
        
    def forward(self, corrected_e, ones, U:Tensor, x: Tensor, edge_index: Tensor, edge_attr: Tensor):
        # gpu_tracker.track() 
        utx = U.t() @ self.emb()
        alphas = [self.basealpha * torch.tanh(_) for _ in self.alphas]
        # gpu_tracker.track() 
        xs = [JacobiConv(0, [ones], corrected_e, alphas)]
        for L in range(1, self.depth + 1):
            tx = JacobiConv(L, xs, corrected_e,  alphas)
            xs.append(tx)
        # gpu_tracker.track() 
        xs = [(x.unsqueeze(1) * utx).unsqueeze(1)  for x in xs]
        x = torch.cat(xs, dim=1)
        # gpu_tracker.track() 
        x = x * self.comb_weight
        x = torch.sum(x, dim=1)
        # gpu_tracker.track() 
        # sys.exit()
        return U @ x
    



class JacobiConv_model_Img(nn.Module):
    def __init__(self,
                 idx,
                 x,
                 out_channels,
                 method,
                 dataset,
                 depth: int = 3,
                 cached: bool = True,
                 alpha: float = 1.0,
                 fixed: float = False,
                 beta: float = 0.0,
                 gamma: float = 0.5):
        super().__init__()
        self.depth = depth
        self.basealpha = alpha
        self.alphas = nn.ParameterList([
            nn.Parameter(torch.tensor(float(min(1 / alpha, 1))),
                         requires_grad=not fixed) for i in range(depth + 1)
        ])
        self.cached = cached
        self.method = method
        self.dataset = dataset
        # self.adj = adj
        self.beta = beta
        self.gamma = gamma
        self.comb_weight = nn.Parameter(torch.ones((1, depth+1, out_channels)))
        self.emb = Seq([TensorMod(x[:, idx].reshape(-1, 1))])
        
    def forward(self,corrected_e,ones,U:Tensor, x1: Tensor, edge_index: Tensor, edge_attr: Tensor):
        # gpu_tracker.track() 
        utx = U.t() @ self.emb()
        alphas = [self.basealpha * torch.tanh(_) for _ in self.alphas]
        # gpu_tracker.track() 
        xs = [JacobiConv(0, [ones], corrected_e, alphas)]
        for L in range(1, self.depth + 1):
            tx = JacobiConv(L, xs, corrected_e,  alphas)
            xs.append(tx)
        # gpu_tracker.track() 
        xs = [(x.unsqueeze(1) * utx).unsqueeze(1)  for x in xs]
        x = torch.cat(xs, dim=1)
        # gpu_tracker.track() 
        x = x * self.comb_weight
        x = torch.sum(x, dim=1)
        # gpu_tracker.track() 
        # sys.exit()
        return U @ x



def JacobiConv(L, xs, adj, alphas, a=1.0, b=1.0, l=-1.0, r=1.0):
    '''
    Jacobi Bases. Please refer to our paper for the form of the bases.
    '''
    if L == 0: return xs[0]
    if L == 1:
        coef1 = (a - b) / 2 - (a + b + 2) / 2 * (l + r) / (r - l)
        coef1 *= alphas[0]
        coef2 = (a + b + 2) / (r - l)
        coef2 *= alphas[0]
        return coef1 * xs[-1] + coef2 * (adj *  xs[-1])
    coef_l = 2 * L * (L + a + b) * (2 * L - 2 + a + b)  
    coef_lm1_1 = (2 * L + a + b - 1) * (2 * L + a + b) * (2 * L + a + b - 2)
    coef_lm1_2 = (2 * L + a + b - 1) * (a**2 - b**2)
    coef_lm2 = 2 * (L - 1 + a) * (L - 1 + b) * (2 * L + a + b)  
    tmp1 = alphas[L - 1] * (coef_lm1_1 / coef_l)
    tmp2 = alphas[L - 1] * (coef_lm1_2 / coef_l)
    tmp3 = alphas[L - 1] * alphas[L - 2] * (coef_lm2 / coef_l)
    tmp1_2 = tmp1 * (2 / (r - l))
    tmp2_2 = tmp1 * ((r + l) / (r - l)) + tmp2
    nx = tmp1_2 * (adj *  xs[-1]) - tmp2_2 * xs[-1]
    nx -= tmp3 * xs[-2]
    return nx