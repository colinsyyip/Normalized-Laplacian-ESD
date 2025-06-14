from typing import Optional
from torch_geometric.typing import OptTensor
import math
import torch
from torch.nn import Parameter
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.utils import remove_self_loops, add_self_loops
from torch_geometric.utils import get_laplacian
from scipy.special import comb
import torch.nn.functional as F
from torch_geometric.nn.conv.gcn_conv import gcn_norm
import numpy as np
from torch import Tensor


class Bern_prop(MessagePassing):
    """
    Implementation of BernProp polynomial filter adapted for eigenvalue functions.

    Adapted from “Improving expressive power of spectral graph neural networks with eigenvalue correction”.
    """
    def __init__(self, K, bias=True, **kwargs):
        super(Bern_prop, self).__init__(aggr='add', **kwargs)
        
        self.K = K
        self.temp = Parameter(torch.Tensor(self.K+1))
        self.reset_parameters()

    def reset_parameters(self):
        self.temp.data.fill_(1)

    def forward(self,x, origin_e,U):
        TEMP=F.relu(self.temp)
        out =(comb(self.K,0)/(2**self.K))*TEMP[0]*((1+origin_e)**self.K)
        for i in range(self.K):
            new_lamma = ((1+origin_e)**(self.K-i-1))*((1-origin_e)**(i+1))
            out += (comb(self.K,i+1)/(2**self.K))*TEMP[i+1]*new_lamma

        return U @ (out.unsqueeze(1) * (U.t() @ x))
    
    def message(self, x_j, norm):
        return norm.view(-1, 1) * x_j

    def __repr__(self):
        return '{}(K={}, temp={})'.format(self.__class__.__name__, self.K,
                                          self.temp)
