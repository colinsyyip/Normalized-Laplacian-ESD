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
# from gpu_mem_track import MemTracker

# gpu_tracker = MemTracker(path='gpu_mem/')   

def buildAdj(gamma:float, beta:float, method:str,dataset:str,edge_index: Tensor, edge_weight: Tensor, n_node: int, aggr: str):
    '''
    convert edge_index and edge_weight to the sparse adjacency matrix.
    Args:
        edge_index (Tensor): shape (2, number of edges).
        edge_attr (Tensor): shape (number of edges).
        n_node (int): number of nodes in the graph.
        aggr (str): how adjacency matrix is normalized. choice: ["mean", "sum", "gcn"]
    '''
    # deg = degree(edge_index[0], n_node)
    # deg[deg < 0.5] += 1.0
    ret = None
    if aggr == "mean":
        val = (1.0 / deg)[edge_index[0]] * edge_weight
    elif aggr == "sum":
        val = edge_weight
    elif aggr == "gcn":
        eigenvectors_path = '/data/lukangkang/data/data/eigenvectors_Img_Asym.npy'
        eigenvalues_path = '/data/lukangkang/data/data/eigenvalues_Img_Asym.npy'
        # gpu_tracker.track() 
        # edge_index,edge_weight = to_undirected(edge_index,edge_weight)

        # A_sym_path = '/data/lukangkang/data/data/'+dataset+"/"+'A_sym'
        # A_sym = torch.load(A_sym_path).to(edge_index.device)
        n = n_node

        A =  to_dense_adj(edge_index).squeeze()
        n = A.shape[0]
        # gpu_tracker.track() 
        # start = time.time()
        degree = torch.diag(A.sum(-1)**(-0.5))
        degree[torch.isinf(degree)] = 0.
        A_sym = degree.mm(A.mm(degree))

        if os.path.exists(eigenvectors_path) and os.path.exists(eigenvalues_path):
            e = torch.load(eigenvalues_path).to(edge_index.device)
            U = torch.load(eigenvectors_path).to(edge_index.device)
        else:
            print('eig start')
            t = time.time()
            e,U=torch.linalg.eigh(A_sym,UPLO='U')
            torch.save(U,eigenvectors_path)
            torch.save(e,eigenvalues_path)
            print('eig end：',time.time()-t)

        # eigenvectors_path = '/data/lukangkang/data/data/eigenvectors.npy'
        # eigenvalues_path = '/data/lukangkang/data/data/eigenvalues.npy'
        # U = torch.FloatTensor(np.load(eigenvectors_path)).to(edge_index.device)
        # e = torch.FloatTensor(np.load(eigenvalues_path)).to(edge_index.device)
        
        # gpu_tracker.track() 
        uniform_e = torch.diag(torch.FloatTensor(np.linspace(-1, 1, n))).to(edge_index.device)

        U = (U.mm(uniform_e)).mm(U.t())

        # gpu_tracker.track() 

        U = beta*A_sym+(1-beta)*U
        return U
    
        
    else:
        raise NotImplementedError
    ret = SparseTensor(row=edge_index[0],
                       col=edge_index[1],
                       value=val,
                       sparse_sizes=(n_node, n_node)).coalesce()
    print(ret)
    ret = ret.cuda() if edge_index.is_cuda else ret
    return ret


class PolyConvFrame(nn.Module):
    '''
    A framework for polynomial graph signal filter.
    Args:
        conv_fn: the filter function, like PowerConv, LegendreConv,...
        depth (int): the order of polynomial.
        cached (bool): whether or not to cache the adjacency matrix.
        alpha (float):  the parameter to initialize polynomial coefficients.
        fixed (bool): whether or not to fix to polynomial coefficients.
    '''
    def __init__(self,
                 method,
                 dataset,
                 conv_fn,
                 depth: int = 3,
                 aggr: int = "gcn",
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
        self.aggr = aggr
        self.adj = None
        self.conv_fn = conv_fn
        self.beta = beta
        self.gamma = gamma
        # self.U = torch.load('/data/lukangkang/data/data/eigenvectors/'+dataset).to('cuda:2')

    def forward(self, x: Tensor, edge_index: Tensor, edge_attr: Tensor):
        '''
        Args:0
            x: node embeddings. of shape (number of nodes, node feature dimension)
            edge_index and edge_attr: If the adjacency is cached, they will be ignored.
        '''
        # a = time.time()
        # gpu_tracker.track() 
        if self.adj is None or not self.cached:
            print('*'*40)
            n_node = x.shape[0]
            self.adj = buildAdj(self.gamma, self.beta,self.method, self.dataset,edge_index, edge_attr, n_node, self.aggr)
        
        # utx = torch.mm(self.U.t(),x)
        # # print(self.adj.size())
        # b = time.time()
        # print('b-a:',b-a)
        # gpu_tracker.track() 
        alphas = [self.basealpha * torch.tanh(_) for _ in self.alphas]

        xs = [self.conv_fn(0, [x], self.adj, alphas)]
        # print(xs[0].shape)
        for L in range(1, self.depth + 1):
            # tx = torch.mm(self.U,self.conv_fn(L, xs, self.adj, alphas))
            tx = self.conv_fn(L, xs, self.adj, alphas)
            xs.append(tx)
        xs = [x.unsqueeze(1) for x in xs]
        x = torch.cat(xs, dim=1)
        # gpu_tracker.track() 
        # c = time.time()
        # print('c-b:',c-b)
        return x

'''
conv_fns to build the polynomial filter.
Args:
    L (int): the order of polynomial basis.
    xs (List[Tensor]): the node embeddings filtered by the previous bases.
    adj (SparseTensor): adjacency matrix
    alphas (List[Float]): List of polynomial coeffcient.
'''


def PowerConv(L, xs, adj, alphas):
    '''
    Monomial bases.
    '''
    if L == 0: return xs[0]
    return alphas[L] * (adj @ xs[-1])


def LegendreConv(L, xs, adj, alphas):
    '''
    Legendre bases. Please refer to our paper for the form of the bases.
    '''
    if L == 0: return xs[0]
    nx = (alphas[L - 1] * (2 - 1 / L)) * (adj @ xs[-1])
    if L > 1:
        nx -= (alphas[L - 1] * alphas[L - 2] * (1 - 1 / L)) * xs[-2]
    return nx


def ChebyshevConv(L, xs, adj, alphas):
    '''
    Chebyshev Bases. Please refer to our paper for the form of the bases.
    '''
    if L == 0: return xs[0]
    nx = (2 * alphas[L - 1]) * (adj @ xs[-1])
    if L > 1:
        nx -= (alphas[L - 1] * alphas[L - 2]) * xs[-2]
    return nx


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
        return coef1 * xs[-1] + coef2 * (adj @ xs[-1])
    coef_l = 2 * L * (L + a + b) * (2 * L - 2 + a + b)  # 通分之后的分母。 L就相当于论文公式14的k
    coef_lm1_1 = (2 * L + a + b - 1) * (2 * L + a + b) * (2 * L + a + b - 2)
    coef_lm1_2 = (2 * L + a + b - 1) * (a**2 - b**2)
    coef_lm2 = 2 * (L - 1 + a) * (L - 1 + b) * (2 * L + a + b)
    tmp1 = alphas[L - 1] * (coef_lm1_1 / coef_l)
    tmp2 = alphas[L - 1] * (coef_lm1_2 / coef_l)
    tmp3 = alphas[L - 1] * alphas[L - 2] * (coef_lm2 / coef_l)  # θ_k^{''}
    tmp1_2 = tmp1 * (2 / (r - l))
    tmp2_2 = tmp1 * ((r + l) / (r - l)) + tmp2
    nx = tmp1_2 * (adj @ xs[-1]) - tmp2_2 * xs[-1]
    nx -= tmp3 * xs[-2]
    return nx


class Bern_prop(MessagePassing):
    # Bernstein polynomial filter from the `"BernNet: Learning Arbitrary Graph Spectral Filters via Bernstein Approximation" paper.
    # Copied from the official implementation.
    def __init__(self, K, bias=True, **kwargs):
        super(Bern_prop, self).__init__(aggr='add', **kwargs)
        self.K = K

    def forward(self, x, edge_index, edge_weight=None):
        #L=I-D^(-0.5)AD^(-0.5)
        edge_index1, norm1 = get_laplacian(edge_index,
                                           edge_weight,
                                           normalization='sym',
                                           dtype=x.dtype,
                                           num_nodes=x.size(0))
        #2I-L
        edge_index2, norm2 = add_self_loops(edge_index1,
                                            -norm1,
                                            fill_value=2.,
                                            num_nodes=x.size(0))

        tmp = []
        tmp.append(x)
        for i in range(self.K):
            x = self.propagate(edge_index2, x=x, norm=norm2, size=None)
            tmp.append(x)

        out = [(comb(self.K, 0) / (2**self.K)) * tmp[self.K]]

        for i in range(self.K):
            x = tmp[self.K - i - 1]
            x = self.propagate(edge_index1, x=x, norm=norm1, size=None)
            for j in range(i):
                x = self.propagate(edge_index1, x=x, norm=norm1, size=None)

            out.append((comb(self.K, i + 1) / (2**self.K)) * x)
        return  torch.stack(out, dim=1)

    def message(self, x_j, norm):
        return norm.view(-1, 1) * x_j

    def __repr__(self):
        return '{}(K={}, temp={})'.format(self.__class__.__name__, self.K,
                                          self.temp)
