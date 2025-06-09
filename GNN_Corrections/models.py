import torch
import random
import math
import torch.nn.functional as F
import os.path as osp
import numpy as np
import torch_geometric.transforms as T
from torch_geometric.nn.conv.gcn_conv import gcn_norm
from torch.autograd import Variable
from torch.nn import Parameter, Linear, Parameter, ParameterList
from torch_geometric.nn import GATConv, GCNConv, ChebConv
from torch_geometric.nn import MessagePassing, APPNP
from torch_geometric.utils import to_scipy_sparse_matrix
import scipy.sparse as sp
from scipy.special import comb
from Bernpro import Bern_prop
from JacobiConv_UpdatedImpl.impl import PolyConv
from JacobiConv_UpdatedImpl.impl import models


class GPR_prop(MessagePassing):
    '''
    propagation class for GPR_GNN
    '''

    def __init__(self, K, alpha, Init, Gamma=None, bias=True, **kwargs):
        super(GPR_prop, self).__init__(aggr='add', **kwargs)
        self.K = K
        self.Init = Init
        self.alpha = alpha

        assert Init in ['SGC', 'PPR', 'NPPR', 'Random', 'WS']
        if Init == 'SGC':
            # SGC-like
            TEMP = 0.0*np.ones(K+1)
            TEMP[-1] = 1.0
        elif Init == 'PPR':
            # PPR-like
            TEMP = alpha*(1-alpha)**np.arange(K+1)
            TEMP[-1] = (1-alpha)**K
        elif Init == 'NPPR':
            # Negative PPR
            TEMP = (alpha)**np.arange(K+1)
            TEMP = TEMP/np.sum(np.abs(TEMP))
        elif Init == 'Random':
            # Random
            bound = np.sqrt(3/(K+1))
            TEMP = np.random.uniform(-bound, bound, K+1)
            TEMP = TEMP/np.sum(np.abs(TEMP))
        elif Init == 'WS':
            # Specify Gamma
            TEMP = Gamma
        self.temp = Parameter(torch.tensor(TEMP, dtype=float))

    def reset_parameters(self):
        torch.nn.init.zeros_(self.temp)
        for k in range(self.K+1):
            self.temp.data[k] = self.alpha*(1-self.alpha)**k
        self.temp.data[-1] = (1-self.alpha)**self.K

    def forward(self, x, origin_e,U):
        # Use polynomials of eigenvalues instead of time-consuming matrix polynomials

        # Start with lowest power as initializing values for all nodes
        out = self.temp[0]*(origin_e**0)
        for k in range(self.K):
            out += self.temp[k+1]*(origin_e**(k+1))
        return U @ (out.unsqueeze(1) * (U.t() @ x))
    
    def forward1(self, x, edge_index, edge_weight=None):
        edge_index, norm = gcn_norm(
            edge_index, edge_weight, num_nodes=x.size(0), dtype=x.dtype)

        hidden = x*(self.temp[0])
        for k in range(self.K):
            x = self.propagate(edge_index, x=x, norm=norm)
            gamma = self.temp[k+1]
            hidden = hidden + gamma*x
        return hidden

    def message(self, x_j, norm):
        return norm.view(-1, 1) * x_j

    def __repr__(self):
        return '{}(K={}, temp={})'.format(self.__class__.__name__, self.K,
                                          self.temp)

    def message(self, x_j, norm):
        return norm.view(-1, 1) * x_j

    def __repr__(self):
        return '{}(K={}, temp={})'.format(self.__class__.__name__, self.K,
                                          self.temp)
    

class GPRGNN(torch.nn.Module):
    def __init__(self, dataset, args, device: str = "cpu"):
        super(GPRGNN, self).__init__()

        # Define the map from the shape of x to the number of hidden layers
        self.lin1 = Linear(dataset.x.shape[1], args.hidden)
        # Define the map from the number of hidden layers to the overall number of classes in x (logits)
        self.lin2 = Linear(args.hidden, dataset.num_classes)
        self.prop1 = GPR_prop(args.K, args.alpha, args.Init)

        if device == "cuda":
            self.lin1.cuda()
            self.lin1.cuda()
            self.prop1.cuda()

        self.Init = args.Init
        self.dprate = args.dprate
        self.dropout = args.dropout

    def reset_parameters(self):
        self.prop1.reset_parameters()

    def forward(self, origin_e, U, data, *args, **kwargs):
        x, edge_index = data.x, data.edge_index

        # Apply a dropout for regularization (randomly zeros entries)
        x = F.dropout(x, p=self.dropout, training=self.training)
        # Rectify x to positive after a transform to the hidden layer
        x = F.relu(self.lin1(x))
        # Apply dropout again
        x = F.dropout(x, p=self.dropout, training=self.training)
        # Transform to output layer/classifications (logits for each class, 1 per node)
        x = self.lin2(x)

        if self.dprate == 0.0:
            # Run prop1 to go to next layer
            x = self.prop1(x, origin_e,U)
            return F.log_softmax(x, dim=1)
        else:
            # Run an additional dropout
            x = F.dropout(x, p=self.dprate, training=self.training)
            x = self.prop1(x, origin_e,U)
            return F.log_softmax(x, dim=1)
        

class BernNet(torch.nn.Module):
    def __init__(self,dataset, args, device: str = "cpu"):
        super(BernNet, self).__init__()

        self.lin1 = Linear(dataset.x.shape[1], args.hidden)
        self.lin2 = Linear(args.hidden, dataset.num_classes)
        self.m = torch.nn.BatchNorm1d(dataset.num_classes)
        self.prop1 = Bern_prop(args.K)
        if device == "cuda":
            self.lin1.cuda()
            self.lin2.cuda()
            self.m.cuda()
            self.prop1.cuda()

        self.dprate = args.dprate
        self.dropout = args.dropout

    def reset_parameters(self):
        self.prop1.reset_parameters()

    def forward(self,origin_e,U, data):
        x, edge_index = data.x, data.edge_index

        x = F.dropout(x, p=self.dropout, training=self.training)
        x = F.relu(self.lin1(x))
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.lin2(x)
        # x= self.m(x)

        if self.dprate == 0.0:
            x = self.prop1(x,origin_e,U)
            return F.log_softmax(x, dim=1)
        else:
            x = F.dropout(x, p=self.dprate, training=self.training)
            x = self.prop1(x, origin_e,U)
            return F.log_softmax(x, dim=1)

class GCN_Net(torch.nn.Module):
    def __init__(self, dataset, args, device: str = "cpu"):
        super(GCN_Net, self).__init__()
        # CHANGED FROM NUM_FEATURES TO NUM_NODE_FEATURES
        self.conv1 = GCNConv(dataset.num_node_features, args.hidden)
        self.conv2 = GCNConv(args.hidden, dataset.num_classes)
        if device == "cuda":
            self.conv1.cuda()
            self.conv2.cuda()
        self.dropout = args.dropout

    def reset_parameters(self):
        self.conv1.reset_parameters()
        self.conv2.reset_parameters()

    def forward(self, data, *args, **kwargs):
        x, edge_index = data.x, data.edge_index
        x = F.relu(self.conv1(x, edge_index))
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.conv2(x, edge_index)
        return F.log_softmax(x, dim=1)


class ChebNet(torch.nn.Module):
    def __init__(self, dataset, args, device: str = "cpu"):
        super(ChebNet, self).__init__()
        self.conv1 = ChebConv(dataset.num_node_features, 32, K=2)
        self.conv2 = ChebConv(32, dataset.num_classes, K=2)
        self.dropout = args.dropout
        if device == "cuda":
            self.conv1.cuda()
            self.conv2.cuda()

    def reset_parameters(self):
        self.conv1.reset_parameters()
        self.conv2.reset_parameters()

    def forward(self, data, *args, **kwargs):
        x, edge_index = data.x, data.edge_index
        x = F.relu(self.conv1(x, edge_index))
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.conv2(x, edge_index)
        return F.log_softmax(x, dim=1)


class GAT_Net(torch.nn.Module):
    def __init__(self, dataset, args):
        super(GAT_Net, self).__init__()
        self.conv1 = GATConv(
            dataset.num_node_features,
            args.hidden,
            heads=args.heads,
            dropout=args.dropout)
        self.conv2 = GATConv(
            args.hidden * args.heads,
            dataset.num_classes,
            heads=args.output_heads,
            concat=False,
            dropout=args.dropout)
        self.dropout = args.dropout

    def reset_parameters(self):
        self.conv1.reset_parameters()
        self.conv2.reset_parameters()

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = F.elu(self.conv1(x, edge_index))
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.conv2(x, edge_index)
        return F.log_softmax(x, dim=1)


class APPNP_Net(torch.nn.Module):
    def __init__(self, dataset, args, device: str = "cpu"):
        super(APPNP_Net, self).__init__()
        self.lin1 = Linear(dataset.num_node_features, args.hidden)
        self.lin2 = Linear(args.hidden, dataset.num_classes)
        self.prop1 = APPNP(args.K, args.alpha)
        if device == "cuda":
            self.lin1.cuda()
            self.lin2.cuda()
            self.prop1.cuda()
        self.dropout = args.dropout

    def reset_parameters(self):
        self.lin1.reset_parameters()
        self.lin2.reset_parameters()

    def forward(self, data, *args, **kwargs):
        x, edge_index = data.x, data.edge_index
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = F.relu(self.lin1(x))
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.lin2(x)
        x = self.prop1(x, edge_index)
        return F.log_softmax(x, dim=1)


class MLP(torch.nn.Module):
    def __init__(self, dataset,args):
        super(MLP, self).__init__()

        self.lin1 = Linear(dataset.num_node_features, args.hidden)
        self.lin2 = Linear(args.hidden, dataset.num_classes)
        self.dropout =args.dropout

    def reset_parameters(self):
        self.lin1.reset_parameters()
        self.lin2.reset_parameters()

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = F.relu(self.lin1(x))
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.lin2(x)

        return F.log_softmax(x, dim=1)


def JacobiConv(L, xs, origin_e, alphas, a=1.0, b=1.0, l=-1.0, r=1.0):
    '''
    Jacobi Bases
    '''
    one_pad = torch.ones(origin_e.shape).cuda()
    if L == 0: 
        return one_pad * alphas[0]
    elif L == 1:
        coef1 = (a - b) / 2 - (a + b + 2) / 2 * (l + r) / (r - l)
        coef1 *= alphas[1]
        coef2 = (a + b + 2) / (r - l)
        coef2 *= alphas[1]
        one_pad = torch.ones(origin_e.shape).cuda()
        return coef1 * one_pad + coef2 * (origin_e) * xs[-1]
    else:
        coef_l = 2 * L * (L + a + b) * (2 * L - 2 + a + b)
        coef_lm1_1 = (2 * L + a + b - 1) * (2 * L + a + b) * (2 * L + a + b - 2)
        coef_lm1_2 = (2 * L + a + b - 1) * (a**2 - b**2)
        coef_lm2 = 2 * (L - 1 + a) * (L - 1 + b) * (2 * L + a + b)
        tmp1 = alphas[L - 1] * (coef_lm1_1 / coef_l)
        tmp2 = alphas[L - 1] * (coef_lm1_2 / coef_l)
        tmp3 = alphas[L - 1] * alphas[L - 2] * (coef_lm2 / coef_l)
        tmp1_2 = tmp1 * (2 / (r - l)) 
        tmp2_2 = tmp1 * ((r + l) / (r - l)) + tmp2
        nx = tmp1_2 * (origin_e * xs[-1]) - tmp2_2 * xs[-1]
        nx -= tmp3 * xs[-2]
        out = nx
        return out


class Jacobi_prop(MessagePassing):
    '''
    Propagation class for Jacobi
    '''

    def __init__(self, K, alpha, Init, out_channels, a, b, Gamma=None, bias=True, **kwargs):
        super(Jacobi_prop, self).__init__(aggr='add', **kwargs)
        self.K = K
        self.Init = Init
        # self.alpha = alpha
        self.basealpha = alpha

        assert Init in ['SGC', 'PPR', 'NPPR', 'Random', 'WS']
        if Init == 'SGC':
            # SGC-like
            TEMP = 0.0*np.ones(K+1)
            TEMP[-1] = 1.0
        elif Init == 'PPR':
            # PPR-like
            TEMP = alpha*(1-alpha)**np.arange(K+1)
            TEMP[-1] = (1-alpha)**K
        elif Init == 'NPPR':
            # Negative PPR
            TEMP = (alpha)**np.arange(K+1)
            TEMP = TEMP/np.sum(np.abs(TEMP))
        elif Init == 'Random':
            # Random
            bound = np.sqrt(3/(K+1))
            TEMP = np.random.uniform(-bound, bound, K+1)
            TEMP = TEMP/np.sum(np.abs(TEMP))
        elif Init == 'WS':
            # Specify Gamma
            TEMP = Gamma

        self.temp =  ParameterList([
            Parameter(torch.ones(self.K + 1),
                         requires_grad=True) for _ in range(out_channels)
        ])

        self.a = a
        self.b = b

    def reset_parameters(self):
        torch.nn.init.zeros_(self.temp)
        for k in range(self.K+1):
            self.temp.data[k] = self.alpha*(1-self.alpha)**k
        self.temp.data[-1] = (1-self.alpha)**self.K

    def forward(self, x, origin_e, U):
        # Use polynomials of eigenvalues instead of time-consuming matrix polynomials
        out_list = []
        for i in range(x.shape[1]):
            alphas = [self.basealpha * torch.tanh(x) for x in self.temp[i]]
            xs = [JacobiConv(0, None, origin_e, alphas, a=self.a, b=self.b)]
            for k in range(self.K):
                xs.append(JacobiConv(k+1, xs, origin_e, alphas, a=self.a, b=self.b))
            out = sum(xs)
            out_list.append(out.unsqueeze(1))

        # Incorporate a combination with a coefficient for each channel and each layer combination
        recomposed_z = U @ (torch.cat([x for x in out_list], dim=1) * (U.t() @ x))

        return recomposed_z


    def message(self, x_j, norm):
        return norm.view(-1, 1) * x_j

    def __repr__(self):
        return '{}(K={}, temp={})'.format(self.__class__.__name__, self.K,
                                          self.temp)

    def message(self, x_j, norm):
        return norm.view(-1, 1) * x_j

    def __repr__(self):
        return '{}(K={}, temp={})'.format(self.__class__.__name__, self.K,
                                          self.temp)
    

class JacobiGNN(torch.nn.Module):
    def __init__(self, dataset, args, device: str = "cpu"):
        super(JacobiGNN, self).__init__()

        # Define the map from the shape of x to the number of hidden layers
        self.lin1 = Linear(dataset.x.shape[1], args.hidden)
        # Define the map from the number of hidden layers to the overall number of classes in x (logits)
        self.lin2 = Linear(args.hidden, dataset.num_classes)
        # self.lin = Linear(dataset.x.shape[1], dataset.num_classes)

        self.prop1 = Jacobi_prop(K=args.K,
                                 alpha=args.alpha, 
                                 Init=args.Init, 
                                 out_channels=dataset.num_classes,
                                 a=args.a,
                                 b=args.b)

        if device == "cuda":
            self.lin1.cuda()
            self.lin2.cuda()
            # self.lin.cuda()
            self.prop1.cuda()

        self.Init = args.Init
        self.dprate = args.dprate
        self.dropout = args.dropout
        self.a = args.a
        self.b = args.b
        self.dpb = args.dpb
        self.dpt = args.dpt


    def reset_parameters(self):
        self.prop1.reset_parameters()


    def forward(self, origin_e, U, data, *args, **kwargs):
        x, edge_index = data.x, data.edge_index

        # Apply a dropout for regularization (randomly zeros entries)
        x = F.dropout(x, p=self.dpb, training=self.training)
        # Rectify x to positive after a transform to the hidden layer
        x = F.relu(self.lin1(x))
        # x = self.lin(x)
        # Apply dropout again
        x = F.dropout(x, p=self.dpt, training=self.training)
        # Transform to output layer/classifications (logits for each class, 1 per node)
        x = self.lin2(x)
        if self.dprate == 0.0:
            # Run prop1 to go to next layer
            x = self.prop1(x, origin_e, U, a=self.a, b=self.b)
            return F.log_softmax(x, dim=1)
        else:
            # Run an additional dropout
            x = F.dropout(x, p=self.dprate, training=self.training)
            x = self.prop1(x, origin_e,U)
            return F.log_softmax(x, dim=1)
        