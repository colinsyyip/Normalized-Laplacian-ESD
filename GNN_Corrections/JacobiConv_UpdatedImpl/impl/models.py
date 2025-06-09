import torch
import torch.nn as nn
from torch import Tensor
from typing import Iterable
import torch.nn.functional as F
import sys
import time



class Seq(nn.Module):
    '''
    An extension of nn.Sequential.
    Args:
        modlist an iterable of modules to add.
    '''
    def __init__(self, modlist: Iterable[nn.Module]):
        super().__init__()
        self.modlist = nn.ModuleList(modlist)

    def forward(self, *args, **kwargs):
        out = self.modlist[0](*args, **kwargs)
        for i in range(1, len(self.modlist)):
            out = self.modlist[i](out)
        return out


class TensorMod(nn.Module):
    '''
    An mod which forwards a Tensor
    Args:
        x: Tensor
    '''
    def __init__(self, x: Tensor):
        super().__init__()
        self.x = nn.parameter.Parameter(x, requires_grad=False)

    def forward(self, *args, **kwargs):
        return self.x


class ResBlock(nn.Module):
    '''
    A block building residual connection.
    '''
    def __init__(self, mod: nn.Module):
        super().__init__()
        self.mod = mod

    def forward(self, x):
        return x + self.mod(x)


class Combination(nn.Module):
    '''
    A mod combination the bases of polynomial filters.
    Args:
        channels (int): number of feature channels.
        depth (int): number of bases to combine.
        sole (bool): whether or not use the same filter for all output channels.
    '''
    def __init__(self, channels: int, depth: int, sole=False):
        super().__init__()
        if sole:
            self.comb_weight = nn.Parameter(torch.ones((1, depth, 1)))
        else:
            self.comb_weight = nn.Parameter(torch.ones((1, depth, channels)))

    def forward(self, x: Tensor, pos: Tensor):
        '''
        x: node features filtered by bases, of shape (number of nodes, depth, channels).
        '''
        # a = time.time()
        # # x = x[pos]
        # b = time.time()
        # print('pos time:',b-a)
        x = x * self.comb_weight
        x = torch.sum(x, dim=1)[pos]
        # print('comb time:',time.time()-b)
        return x


class Gmodel(nn.Module):
    '''
    A framework for GNN models.
    Args:
        embs (nn.Module): produce node features.
        conv (nn.Module): do message passing.
        comb (nn.Module): combine bases to produce the filter function.
    '''
    def __init__(self, emb: nn.Module, conv: nn.Module, comb: nn.Module):
        super().__init__()
        self.emb = emb
        self.conv = conv
        self.comb = comb

    def forward(self, edge_index: Tensor, edge_weight: Tensor, pos: Tensor):
        '''
        pos: mask of node whose embeddings is needed.
        '''
        nemb = self.comb(self.conv(self.emb(), edge_index, edge_weight),pos)
        # print(nemb.size())
        # print(type(nemb))
        # print(nemb)
        # print(pos.flatten())
        # sys.exit()
        # start = time.time()
        # result = nemb[pos]
        # print('nemb time:',time.time()-start)
        return nemb
