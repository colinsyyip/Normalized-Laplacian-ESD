import numpy as np
import random
import torch
import argparse
import time
import sys

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def parse_args():
    parser = argparse.ArgumentParser(description='')
    # Data settings
    parser.add_argument('--dataset', type=str,default='pubmed')
    parser.add_argument('--split', type=str, default="dense")
    parser.add_argument('--method', type=str, default="A_sym_uniform_A_sym")  # A_sym,uniform_A_sym,A_sym_uniform_A_sym
    parser.add_argument('--device', type=int, default=0)

    # Train settings
    parser.add_argument('--repeat', type=int, default=10)
    parser.add_argument('--test', action='store_true')

    # Optuna Settings
    parser.add_argument('--optruns', type=int, default=1)
    parser.add_argument('--path', type=str, default="")
    parser.add_argument('--name', type=str, default="opt")

    # Model Settings
    parser.add_argument('--detach', action='store_true')
    parser.add_argument('--savemodel', action='store_true')
    parser.add_argument('--power', action="store_true")
    parser.add_argument('--cheby', action="store_true")
    parser.add_argument('--legendre', action="store_true")
    parser.add_argument('--bern', action="store_true")
    parser.add_argument('--sole', action="store_true")
    parser.add_argument('--fixalpha', action="store_true")
    parser.add_argument('--multilayer', action="store_true")
    parser.add_argument('--resmultilayer', action="store_true")

    args = parser.parse_args()
    print("args = ", args)
    return args


def train(corrected_e,ones,U,optimizer, model, ds, loss_fn, i):
    
    optimizer.zero_grad()
    model.train()
    pred = model(corrected_e,ones,U,ds.edge_index, ds.edge_attr, ds.mask)
    pred = pred[ds.mask]
    loss = loss_fn(pred, ds.y)
    loss.backward()
    optimizer.step()
    return 



@torch.no_grad()
def test(origin_e,ones,U,model, ds, metrics):
    model.eval()
    pred = model(origin_e,ones,U,ds.edge_index, ds.edge_attr, ds.mask)
    pred = pred[ds.mask]
    y = ds.y
    return metrics(pred.cpu().numpy(), y.cpu().numpy())

