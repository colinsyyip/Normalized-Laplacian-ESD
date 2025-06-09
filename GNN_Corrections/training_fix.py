# File to tune beta for all three filters and across all correction strategies.

from local_amd_setup import local_setup

device = 'cuda' if local_setup() else 'cpu'

import argparse
from utils import random_planetoid_splits, net_arg_converter
from models import *
from torch_geometric.utils import *
import torch
import torch.nn.functional as F
from torch.optim import Adam
from tqdm.auto import tqdm
import random
import seaborn as sns
import numpy as np
import time
import pandas as pd
from JacobiConv_UpdatedImpl.impl import models as jmodels
from functools import partial
import datasets


SEEDS=[1941488137,4198936517,983997847,4023022221,4019585660,2108550661,1648766618,629014539,3212139042,2424918363]

def net_map(net_name):
    if net_name == 'GCN':
        Net = GCN_Net
    elif net_name == 'GAT':
        Net = GAT_Net
    elif net_name == 'APPNP':
        Net = APPNP_Net
    elif net_name == 'ChebNet':
        Net = ChebNet
    elif net_name == 'GPRGNN':
        Net = GPRGNN
    elif net_name == 'BernNet':
        Net = BernNet
    elif net_name =='MLP':
        Net = MLP
    elif net_name == "JacobiConv":
        Net = JacobiGNN

    return Net


def RunExp(origin_e, U, dataset, data, net, net_name, percls_trn, val_lb, device,
           weight_decay: float = 0.0005,
           lr: float = 0.01,
        #    Bern_lr: float = 0.01,
           split_seed: int = 2108550661,
           epochs: int = 1000,
           dprate: float = 0.5,
           alpha: float = 0.1,
           early_stopping: int = 200,
           net_params: dict = None,
           img_filter: bool = False,
           n_repeat: int = 1,
           conv_layer: int = 10,
           *args, **kwargs):
    """
    Model fit and evaluation function for a specific Net.
    """

    def train(origin_e,U,model, optimizer, data, dprate, loss_fn, x_idx: int = None):
        model.train()
        optimizer.zero_grad()
        if x_idx is None:
            out = model(origin_e = origin_e, 
                        U = U, 
                        data = data)[data.train_mask]
        else:
            out = model(origin_e = origin_e, 
                        U = U, 
                        data = data)
        # out = model(data)[data.train_mask]
        loss = loss_fn(out, data.y[data.train_mask])
        reg_loss=None
        loss.backward()
        optimizer.step()
        del out

    def test(origin_e,U,model, data, loss_fn):
        model.eval()
        logits, accs, losses, preds = model(origin_e = origin_e,
                                            U = U, 
                                            data = data), [], [], []
        for mask in [data.train_mask, data.val_mask, data.test_mask]:
            pred = logits[mask].max(1)[1]
            acc = pred.eq(data.y[mask]).sum().item() / mask.sum().item()
            loss = loss_fn(model(origin_e = origin_e,
                                    U = U,
                                    data = data)[mask], data.y[mask])
            preds.append(pred.detach().cuda())
            accs.append(acc)
            losses.append(loss.detach().cuda())
        return accs, preds, losses


    # Fit a dummy specified Net using the provided dataset and args
    net_fmted_args = net_arg_converter(net_params)
    tmp_net = net(dataset = dataset, args = net_fmted_args, device = device)

    # Randomly split dataset given the provided seeds and a lower bound integer for validation set size
    if img_filter:
        from JacobiConv.impl import GDataset, utils as jutils
        data = GDataset.GDataset(*dataset.get_split("valid"))
    else:
        permute_masks = random_planetoid_splits
        data = permute_masks(data, dataset.num_classes, percls_trn, val_lb, split_seed)
    
    # Send net and permuted/split data masks to selected torch device
    model, data = tmp_net.to(device), data.to(device)

    # Define stochastic optimizer with specific parameters depending on the selected net
    loss_func = F.nll_loss
    if net_name=='GPRGNN':
        # Lin params are weights (W), and bias (+/- b), and initialized uniformly depending on the in_feature dimensions
        optimizer = torch.optim.Adam([{ 'params': model.lin1.parameters(), 'weight_decay': net_fmted_args.weight_decay, 'lr': net_fmted_args.lr},
                                      {'params': model.lin2.parameters(), 'weight_decay': net_fmted_args.weight_decay, 'lr': net_fmted_args.lr},
                                      {'params': model.prop1.parameters(), 'weight_decay': 0.00, 'lr': net_fmted_args.lr}])
    elif net_name =='BernNet':
        optimizer = torch.optim.Adam([{'params': model.lin1.parameters(),'weight_decay': net_fmted_args.weight_decay, 'lr': net_fmted_args.lr},
                                      {'params': model.lin2.parameters(), 'weight_decay': net_fmted_args.weight_decay, 'lr': net_fmted_args.lr},
                                      {'params': model.prop1.parameters(), 'weight_decay': 0.0, 'lr': net_fmted_args.Bern_lr}])
    elif net_name == "JacobiConv":
        optimizer = torch.optim.Adam([{'params': model.lin1.parameters(), 'weight_decay': net_fmted_args.wd2, 'lr': net_fmted_args.lr2},
                                      {'params': model.lin2.parameters(), 'weight_decay': net_fmted_args.wd2, 'lr': net_fmted_args.lr2},
                                      {'params': model.prop1.parameters(), 'weight_decay': 0, 'lr': net_fmted_args.lr3}])
        loss_func = torch.nn.CrossEntropyLoss()
    else:
        optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    # Loop setup
    best_val_acc = test_acc = 0
    best_val_loss = float('inf')
    val_loss_history = []
    val_acc_history = []
    theta = alpha

    time_run=[]
   
    # Run for specified number of epochs
    for epoch in range(epochs):
        t_st=time.time()
        # Run train subfunction using equidistant merged eigenvalues. dprate is dropout for propagation layer (not used???)
        train(origin_e, U, model, optimizer, data, dprate, loss_fn = loss_func)
        time_epoch=time.time()-t_st  # Each epoch train times
        time_run.append(time_epoch)

        # Run test subfunction to retrieve accuracies, predictions, and negative loglikelihood loss
        [train_acc, val_acc, tmp_test_acc], preds, [train_loss, val_loss, tmp_test_loss] = test(origin_e,U,model, data, loss_fn = loss_func)

        # Select new best depending on loss comparison
        if val_loss < best_val_loss:
            best_val_acc = val_acc
            best_val_loss = val_loss
            test_acc = tmp_test_acc
            if net =='BernNet':
                TEST = tmp_net.prop1.temp.clone()
                theta = TEST.detach().cuda()
                theta = torch.relu(theta).numpy()

        if epoch >= 0:
            val_loss_history.append(val_loss)
            val_acc_history.append(val_acc)
            if early_stopping > 0 and epoch > early_stopping:
                tmp = torch.tensor(
                    val_loss_history[-(early_stopping + 1):-1])
                if val_loss > tmp.mean().item():
                    # print('The sum of epochs:',epoch)
                    break
    
    return test_acc, best_val_acc, theta, time_run


def gnn_train(
        dataset_name: str,
        net_name: str,
        train_rate: float = 0.6,
        val_rate: float = 0.2,
        beta: float = 0.5,
        n_runs: int = 10,
        cuda_device: str = None,
        ev_correction: str = "none",
        ev_path: str = "data",
        *args,
        **kwargs
):
    """
    Generalized training function for multiple GNN classes. Requires eigenvalue and eigenvector to be generated and stored locally in directories labelled eigenvaluees and eigenvectors.

    Arguments:
        dataset_name [str]: Name of the dataset to be used. Must be one of 'cora', 'citeseer', 'pubmed', 
            'computers', 'photo', 'texas', 'cornell', 'chameleon', 'film', 'squirrel'.
        net_name [str]: Neural network algorithm name. Must be one of GCN, GAT, APPNP, ChebNet, GPRGNN, BernNet, MLP.
        train_rate [float]: Proportion of data to use in training. Must be bounded (0, 1)
        val_rate [float]: Proportion of data to use in validation. Must be bounded (0, 1)
        beta [float]: Control parameter between corrected proposed eigenvalue distribution and empirical distribution. Must be bounded (0, 1)
        n_runs [int]: Number of times to run the train/validate flow. 
        cuda_device [str]: CUDA device. Dependent on Pytorch version/hardware combination. In absence, uses CPU
        ev_correction [str]: Correction strategy for raw eigenvalues. Must be one of 'none', 'uniform', 'mu_f'.
        ev_path [str]: Directory containing both eigenvectors and eigenvalues directories for the corresponding dataset_name      
    """

    accepted_nets = ('GCN', 'GAT', 'APPNP', 'ChebNet', 'GPRGNN', 'BernNet', 'MLP', 'JacobiConv')

    if train_rate < 0 or train_rate > 1:
        raise ValueError("train_rate must be between 0 and 1.")
    
    if val_rate < 0 or val_rate > 1:
        raise ValueError("val_rate must be between 0 and 1.")
    
    if net_name not in accepted_nets:
        raise ValueError("net must be one of %s." % accepted_nets)

    # Load data from local tensor file
    savepath = f"data/{dataset_name.lower()}.pt"
    data = torch.load(savepath, map_location=device)

    # Get data shape and calculate the appropriate number of training/validation set data points
    n = data.x.shape[0]
    num_classes = torch.unique(data.y).shape[0]
    percls_trn = int(round(train_rate * len(data.y) / num_classes))
    val_lb = int(round(val_rate * len(data.y)))

    results = []
    time_results=[]

    # Retrieve eigenvalues/vectors 
    eigenvectors_path = "/".join([ev_path, 'eigenvectors', (dataset_name).lower()])
    eigenvalues_path = "/".join([ev_path, 'eigenvalues', (dataset_name).lower()])
    U = torch.load(eigenvectors_path).float().to(device)
    e = torch.load(eigenvalues_path).float().to(device)

    ev_correction_options = ['none', 'uniform', 'mu_f']    
    
    # Perform specified correction
    if ev_correction == "none":
        corrected_e = e
    elif ev_correction == "uniform":
        corrected_e = torch.FloatTensor(np.linspace(-1, 1, n)).to(device)
        corrected_e = beta * e + (1 - beta) * corrected_e
    elif ev_correction == "mu_f":
        edge_index = to_undirected(data.edge_index)
        A =  to_dense_adj(edge_index).squeeze().to(device)
        n = A.shape[0]
        p = A.sum() / (n * n)
        I = torch.eye(A.shape[0]).to(device) 
        degree = torch.diag(A.sum(-1)**(-0.5)).to(device)
        degree[torch.isinf(degree)] = 0.
        L = I - degree.mm(A.mm(degree))
        bound_value = (2/torch.sqrt(torch.Tensor([(n * p)]))).to(device)

        L_mod_var = [1 - bound_value,
                    1 + bound_value]
        
        lmod_e = torch.load("data/eigenvalues/%s_lmod" % dataset).float().to(device)

        e_mask = (lmod_e >= L_mod_var[0]) & (lmod_e <= L_mod_var[1])
        bounded_e = lmod_e[e_mask]
        n_e_mask = len(bounded_e)
        corrected_bounded_e = torch.FloatTensor(np.linspace(L_mod_var[0].cpu(), 
                                                            L_mod_var[1].cpu(), 
                                                            n_e_mask).reshape(-1)).to(device)
        corrected_bounded_e = beta * bounded_e + (1 - beta) * corrected_bounded_e
        degree_corrected_e = lmod_e.clone().detach()
        degree_corrected_e[e_mask] = corrected_bounded_e
        # DOUBLE CHECK THIS!!! (i dont think this makes a diff)
        corrected_e = beta * e + (1 - beta) * degree_corrected_e
    else:
        raise ValueError("ev_correction must be one of %s." % ev_correction_options)
    
    net = net_map(net_name)
    
    # Over n_runs, run the specified model and store results
    # IF the dataset_name is one of the image filter tests, need to run across the full range of 50 and then take the iterations for each
    for RP in tqdm(range(n_runs), position=0, leave=True):
        seed = SEEDS[RP]
        test_acc, best_val_acc,theta_0, time_run = RunExp(origin_e = corrected_e,
                                                          U = U, 
                                                          dataset = data,
                                                          data = data, 
                                                          net = net, 
                                                          net_name = net_name,
                                                          percls_trn = percls_trn, 
                                                          val_lb = val_lb,
                                                          device = device, 
                                                          split_seed = seed,
                                                          *args, **kwargs)
        time_results.append(time_run)
        results.append([test_acc, best_val_acc, theta_0])

    # Aggregate diagnostic information
    run_sum=0
    epochsss=0
    for i in time_results:
        run_sum+=sum(i)
        epochsss+=len(i)

    # Print diagnostics
    print("each run avg_time:",run_sum/(n_runs),"s")
    print("each epoch avg_time:",1000*run_sum/epochsss,"ms")

    test_acc_mean, val_acc_mean, _ = np.mean(results, axis=0) * 100
    test_acc_std = np.sqrt(np.var(results, axis=0)[0]) * 100

    values=np.asarray(results)[:,0]
    uncertainty=np.max(np.abs(sns.utils.ci(sns.algorithms.bootstrap(values,func=np.mean,n_boot=1000),95)-values.mean()))

    print(f'{net_name} (Correction: {ev_correction}) on dataset {dataset_name}, in {n_runs} repeated experiment:')
    print(f'test acc mean = {test_acc_mean:.4f} ± {uncertainty*100:.4f}  \t val acc mean = {val_acc_mean:.4f}')

    return [dataset_name, test_acc_mean, uncertainty, val_acc_mean]


if __name__ == "__main__":
    datasets_names = [
                'cora', 'citeseer', 'pubmed', 
                'computers', 'photo', 'texas', 'cornell', 'chameleon',
                'actor', 'squirrel'
    ]

    nn_options = ["GPRGNN", "EC-GPRGNN", "DEC-GPRGNN"
                  "BernNet", "EC-BernNet",  "DEC-BernNet", 
                  "JacobiConv", "EC-JacobiConv", "DEC-JacobiConv"]

    default_params = {
        "epochs": 1000,
        "lr": 0.01,
        "weight_decay": 0.0005,
        "early_stopping": 200,
        "hidden": 64,
        "dropout": 0.5,
        "train_rate": 0.6,
        "val_rate": 0.2,
        "K": 10,
        "alpha": 0.1,
        "dprate": 0.5,
        "Init": "PPR",
        "heads": 8,
        "output_heads": 1,
        "device": 0,
        "runs": 10,
        "Bern_lr": 0.01,
        "beta": 0.5,
        "optruns": 100
    }

    tuning_beta_range = np.linspace(0, 1, 21)
    is_tuning = False

    for nn in nn_options:
        results = []

        print(nn)
        if nn[:3] == "DEC":
            ev_correction = "mu_f"
            nn_parsed = nn.split("-")[1]
        elif nn[:2] == "EC":
            ev_correction = "uniform"
            nn_parsed = nn.split("-")[1]
        else:
            ev_correction = "none"
            nn_parsed = nn

        if nn_parsed == "JacobiConv":
            from realworld_params import jacobi_rw_params as net_specific_params
        elif nn_parsed == "BernNet":
            from realworld_params import bernnet_rw_params as net_specific_params
        elif nn_parsed == "GPRGNN": 
            from realworld_params import gprgnn_rw_params as net_specific_params
        
        for dataset in datasets_names:
            print(dataset)
            net_ds_params = net_specific_params[dataset]

            combined_params = default_params | net_ds_params

            if is_tuning and ev_correction == "mu_f":

                for beta in tuning_beta_range:
                    print(beta)

                    beta_result = gnn_train(dataset, nn_parsed,
                                            ev_correction = ev_correction,
                                            n_runs=10,
                                            beta=beta,
                                            net_params=combined_params)
                    
                    beta_result = [beta] + beta_result

                    results.append(beta_result)

                    results_df = pd.DataFrame(results,
                                        columns = ['beta', 'dataset', 'test_accuracy', 'uncertainty', 'validation_accuracy'])

                    # results_df.to_csv("%s_default_config_tuning_results.csv" % nn.replace("-", "_"),
                    #                 index = False)

            else:

                results.append(gnn_train(dataset, nn_parsed, 
                                        ev_correction = ev_correction, 
                                        n_runs=10,
                                        beta=combined_params['beta'][ev_correction],
                                        net_params=combined_params))
            
                results_df = pd.DataFrame(results,
                                        columns = ['dataset', 'test_accuracy', 'uncertainty', 'validation_accuracy'])

                # results_df.to_csv("%s_default_config_training_results.csv" % nn.replace("-", "_"),
                                # index = False)