# File for generating spectral distribution plots, such as for Figure 5

# Eigenvalue decomposition does not work on AMD machines, so we default to CPU processing
# Running this on Nvidia on GPU is possible.

# from local_amd_setup import local_setup

# device = 'cuda' if local_setup() else 'cpu'
device ='cpu'

import numpy as np
import matplotlib.pyplot as plt
import torch
from torch_geometric.data import Data
from torch_geometric.datasets import FakeDataset,FakeHeteroDataset
from torch.nn.init import orthogonal_
from torch_geometric.utils import *
import networkx as nx
import datasets

np.set_printoptions(formatter={'float': '{:0.5f}'.format})
path = 'data'
dataset_list = ['cora', 'citeseer', 'pubmed', 'computers', 'photo', 'texas', 'cornell', 'chameleon', 'film', 'squirrel' ]

def draw_density(dataset):
    """
    Plot the empirical spectral distribution of a given dataset
    """
    data = datasets.load_dataset(dataset, 'dense')
    edge_index = to_undirected(data.edge_index)
    A =  to_dense_adj(edge_index).squeeze().to(device)
    n = A.shape[0]
    data.edge_index, data.edge_attr = remove_self_loops(data.edge_index)
    I = torch.eye(A.shape[0]).to(device) 
    degree = torch.diag(A.sum(-1)**(-0.5)).to(device)
    degree[torch.isinf(degree)] = 0.
    L_sym = I - degree.mm(A.mm(degree))
    p = A.sum() / (n * n)
    print([n, p, np.log(n)])
    e, _ = torch.linalg.eigh(L_sym, UPLO='U')
    dri = torch.histc(e, 100, -0.001, 2.001) / L_sym.shape[0]
    x = np.arange(0, 2, 0.02)
    plt.plot(x, dri.cpu().numpy())
    plt.tick_params(labelsize=17)
    plt.xlabel('λ',fontsize=18)
    plt.ylabel('Density',fontsize=18)
    plt.savefig(dataset+'_dist.pdf', bbox_inches='tight')
    print(e.numpy())


def draw_uniform_corrected_density(dataset, beta: float = 0.5):
    """
    Plot the empirical spectral distribution of a given dataset after uniform eigenvalue correction at a given value of beta.

    Uniform correction methodology outlined in "Improving Expressive Power of Spectral Graph Neural Networks with Eigenvalue Correction"
    """
    data = datasets.load_dataset(dataset, 'dense')
    edge_index = to_undirected(data.edge_index)
    A =  to_dense_adj(edge_index).squeeze().to(device)
    n = A.shape[0]
    data.edge_index, data.edge_attr = remove_self_loops(data.edge_index)
    print(contains_self_loops(data.edge_index))
    I = torch.eye(A.shape[0]).to(device) 
    degree = torch.diag(A.sum(-1)**(-0.5)).to(device)
    degree[torch.isinf(degree)] = 0.
    L_sym = I - degree.mm(A.mm(degree))
    e = torch.linalg.eigvalsh(L_sym, UPLO='U')
    corrected_e = torch.FloatTensor(np.linspace(0, 2, n)).to(device)
    corrected_e = beta * e + (1 - beta) * corrected_e
    dri = torch.histc(corrected_e, 100, -0.001, 2.001) / L_sym.shape[0]
    x = np.arange(0, 2, 0.02)
    plt.plot(x, dri.cpu().numpy())
    plt.tick_params(labelsize=17)
    plt.xlabel('µ',fontsize=18)
    plt.ylim(0, 0.06)
    plt.ylabel('Density',fontsize=18)
    plt.savefig(dataset+'_uniform_corrected_band_test.pdf', bbox_inches='tight')
    print(e.numpy())


def sim_homog_density(n:int,
                      center: bool = True,
                      scale: bool = True,
                      correction: str = "none",
                      beta: float = 0.5,
                      p: float = 0.5):
    """
    Simulate a density and execute one or all correction methodologies, including no correction to see the change in shape. 
    """
    if correction not in ("none", "uniform", "degree_based", "all"):
        raise ValueError("correction is not a valid value.")
    A = torch.distributions.Bernoulli(p).expand([n, n]).sample()
    A.fill_diagonal_(0)
    i, j = torch.triu_indices(n, n)
    A.T[i, j] = A[i, j]
    I = torch.zeros([n, n])
    I.fill_diagonal_(1)
    D_vec = A.sum(1)
    D_mask = torch.diag(torch.ones_like(D_vec))
    D = torch.zeros([n, n])
    D = D_mask * torch.diag(D_vec) + (1. - D_mask) * D
    inv_D = D ** (-0.5)
    inv_D[torch.isinf(inv_D)] = 0.
    L = I - (inv_D).mm(A).mm(inv_D)

    bound_value = 2/torch.sqrt(torch.Tensor([(n * p)]))

    L_mod_var = [1 - bound_value,
                 1 + bound_value]
    label_text = "Degree Based Mass Bound"
    if center and scale:
        L_mod_var = [0 - 2,
                     0 + 2]
        L_mod = torch.sqrt(torch.Tensor([n * p/(1 - p)])) * (I - L)
        label_text = "Semicircle Mass Bound"
    elif center:
        L_mod_var = [0 - bound_value,
                     0 + bound_value]
        L_mod = (I - L)
    elif scale:
        L_mod = torch.sqrt(torch.Tensor([n * p/(1 - p)])) * L
    else:
        L_mod = L
    
    e, _ = torch.linalg.eigh(L_mod, UPLO='U')

    x = np.arange(0, 2, 0.02)

    if correction == "all":
        uncorrected_e = e
        uniform_corrected_e = torch.FloatTensor(np.linspace(0, 2, n)).to(device)
        uniform_corrected_e = beta * e + (1 - beta) * uniform_corrected_e
        e_mask = (e >= L_mod_var[0]) & (e <= L_mod_var[1])
        bounded_e = e[e_mask]
        n_e_mask = len(bounded_e)
        corrected_bounded_e = torch.FloatTensor(np.linspace(L_mod_var[0], L_mod_var[1], n_e_mask).reshape(-1)).to(device)
        corrected_bounded_e = beta * bounded_e + (1 - beta) * corrected_bounded_e
        degree_corrected_e = e
        degree_corrected_e[e_mask] = corrected_bounded_e

        uncorrected_dri = torch.histc(uncorrected_e, 100, -0.001, 2.001) / n
        uniform_corrected_dri = torch.histc(uniform_corrected_e, 100, -0.001, 2.001) / n
        degree_corrected_dri = torch.histc(degree_corrected_e, 100, -0.001, 2.001) / n

        plt.plot(x, uncorrected_dri.cpu().numpy(),
                 label="Uncorrected",
                 color="cornflowerblue")
        plt.plot(x, uniform_corrected_dri.cpu().numpy(),
                 linestyle='--',
                 label="Uniform EC",
                 color="limegreen")
        plt.plot(x, degree_corrected_dri.cpu().numpy(),
                 linestyle=':',
                 label="Degree Based EC",
                 color="firebrick")

    else:
        if correction == "none":
            corrected_e = e
        elif correction == "uniform":
            corrected_e = torch.FloatTensor(np.linspace(0, 2, n)).to(device)
            corrected_e = beta * e + (1 - beta) * corrected_e
        else:
            e_mask = (e >= L_mod_var[0]) & (e <= L_mod_var[1])
            bounded_e = e[e_mask]
            n_e_mask = len(bounded_e)
            corrected_bounded_e = torch.FloatTensor(np.linspace(L_mod_var[0], L_mod_var[1], n_e_mask).reshape(-1)).to(device)
            corrected_bounded_e = beta * bounded_e + (1 - beta) * corrected_bounded_e
            corrected_e = e
            corrected_e[e_mask] = corrected_bounded_e
        dri = torch.histc(corrected_e, 100, -0.001, 2.001) / n
        plt.plot(x, dri.cpu().numpy())

    plt.tick_params(labelsize=17)
    plt.axvline(L_mod_var[0],
                color='red',
                linewidth=0.2)
    plt.axvline(L_mod_var[1],
                color='red',
                linewidth=0.2,
                label=label_text)
    plt.legend(loc='upper center',
        bbox_to_anchor=(0.5, 1.11),
        ncol=4)
    plt.xlabel('λ',fontsize=18)
    plt.ylabel('Density',fontsize=18)
    plt.title("n=%s, p=%s Homogeneous ER, Correction: %s\n\n" % (n, p, correction.upper()))
    if correction == "none":
        plt.savefig('sim_homog_band_test.pdf',
                            bbox_inches='tight')
    else:
        plt.savefig('sim_homog_band_test_%scorrection.pdf' % correction,
                    bbox_inches='tight')


def multiplicity_check(e,
                       rtol: float = 1e-5,
                       atol: float = 1e-8):
    """
    Check for the multiplicities of a vector of eigenvalues, up to a given relative and absolute tolerance 
    due to approximation in numerical solution finding.
    """
    e = e.sort()[0]
    e_diffs = torch.diff(e)

    approx_thresh = atol + rtol * e[:-1].abs()
    diff_mask = e_diffs < approx_thresh

    e_multiplicities = {}
    mult_counter = 0
    fix_e_i = e[0]

    for e_i, i in zip(e[1:], range(len(e[1:]))):
        mult_counter += 1
        if not(diff_mask[i]):
            e_multiplicities[fix_e_i] = mult_counter
            mult_counter = 0      
            fix_e_i = e_i
        else:
            continue   

    return e_multiplicities


def homog_correction_comp(dataset,
                          center: bool = False,
                          scale: bool = False,
                          beta: float = 0.5,
                          beta_2: float = 0.5):
    """
    Correct the empiricial spectral distribution of a given dataset, ranging from one correction to all corrections.
    """
    data = datasets.load_dataset(dataset, 'dense')
    edge_index = to_undirected(data.edge_index)
    A =  to_dense_adj(edge_index).squeeze().to(device)
    n = A.shape[0]
    p = A.sum() / (n * n)
    I = torch.eye(A.shape[0]).to(device) 
    degree = torch.diag(A.sum(-1)**(-0.5)).to(device)
    degree[torch.isinf(degree)] = 0.
    L = I - degree.mm(A.mm(degree))

    A_e, _ = torch.linalg.eigh((A - p) / np.sqrt(n * p), UPLO='U')
    A_x = np.arange(-5, 5, 0.02)
    A_e_density = torch.histc(A_e, 500, -5.001, 5.001) / n

    plt.plot(A_x, A_e_density.cpu().numpy(),
            color="cornflowerblue")
    plt.tick_params(labelsize=17)
    plt.xlabel('λ',fontsize=18)
    plt.ylabel('Density',fontsize=18)
    plt.title("Dataset: %s, n=%s, p=%s, Adjacency Spectra" % (dataset, n, 
                                                               round(float(p), 5)))
    plt.savefig('%s_normalized_adjacency.pdf' % (dataset),
                bbox_inches='tight')
    plt.clf()
    
    bound_value = 2/torch.sqrt(torch.Tensor([(n * p)]))

    L_mod_var = [1 - bound_value,
                 1 + bound_value]
    label_text = "Degree Based Mass Bound"
    if center and scale:
        L_mod_var = [0 - 2,
                     0 + 2]
        L_mod = torch.sqrt(torch.Tensor([n * p/(1 - p)])) * (I - L)
        label_text = "Semicircle Mass Bound"
    elif center:
        L_mod_var = [0 - bound_value,
                     0 + bound_value]
        L_mod = (I - L)
    elif scale:
        L_mod = torch.sqrt(torch.Tensor([n * p/(1 - p)])) * L
    else:
        L_mod = L
    
    e, _ = torch.linalg.eigh(L_mod, UPLO='U')

    torch.save(e, "data/eigenvalues/%s_lmod" % dataset)

    x = np.arange(0, 2, 0.02)

    print("Homog. Correction Range %s" % L_mod_var)

    uncorrected_e = e.clone().detach()
    uniform_corrected_e = torch.FloatTensor(np.linspace(0, 2, n)).to(device)
    uniform_corrected_e = beta * e + (1 - beta) * uniform_corrected_e
    e_mask = (e >= L_mod_var[0]) & (e <= L_mod_var[1])
    bounded_e = e[e_mask]
    n_e_mask = len(bounded_e)
    corrected_bounded_e = torch.FloatTensor(np.linspace(L_mod_var[0], 
                                                        L_mod_var[1], 
                                                        n_e_mask).reshape(-1)).to(device)
    corrected_bounded_e = beta_2 * bounded_e + (1 - beta_2) * corrected_bounded_e
    degree_corrected_e = e.clone().detach()
    degree_corrected_e[e_mask] = corrected_bounded_e

    uncorrected_multiplicities = list(multiplicity_check(uncorrected_e).values())
    uniform_EC_multiplicities = list(multiplicity_check(uniform_corrected_e).values())
    degree_corrected_EC_multiplicities = list(multiplicity_check(degree_corrected_e).values())

    print("Uncorrected Max Mult: %s" % max(uncorrected_multiplicities))
    print("Uniform EC Max Mult: %s" % max(uniform_EC_multiplicities))
    print("Degree EC Max Mult: %s" % max(degree_corrected_EC_multiplicities))

    uncorrected_dri = torch.histc(uncorrected_e, 100, -0.001, 2.001) / n
    uniform_corrected_dri = torch.histc(uniform_corrected_e, 100, -0.001, 2.001) / n
    degree_corrected_dri = torch.histc(degree_corrected_e, 100, -0.001, 2.001) / n

    plt.plot(x, uncorrected_dri.cpu().numpy(),
                label="Uncorrected (%s)" % max(uncorrected_multiplicities),
                color="cornflowerblue")
    plt.plot(x, uniform_corrected_dri.cpu().numpy(),
                linestyle='--',
                label="Uniform EC (%s)" % max(uniform_EC_multiplicities),
                color="limegreen")
    plt.plot(x, degree_corrected_dri.cpu().numpy(),
                linestyle=':',
                label="Degree Based EC (%s)" % max(degree_corrected_EC_multiplicities),
                color="firebrick")

    plt.tick_params(labelsize=17)
    plt.axvline(L_mod_var[0],
                color='red',
                linewidth=0.2)
    plt.axvline(L_mod_var[1],
                color='red',
                linewidth=0.2,
                label=label_text)
    plt.legend(loc='upper center',
        bbox_to_anchor=(0.5, 1.11),
        ncol=4)
    plt.xlabel('λ',fontsize=18)
    plt.ylabel('Density',fontsize=18)
    plt.title("Dataset: %s, n=%s, p=%s, Correction: %s\n\n" % (dataset, n, 
                                                               round(float(p), 5), 
                                                               "all".upper()))
    plt.savefig('%s_homog_correction_comp.pdf' % (dataset),
                bbox_inches='tight')
    plt.clf()


if __name__ == "__main__":
    datasets_names = [
                'cora', 'citeseer', 'pubmed', 
                'computers', 'photo', 'texas', 'cornell', 'chameleon',
                'actor', 'squirrel'
    ]
    [homog_correction_comp(x) for x in datasets_names]