from impl import metrics, PolyConv, models, GDataset
from impl import Model
import datasets
import torch
from torch.optim import Adam
# import optuna
import torch.nn as nn
import numpy as np
import seaborn as sns
import wandb
from bestHyperparams import realworld_params
import time
import sys

from impl import utils


def split():
    global baseG, trn_dataset, val_dataset, tst_dataset
    baseG.mask = datasets.split(baseG, split=args.split)
    trn_dataset = GDataset.GDataset(*baseG.get_split("train"))
    val_dataset = GDataset.GDataset(*baseG.get_split("valid"))
    tst_dataset = GDataset.GDataset(*baseG.get_split("test"))


def work(strategy:str='low1',
        conv_layer: int = 10,
         alpha: float = 0.2,
         lr1: float = 1e-3,
         lr2: float = 1e-3,
         lr3: float = 1e-3,
         wd1: float = 0,
         wd2: float = 0,
         wd3: float = 0,
         dpb=0.0,
         dpt=0.0,
         beta: float = 0,
         **kwargs):
    outs = []
    global device 
    
    eigenvectors_path = '/data/eigenvectors/'+args.dataset
    eigenvalues_path = '/data/eigenvalues/'+args.dataset
    n = baseG.num_nodes
    # k = 4000
    # gpu_tracker.track() 
    U = torch.load(eigenvectors_path).to(device)
    e = torch.load(eigenvalues_path).to(device)


    if strategy=='low1':
        corrected_e = torch.FloatTensor(np.linspace(-1, 1, n)).to(device)
    elif strategy=='low2':
        corrected_e = torch.FloatTensor(np.linspace(0, 1, n)).to(device)
    elif strategy=='low3':
        corrected_e = torch.FloatTensor(np.linspace(-1, 0, n)).to(device)
    elif strategy=='high':
        corrected_e = torch.FloatTensor(np.linspace(1, -1, n)).to(device)
    elif strategy=='rand':
        corrected_e = (2*torch.rand(n,1)-1).to(device).squeeze()
    else:
        raise 'no this strategy: '+strategy

    corrected_e = beta*e+(1-beta)*corrected_e
    ones = torch.ones(n).to(device)
    
    for rep in range(args.repeat):
        utils.set_seed(rep)
        split()
        gnn = Model.JacobiConv_model(None,baseG.x,dpb,dpt,output_channels,method=args.method,dataset=args.dataset,
                        depth=conv_layer,
                        alpha=alpha,
                        beta=beta,
                        fixed=args.fixalpha).to(device)
        optimizer = Adam([{
            'params': gnn.emb.parameters(),
            'weight_decay': wd1,
            'lr': lr1
        }, {
            'params': gnn.alphas.parameters(),
            'weight_decay': wd2,
            'lr': lr2
        }, {
            'params': gnn.comb_weight,
            'weight_decay': wd3,
            'lr': lr3

        }])
        val_score = 0
        early_stop = 0
        s = time.time()
        for i in range(1000):
            utils.train(corrected_e,ones,U,optimizer, gnn, trn_dataset, loss_fn, i)
            score = utils.test(corrected_e,ones,U,gnn, val_dataset, score_fn)
            if score >= val_score:
                early_stop = 0
                val_score = score
                tst_score = utils.test(corrected_e,ones,U,gnn,
                                          tst_dataset,
                                          score_fn
                                         )
            else:
                early_stop += 1
            if early_stop > 200:
                print('epoch:',i)
                break
        outs.append(tst_score)
        print('repeat:',rep, time.time()-s,tst_score)
        # if not args.test:
        wandb.log({'tst_score': tst_score})
    outs = np.array(outs)
    # if not args.test:
    wandb.log({'acc': np.average(outs)})
    wandb.log({'err': np.max(np.abs(sns.utils.ci(sns.algorithms.bootstrap(outs,func=np.mean,n_boot=1000),95)-outs.mean()))})
    print(
        f"avg {np.average(outs):.4f} error {np.max(np.abs(sns.utils.ci(sns.algorithms.bootstrap(outs,func=np.mean,n_boot=1000),95)-outs.mean())):.4f}"
    )
    return np.average(outs)



def train(config=None):
    with wandb.init(config=None):
        config = wandb.config
        conv_layer = 10
        work(config.strategy,
             conv_layer,
                config.alpha,
                config.lr1,
                config.lr2,
                config.lr3,
                config.wd1,
                config.wd2,
                config.wd3,
                config.dpb,
                config.dpt,
                config.beta,
                a=config.a,
                b=config.b)



if __name__ == '__main__':
    wandb.login()

    sweep_config = {
        'method': 'grid'
    }

    args = utils.parse_args()
    device = torch.device('cuda:'+str(args.device) if torch.cuda.is_available() else 'cpu')
    # device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
    print(device)
    baseG = datasets.load_dataset(args.dataset, args.split).to(device)
    trn_dataset, val_dataset, tst_dataset = None, None, None
    output_channels = baseG.y.unique().shape[0]

    loss_fn = nn.CrossEntropyLoss()
    score_fn = metrics.multiclass_accuracy
    split()

    parameters_dict= realworld_params[args.dataset]

    print(parameters_dict)
    print(parameters_dict['a'])

    parameters_dict = {
        'lr1':{'values':[parameters_dict['lr1']]},
        'lr2':{'values':[parameters_dict['lr2']]},
        'lr3':{'values':[parameters_dict['lr3']]},
        'wd1':{'values':[parameters_dict['wd1']]},
        'wd2':{'values':[parameters_dict['wd2']]},
        'wd3':{'values':[parameters_dict['wd3']]},
        'alpha':{'values':[parameters_dict['alpha']]},
        'a':{'values':[parameters_dict['a']]},
        'b':{'values':[parameters_dict['b']]},
        'dpb':{'values':[parameters_dict['dpb']]},
        'dpt':{'values':[parameters_dict['dpt']]},
        'strategy':{'values':['low1']},
        'beta':{'distribution':'q_uniform','min':0.,'max':1.0,'q':0.01}
    }

    sweep_config['parameters'] = parameters_dict

    breakpoint()

    sweep_id = wandb.sweep(sweep_config, project="opt_fast_"+args.dataset)

    wandb.agent(sweep_id, train, count=args.optruns)

    wandb.finish()
    # wandb.init(project='new_'+args.dataset, entity='lukangkang123', config={'dataset': args.dataset,'method':args.method})