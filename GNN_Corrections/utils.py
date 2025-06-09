import torch
import math
import numpy as np


def index_to_mask(index, size):
    """
    Generate a binary mask for a vector of length size at the provided indexes.
    """
    mask = torch.zeros(size, dtype=torch.bool)
    mask[index] = 1
    return mask


def random_planetoid_splits(data, num_classes, percls_trn=20, val_lb=500, seed=12134):
    """
    Execute planetoid splitting for a passed graph dataset. 
    """
    index=[i for i in range(0,data.y.shape[0])]
    train_idx=[]
    rnd_state = np.random.RandomState(seed)
    for c in range(num_classes):
        class_idx = np.where(data.y.cpu() == c)[0]
        if len(class_idx)<percls_trn:
            train_idx.extend(class_idx)
        else:
            train_idx.extend(rnd_state.choice(class_idx, percls_trn,replace=False))
    rest_index = [i for i in index if i not in train_idx]
    val_idx=rnd_state.choice(rest_index,val_lb,replace=False)
    test_idx=[i for i in rest_index if i not in val_idx]
    # print(test_idx)

    data.train_mask = index_to_mask(train_idx,size=data.num_nodes)#.cuda()
    data.val_mask = index_to_mask(val_idx,size=data.num_nodes)#.cuda()
    data.test_mask = index_to_mask(test_idx,size=data.num_nodes)#.cuda()
    
    return data


class net_arg_converter:
    """
    Class for converting dicts to allow attribute based argument retrieval
    """
    def __init__(self, init_dict):
        for key, value in init_dict.items():
            # if the value is a dict, call create_attrs recursively
            setattr(self, key, value)