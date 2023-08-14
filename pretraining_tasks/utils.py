## utils.py
# some useful functions!
import numpy as np
from itertools import cycle
import torch
import shutil
import os
import logging
import argparse
import sys
import math
import warnings
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt



def getDirs(parent_dir):
    ls = []
    for dir_name in os.listdir(parent_dir):
        path = os.path.join(parent_dir, dir_name)
        if os.path.isdir(path):
            ls.append(dir_name)
    return ls

def getFiles(targetdir):
    ls = []
    for fname in os.listdir(targetdir):
        path = os.path.join(targetdir, fname)
        if os.path.isdir(path):
            continue
        ls.append(fname)
    return ls

def sample_ct_on_nodes(node_coords, ct_subvolume, patch_size=5):
    #
    # IMPORTANT: expects node coordinates in CT voxels coordinate system
    #
    low_bound = patch_size // 2
    high_bound = (patch_size + 1) // 2 
    patches_tensor = torch.zeros((node_coords.size(0), patch_size, patch_size, patch_size))
    # sample a 3D patch on the ct subvolume for every node in the mesh
    for node_idx, node_coord in enumerate(node_coords):
        # use clamping to deal with the case that the patch is outside the ct_subvolume -> raise a warning though?
        cent_coord_unclamped = node_coord
        cent_coord = torch.clamp(cent_coord_unclamped, min=torch.tensor([low_bound,low_bound,low_bound]), max=torch.tensor(ct_subvolume.shape) - high_bound)
        cent_coord = torch.round(cent_coord).type(torch.int)
        cc_lo, cc_hi = cent_coord[0] - low_bound, cent_coord[0] + high_bound
        ap_lo, ap_hi = cent_coord[1] - low_bound, cent_coord[1] + high_bound
        lr_lo, lr_hi = cent_coord[2] - low_bound, cent_coord[2] + high_bound
        patches_tensor[node_idx] = ct_subvolume[cc_lo:cc_hi, ap_lo:ap_hi, lr_lo:lr_hi]
    return patches_tensor



def Normalize(image):
    minval = image.min()
    maxval = image.max()
    wld = np.clip(image, minval, maxval)
    wld -= minval
    wld *= (1 / (maxval-minval))
    return wld

def try_mkdir(dir_name):
    try:
        os.mkdir(dir_name)
    except OSError:
        pass

def get_logger(name, level=logging.INFO):
    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.propagate = False
    # Logging to console
    stream_handler = logging.StreamHandler(sys.stdout)
    formatter = logging.Formatter(
        '%(asctime)s [%(threadName)s] %(levelname)s %(name)s - %(message)s')
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)
    return logger

def get_number_of_learnable_parameters(model):
    model_parameters = filter(lambda p: p.requires_grad, model.parameters())
    return sum([np.prod(p.size()) for p in model_parameters])


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')
        

def split_train_val_test(dataset_size, seed):
    train_ims, val_ims, test_ims = math.floor(dataset_size*0.7), math.floor(dataset_size*0.1), math.ceil(dataset_size*0.2)
    if dataset_size - (train_ims+val_ims+test_ims) == 1:
        val_ims += 1 # put the extra into val set
    try:
        assert(train_ims+val_ims+test_ims == dataset_size)
    except AssertionError:
        print("Check the k fold data splitting, something's dodgy...")
        exit(1)
    train_inds, val_inds, test_inds = [], [], []
    # initial shuffle
    np.random.seed(seed)
    shuffled_ind_list = np.random.permutation(dataset_size)

    for i in range(test_ims):
        test_inds.append(shuffled_ind_list[i])
    for i in range(train_ims):
        train_inds.append(shuffled_ind_list[i + test_ims])
    for i in range(val_ims):
        val_inds.append(shuffled_ind_list[i + test_ims + train_ims])

    return train_inds, val_inds, test_inds


def k_fold_split_train_val_test(dataset_size, fold_num, seed):
    k = int(fold_num-1)
    train_ims, val_ims, test_ims = math.floor(dataset_size*0.7), math.floor(dataset_size*0.1), math.ceil(dataset_size*0.2)
    if dataset_size - (train_ims+val_ims+test_ims) == 1:
        val_ims += 1 # put the extra into val set
    try:
        assert(train_ims+val_ims+test_ims == dataset_size)
    except AssertionError:
        print("Check the k fold data splitting, something's dodgy...")
        exit(1)
    train_inds, val_inds, test_inds = [], [], []
    # initial shuffle
    np.random.seed(seed)
    shuffled_ind_list = np.random.permutation(dataset_size)
    # allocate dataset indices based upon the fold number --> not the prettiest or most efficient implementation, but functional
    cyclic_ind_list = cycle(shuffled_ind_list)
    for i in range(k*test_ims):
        next(cyclic_ind_list)   # shift start pos
    for i in range(test_ims):
        test_inds.append(next(cyclic_ind_list))
    for i in range(train_ims):
        train_inds.append(next(cyclic_ind_list))
    for i in range(val_ims):
        val_inds.append(next(cyclic_ind_list))
    return train_inds, val_inds, test_inds

def dice_coef(score, target):
    score,target = torch.from_numpy(score), torch.from_numpy(target)
    target = target.float()
    smooth = 1e-5
    intersect = torch.sum(score * target)
    y_sum = torch.sum(target * target)
    z_sum = torch.sum(score * score)
    coef = (2 * intersect + smooth) / (z_sum + y_sum + smooth)
    return coef


def product_loss(score, target):
    score,target = score, target
    target = target.float()
    smooth = 1e-5
    intersect = torch.sum(score * target)
    y_sum = torch.sum(target * target)
    z_sum = torch.sum(score * score)
    loss = -(intersect) / (z_sum + y_sum + intersect)
    return loss
