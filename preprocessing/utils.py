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

def windowLevelNormalize(image, level, window):
    minval = level - window/2
    maxval = level + window/2
    wld = np.clip(image, minval, maxval)
    wld -= minval
    wld *= (1 / window)
    return wld

def Normalize(image):
    minval = image.min()
    maxval = image.max()
    wld = np.clip(image, minval, maxval)
    wld -= minval
    wld *= (1 / (maxval-minval))
    return wld

def clean_mesh(mesh, max_components_to_keep="auto"):
    # get number to keep
    if max_components_to_keep == "auto":
        _, num_tris_per_cluster, _ = mesh.cluster_connected_triangles()
        max_components_to_keep = len(num_tris_per_cluster)
    # identify connected triangles
    cluster_ind_per_tri, num_tris_per_cluster, _ = mesh.cluster_connected_triangles()
    if len(num_tris_per_cluster) < (max_components_to_keep + 1):
        return mesh # mesh needs no cleaning
    # identify triangles to toss
    num_tris = np.asarray(mesh.triangles).shape[0]
    cut_ind = max_components_to_keep - 1
    tris_to_yeet = list(filter(lambda tri_index: cluster_ind_per_tri[tri_index] > cut_ind, range(num_tris)))
    # remove triangles
    mesh.remove_triangles_by_index(triangle_indices=tris_to_yeet)
    # remove nodes
    mesh.remove_unreferenced_vertices()
    # return cleaned mesh
    return mesh


def get_bounds(seg, margin=(5, 10, 10)):
    # input: numpy binary segmentation, output: tuple of start positions (cc,ap,lr) and tuple of extents of bounding box 
    bounds = np.argwhere((seg == 1))
    cc = np.array((min(bounds[:,0]) - margin[0], max(bounds[:,0]) + margin[0]))
    ap = np.array((min(bounds[:,1]) - margin[1], max(bounds[:,1]) + margin[1]))
    lr = np.array((min(bounds[:,2]) - margin[2], max(bounds[:,2]) + margin[2]))
    # clip to avoid going out of range
    cc = np.clip(cc, 0, seg.shape[0]-1)
    ap = np.clip(ap, 0, seg.shape[1]-1)
    lr = np.clip(lr, 0, seg.shape[2]-1)
    # put the tuples together
    starts = (cc[0], ap[0], lr[0])
    extents = (cc[1]-cc[0], ap[1]-ap[0], lr[1]-lr[0])
    return starts, extents

def get_class_signed(dist): #uCT
    # signed 0: -0.16mm-, 1: -0.16 - -0.1mm, 2: -0.1 - 0.1mm, 3: 0.1 - 0.16mm, 4: 0.16mm+
    if dist < -0.16:
        return 0
    if -0.16 < dist < -0.1:
        return 1
    if -0.1 < dist < 0.1:
        return 2
    if 0.1<dist<0.16:
        return 3
    return 4

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

class ConfusionMatrix:
    # Computes and plots a confusion matrix for the classification task
    def __init__(self, n_classes=5):
        self.n_classes = n_classes
        self.confusion_matrix = np.zeros((n_classes, n_classes), int)

    def update(self, targets, soft_preds):
        targets = np.argmax(targets,axis=1)
        hard_preds = np.argmax(soft_preds, axis=1)
        self.confusion_matrix += confusion_matrix(y_true=targets, y_pred=hard_preds, labels=np.arange(self.n_classes)).astype(int)

    def _normalise_by_true(self):
        return np.transpose(np.transpose(self.confusion_matrix) / self.confusion_matrix.sum(axis=1))

    def _normalise_by_pred(self):
        return self.confusion_matrix / self.confusion_matrix.sum(axis=0)

    def gen_matrix_fig(self):
        # setup
        fig, ax = plt.subplots(1,1, figsize=(6, 6), tight_layout=True)
        # normalise by true classes
        normed_confusion_matrix = self._normalise_by_true()
        I_blanker = np.ones((self.n_classes, self.n_classes))
        I_blanker[np.identity(self.n_classes, bool)] = np.nan
        ax.imshow(normed_confusion_matrix, cmap='Greens', vmin=0, vmax=1)
        ax.imshow(normed_confusion_matrix*I_blanker, cmap='Reds', vmin=0, vmax=1)
        for target_idx in range(self.n_classes):
            for pred_idx in range(self.n_classes):
                ax.text(pred_idx, target_idx, s=f"{np.round(normed_confusion_matrix[target_idx, pred_idx]*100, 1)}%\nn={self.confusion_matrix[target_idx, pred_idx]}", ha='center', va='center')
        ax.set_xlabel("Pred class")
        ax.set_ylabel("True class")
        ax.set_xticks(np.arange(self.n_classes))
        ax.set_yticks(np.arange(self.n_classes))
        if self.n_classes == 5:
            ax.set_xticklabels(["<-0.16mm","-0.16 - -0.1mm","-0.1 - 0.1mm", "0.1 - 0.16mm", ">0.16mm"],rotation=45)
            ax.set_yticklabels(["<-0.16mm","-0.16 - -0.1mm","-0.1 - 0.1mm", "0.1 - 0.16mm", ">0.16mm"]) ##uCT
        elif self.n_classes == 3:
            ax.set_xticklabels(["<-0.1mm","-0.1 -> 0.1mm", ">0.1mm"])
            ax.set_yticklabels(["<-0.1mm","-0.1 -> 0.1mm", ">0.1mm"])
        else:
            raise NotImplementedError("Check confusion matrix labels!")
            exit()
        # return figure
        return fig,normed_confusion_matrix,self.confusion_matrix

    def retrieve_data(self):
        return self.confusion_matrix

def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

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


