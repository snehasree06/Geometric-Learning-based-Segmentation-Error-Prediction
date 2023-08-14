import os
from os.path import join
import numpy as np
import SimpleITK as sitk

import open3d as o3d
from utils import *
import pyvista as pv

from tqdm import tqdm

import torch
import torch_geometric
from torch_geometric.transforms import FaceToEdge, ToUndirected

from skimage.transform import rescale, resize
from skimage.measure import marching_cubes
from scipy.ndimage import distance_transform_edt
from scipy.signal import fftconvolve
from scipy.interpolate import interpn
from scipy.ndimage import gaussian_filter
from scipy.spatial.distance import directed_hausdorff

device = "cuda" if torch.cuda.is_available() else "cpu" #cuda is for gpu

source_dir = "/path/to/bony_labyrinth_dataset/" #F01/F01/CT/"    ## TODO: update path variable here ##
output_dir = "/GL_preprocessed_data/"        ## TODO: update path variable here ##
dataset_dir_uCT = "/GL_preprocessed_data/DS/"
# os.listdir(source_dir)

specimen_list=os.listdir(source_dir)

pat_dir_uCT_RAW=[]
for i in range(0,len(specimen_list)):
    pat_dir_uCT_RAW.append(source_dir+specimen_list[i]+"/"+specimen_list[i]+"/uCT/"+f"{specimen_list[i]}_uCT_RAW.nii")

pat_dir_uCT_LABELS=[]
for i in range(0,len(specimen_list)):
    pat_dir_uCT_LABELS.append(source_dir+specimen_list[i]+"/"+specimen_list[i]+"/uCT/"+f"{specimen_list[i]}_uCT_LABELS.nii")


n_bins=5
global_signed_classes = torch.zeros((n_bins))

for i in tqdm(range(len(specimen_list))):
    
    CT = sitk.ReadImage(pat_dir_uCT_RAW[i])
    CT_spacing = np.array(CT.GetSpacing())
    desired_spacing = CT_spacing
    num_deformations_to_generate_per_seg = 100

    # get distance xfm
    dist_xfm = np.load(join(output_dir,"distance_transform","GS",f'{specimen_list[i]}.npy'))
    # Generate training data using realistic deformations!
    for deformation_num in range(num_deformations_to_generate_per_seg):
        
        data = torch.load(join(output_dir, "graph_objects", f"{specimen_list[i]}_{deformation_num}.pt"))

        node_coords = (data.pos)/torch.tensor(desired_spacing)
        for node_idx, node_coord in enumerate(node_coords):
            node_coord = torch.clamp(node_coord, min=torch.tensor([0,0,0]), max=torch.tensor(dist_xfm.shape) - 1)
            node_coords[node_idx]=node_coord 

        # signed dists
        points = (np.arange(dist_xfm.shape[0]), np.arange(dist_xfm.shape[1]), np.arange(dist_xfm.shape[2]))
        signed_dists = torch.load(join(output_dir, "signed_distances", f"{specimen_list[i]}_{deformation_num}.pt"))
        
        n_classes = n_bins
        node_classes_signed = torch.zeros(size=(data.pos.size(0), n_classes), dtype=int)
        for node_idx in range(data.pos.size(0)):
            node_classes_signed[node_idx, get_class_signed(dist=signed_dists[node_idx])] = 1
        global_signed_classes += torch.tensor([node_classes_signed[:,i].sum() for i in range(n_classes)])
        torch.save(node_classes_signed, join(output_dir, f"signed_classes", f"{specimen_list[i]}_{deformation_num}.pt"))

torch.save(global_signed_classes, join(output_dir, "all_signed_classes","all_signed_classes.pt"))