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
import h5py

device = "cuda" if torch.cuda.is_available() else "cpu" #cuda is for gpu

source_dir = "/path/to/bony_labyrinth_dataset/" #F01/F01/CT/"    ## TODO: update path variable here ##
output_dir = "/GL_preprocessed_data/"        ## TODO: update path variable here ##
dataset_dir_uCT = "/GL_preprocessed_data/DS/"
# os.listdir(source_dir)

# specimen_list=os.listdir(source_dir) 

specimen_list = [fname for fname in os.listdir(source_dir) if fname!="F12"]
pat_dir_uCT_RAW=[]
for i in range(0,len(specimen_list)):
    pat_dir_uCT_RAW.append(source_dir+specimen_list[i]+"/"+specimen_list[i]+"/uCT/"+f"{specimen_list[i]}_uCT_RAW.nii")

pat_dir_uCT_LABELS=[]
for i in range(0,len(specimen_list)):
    pat_dir_uCT_LABELS.append(source_dir+specimen_list[i]+"/"+specimen_list[i]+"/uCT/"+f"{specimen_list[i]}_uCT_LABELS.nii")

all_signed_classes = torch.load(join(output_dir,"all_signed_classes","all_signed_classes.pt"))
num_deformations_to_generate_per_seg=50
counter = 0
min_sd = 0.0
max_sd = 0.5
for i in range(len(specimen_list)):
    for deformation_num in range(num_deformations_to_generate_per_seg):
        signed_dists = torch.load(join(output_dir, "signed_distances", f"{specimen_list[i]}_{deformation_num}.pt"))
        # signed_dists = torch.clamp(signed_dists,min=-1.0,max=1.0)
        if min(signed_dists)<min_sd:
            min_sd = min(signed_dists)
            print(min_sd)
        if max(signed_dists)>max_sd:
            max_sd = max(signed_dists)
            print(max_sd)
        # print(f"min:{min(signed_dists)},max:{max(signed_dists)}")
        if min(signed_dists)<-1.0 or max(signed_dists)>1.0:
            print(f"fname: {specimen_list[i]}_{deformation_num}, min:{min(signed_dists)},max:{max(signed_dists)}")
            counter = counter + 1
print(counter)







