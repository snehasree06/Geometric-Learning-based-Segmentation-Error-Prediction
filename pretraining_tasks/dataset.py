# dataset.py
import os
from os.path import join
import numpy as np
import torch
import random
from utils import *
import torch_geometric
from utils import *


class MyData(torch_geometric.data.Data):
    def __cat_dim__(self, key, value, *args, **kwargs):
            return super().__cat_dim__(key, value, *args, **kwargs)


class pretrain_dataset(torch.utils.data.Dataset):
    def __init__(self, ct_volume_dir, vertex_normals_dir, mesh_dir, mesh_inds, flag="vn", seed=None):
        super(pretrain_dataset).__init__()
        
        self.mesh_dir = mesh_dir
        self.mesh_inds = mesh_inds
        self.flag = flag
        self.ct_volume_dir = ct_volume_dir
        all_pat_names = sorted(getFiles(ct_volume_dir))
        all_mesh_names = sorted(getFiles(mesh_dir))
        mesh_names = [all_mesh_names[ind] for ind in mesh_inds]
        pat_names = [all_pat_names[ind] for ind in mesh_inds]

        self.num_pats = len(pat_names)
        self.examples = []
        for idx in range(len(pat_names)):
            mesh = torch.load(join(self.mesh_dir,mesh_names[idx]))
            vertex_normals = np.load(join(vertex_normals_dir,pat_names[idx]))
            vertex_normals = 2*(vertex_normals - vertex_normals.min())/(vertex_normals.max()-vertex_normals.min())-1
            ct_volume = np.load(join(self.ct_volume_dir,pat_names[idx]))
            for i in range(mesh.pos.shape[0]):
                node_coord = mesh.pos[i]
                vertex_normal = vertex_normals[i]
                self.examples.append((node_coord,mesh_names[idx],pat_names[idx],vertex_normal,ct_volume,mesh))        
        
        if seed != None:
            random.seed(seed)
        
    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        
        node_coord,mesh_name,pat_name,vertex_normal,ct_volume,mesh = self.examples[idx]
        ct_volume_size = [ct_volume.shape[0],ct_volume.shape[1],ct_volume.shape[2]]
        node_coord = node_coord/torch.tensor([0.06,0.06,0.06])
        node_coord = torch.clamp(node_coord, min=torch.tensor([2,2,2]), max=torch.tensor(ct_volume_size) - 3)
        node_coord = node_coord.numpy()
        point = np.round(node_coord).astype(int)
        point_tensor = torch.from_numpy(point)
        point = point_tensor.cpu().numpy()
        ct_sub_vol = ct_volume[point[0]-2:point[0]+3, point[1]-2:point[1]+3, point[2]-2:point[2]+3]
    
        if self.flag == "vn":
            sample = {"patch": torch.unsqueeze(torch.from_numpy(ct_sub_vol), dim=0).float(), "label": torch.from_numpy(vertex_normal).float()}
            return sample

        elif self.flag == "vae_recon":
            sample = {"patch": torch.unsqueeze(torch.from_numpy(ct_sub_vol), dim=0).float(), "label": torch.unsqueeze(torch.from_numpy(ct_sub_vol), dim=0).float()}
            return sample

        elif self.flag == "mask_ae":
            mask = np.random.choice([0, 1], size=(5, 5, 5))
            masked_ct_sub_vol = ct_sub_vol*mask
            sample = {"patch": torch.unsqueeze(torch.from_numpy(masked_ct_sub_vol), dim=0).float(), "label": torch.unsqueeze(torch.from_numpy(ct_sub_vol), dim=0).float()}
            return sample






