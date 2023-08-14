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

source_dir = "/path/to/bony_labyrinth_dataset/" #F01/F01/uCT/"    ## TODO: update path variable here ##
output_dir = "/path/to/output/dir/GL_preprocessed_data/"        ## TODO: update path variable here ##
dataset_dir_uCT = "/GL_preprocessed_data/dataset/"
# os.listdir(source_dir)

specimen_list=os.listdir(source_dir)

pat_dir_uCT_RAW=[]
for i in range(0,len(specimen_list)):
    pat_dir_uCT_RAW.append(source_dir+specimen_list[i]+"/"+specimen_list[i]+"/uCT/"+f"{specimen_list[i]}_uCT_RAW.nii")

pat_dir_uCT_LABELS=[]
for i in range(0,len(specimen_list)):
    pat_dir_uCT_LABELS.append(source_dir+specimen_list[i]+"/"+specimen_list[i]+"/uCT/"+f"{specimen_list[i]}_uCT_LABELS.nii")

def Normalize(image):
    minval = image.min()
    maxval = image.max()
    wld = np.clip(image, minval, maxval)
    wld -= minval
    wld *= (1 / (maxval-minval))
    return wld

def hd(seg_image,ground_truth):  
    seg_surface_points = np.transpose(np.nonzero(seg_image))
    ground_truth_surface_points = np.transpose(np.nonzero(ground_truth))
    hausdorff_distance1 = directed_hausdorff(seg_surface_points, ground_truth_surface_points)[0]   # directed HD from seg to ground truth
    hausdorff_distance2 = directed_hausdorff(ground_truth_surface_points, seg_surface_points)[0]   # directed HD from ground truth to seg
    hausdorff_distance = max(hausdorff_distance1, hausdorff_distance2)
    return hausdorff_distance

def pyvistarise(mesh):
    return pv.PolyData(np.asarray(mesh.vertices), np.insert(np.asarray(mesh.triangles), 0, 3, axis=1), deep=True, n_faces=len(mesh.triangles))

def gs_mesh(dist_xfm,desired_spacing):
    # get gold standard
    verts_gs, faces_gs, normals_gs, values_gs = marching_cubes(volume=dist_xfm, level=0., spacing=desired_spacing)
    # use open3d to smooth mesh
    seg_mesh_gs = o3d.geometry.TriangleMesh()
    seg_mesh_gs.vertices = o3d.utility.Vector3dVector(verts_gs)
    seg_mesh_gs.triangles = o3d.utility.Vector3iVector(faces_gs)
    seg_mesh_gs.vertex_normals = o3d.utility.Vector3dVector(normals_gs)
    seg_mesh_gs = seg_mesh_gs.filter_smooth_taubin(number_of_iterations=100)
    seg_mesh_gs = seg_mesh_gs.simplify_quadric_decimation(target_number_of_triangles=70000)
    seg_mesh_gs = seg_mesh_gs.filter_smooth_taubin(number_of_iterations=20)
    seg_mesh_gs.remove_unreferenced_vertices()

    # grab the smoothed vertices and triangles
    verts_smooth_gs = np.asarray(seg_mesh_gs.vertices)
    triangles_smooth_gs = np.asarray(seg_mesh_gs.triangles)
    vertex_normals_smooth_gs = np.asarray(seg_mesh_gs.vertex_normals)

    pos_gs = torch.from_numpy(verts_smooth_gs).to(torch.float)
    face_gs = torch.from_numpy(triangles_smooth_gs).t().contiguous() 
    data = ToUndirected()(FaceToEdge()(torch_geometric.data.Data(pos=pos_gs, face=face_gs))) 
    
    pcd = seg_mesh_gs.sample_points_uniformly(number_of_points=1000000)
    uniform_points_gs = np.asarray(pcd.points)
    
    return seg_mesh_gs, triangles_smooth_gs, data, uniform_points_gs, vertex_normals_smooth_gs


def def_mesh(deformed_dist_xfm,desired_spacing):
    verts, faces, normals, values = marching_cubes(volume=deformed_dist_xfm,level=0.,spacing=desired_spacing) # keep spacing in voxel space
    seg_mesh = o3d.geometry.TriangleMesh()
    seg_mesh.vertices = o3d.utility.Vector3dVector(verts)
    seg_mesh.triangles = o3d.utility.Vector3iVector(faces)
    seg_mesh.vertex_normals = o3d.utility.Vector3dVector(normals)
    seg_mesh = clean_mesh(mesh=seg_mesh, max_components_to_keep="auto")
    seg_mesh = seg_mesh.filter_smooth_taubin(number_of_iterations=100)
    seg_mesh = seg_mesh.simplify_quadric_decimation(target_number_of_triangles=70000)
    seg_mesh = seg_mesh.filter_smooth_taubin(number_of_iterations=20)
    seg_mesh.remove_unreferenced_vertices()

    verts_smooth = np.asarray(seg_mesh.vertices)
    triangles_smooth = np.asarray(seg_mesh.triangles)
    vertex_normals_smooth = np.asarray(seg_mesh.vertex_normals)
    
    pos = torch.from_numpy(verts_smooth).to(torch.float)
    face = torch.from_numpy(triangles_smooth).t().contiguous()
    data = ToUndirected()(FaceToEdge()(torch_geometric.data.Data(pos=pos, face=face)))

    return seg_mesh,triangles_smooth,data, vertex_normals_smooth

def node_signed_distances(dist_xfm,node_coords):
    points = (np.arange(dist_xfm.shape[0]), np.arange(dist_xfm.shape[1]), np.arange(dist_xfm.shape[2]))
    signed_dists = interpn(points=points, values=dist_xfm, xi=node_coords.numpy())
    return signed_dists


signed_distances_list = []
n_bins=5
global_signed_classes = torch.zeros((n_bins))

for i in tqdm(range(len(specimen_list))):
    
    CT = sitk.ReadImage(pat_dir_uCT_RAW[i])
    CT_spacing = np.array(CT.GetSpacing())
        
    npy_CT = np.load(join(output_dir, "CT_vol",f"{specimen_list[i]}.npy"))
    npy_seg = np.load(join(output_dir, "Seg","GS", f"{specimen_list[i]}.npy"))
    
    
    assert((npy_seg.shape==npy_CT.shape))
    num_deformations_to_generate_per_seg = 100
    desired_spacing = CT_spacing
     
    dist_xfm = np.load(join(output_dir, "distance_transform", "GS",f"{specimen_list[i]}.npy"))
    
  

    for deformation_num in range(num_deformations_to_generate_per_seg):


        deformed_dist_xfm = np.load(join(output_dir, "distance_transform",f"{specimen_list[i]}_{deformation_num}.npy"))

        
        data = torch.load(join(output_dir,"graph_objects",f"{specimen_list[i]}_{deformation_num}.pt"))
    
        node_coords = (data.pos)/torch.tensor(desired_spacing)
        for node_idx, node_coord in enumerate(node_coords):
            node_coord = torch.clamp(node_coord, min=torch.tensor([0,0,0]), max=torch.tensor(dist_xfm.shape) - 1)
            node_coords[node_idx]=node_coord 
            
        patches_tensor = sample_ct_on_nodes(node_coords, torch.tensor(npy_CT),patch_size=9)
        torch.save(patches_tensor, join(output_dir, "ct_patches_9x9x9", f"{specimen_list[i]}_{deformation_num}.pt"))



