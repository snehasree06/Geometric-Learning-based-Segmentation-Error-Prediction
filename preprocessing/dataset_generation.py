import numpy as np
import h5py
import matplotlib.pyplot as plt
from tqdm import tqdm
import os

ct_dir ='/GL_preprocessed_data/CT_vol/'
seg_dir = '/GL_preprocessed_data/Seg/GS/'
deformed_seg_dir = '/GL_preprocessed_data/Seg/'
dist_xfm_dir = '/GL_preprocessed_data/distance_transform/GS/'
deformed_dist_xfm_dir = '/GL_preprocessed_data/distance_transform/'




source_dir = "/path/to/bony_labyrinth_dataset/" #F01/F01/CT/"    ## TODO: update path variable here ##
output_dir = "/GL_preprocessed_data/"        ## TODO: update path variable here ##
dataset_dir_uCT = "/GL_preprocessed_data/DS/"
# os.listdir(source_dir)

specimen_list=os.listdir(source_dir)



for i in tqdm(range(len(specimen_list))):    
    for def_num in range(100):
        filename = str(dataset_dir_uCT) + f"{specimen_list[i]}_{def_num}.h5" 
        with h5py.File(filename,'w') as data:
            data.create_dataset('input',data = np.load(os.path.join(ct_dir,f"{specimen_list[i]}.npy")))
            data.create_dataset('target_seg',data = np.load(os.path.join(seg_dir,f"{specimen_list[i]}.npy"))) 
            data.create_dataset('target_sdt',data = np.load(os.path.join(dist_xfm_dir,f"{specimen_list[i]}.npy")))
            data.create_dataset('deformed_seg',data = np.load(os.path.join(deformed_seg_dir,f"{specimen_list[i]}_{def_num}.npy")))
            data.create_dataset('deformed_sdt',data = np.load(os.path.join(deformed_dist_xfm_dir,f"{specimen_list[i]}_{def_num}.npy")))
