import os

import pandas as pd
import numpy as np
import scipy 
import torch

import ripley


#define genes,molecules:
sample_number = "219KS"
goi = ["SOD1","MLKL"]   #genes of interest
coi = ["connective tissue"]   #cell types of interest

microscopy_resolution = 0.1625
ssam_um_p_px = 2.5

cell_color_file = sample_number+"_custom_colour_celltypes.csv"
coordinate_file = sample_number+"_Decoded_LowThreshold.csv"
cell_map_file = "celltypefile.npy"
data_path = os.path.join(os.path.dirname(__file__),"..","..","data",sample_number,)

colors = pd.read_csv(os.path.join(data_path,cell_color_file))
coordinates = pd.read_csv(os.path.join(data_path,coordinate_file))
celltype_map = np.load(os.path.join(data_path,cell_map_file))

#rescale coordinates
coordinates['x']*=microscopy_resolution/ssam_um_p_px
coordinates['y']*=microscopy_resolution/ssam_um_p_px

celltype_matrix = torch.zeros(celltype_map.shape+(len(goi)+len(coi),))

i=0
for ct in coi:
    tissue_idcs = np.array(colors[colors.celltype.str.contains(ct)].index)
    for idx in tissue_idcs:
        celltype_matrix[i][celltype_map==idx]=1
    i+=1

for g in goi:
    gene_idcs = coordinates.gene
    molecule_idcs = scipy.gaussian_kde()


print('finished.')
