"""
This file contains utility functions for dada handling
"""
import os.path as osp
import numpy as np
import torch
from sklearn.metrics.pairwise import pairwise_distances
from sklearn import preprocessing
import copy
import math
import random
from torch_geometric.data import Data
import os
import re
import fnmatch

# Get the number of files with names following file_format_N, in which N is integer
def get_numFiles(folder_path):
   # folder_path: the folder of interest. eg. 'data/JSNN_DATA/LaCoO3'
   # file_format: eg. CONFIG

   # Define the folder path and the pattern
   
   # pattern = fr'^{re.escape(file_format)}_\d+$'
   # pattern = fr'^{re.escape(file_format)}_\d+$'  # Matches strings like "CONFIG_123"

   # List all files in the folder
   files = os.listdir(folder_path)

   # Initialize a counter for matching files
   matching_count = 0
   # Iterate through the files and count the ones that match the pattern
   for filename in files:
      #if re.match(pattern, filename):
      matching_count += 1

   return matching_count

def load_energyList(folder, file):
   filename = osp.join(folder, file)
   with open(filename, "r") as f:
      data = f.readlines()

   data = np.asarray(data, dtype=float)
   return data

# Load gap from a data file
def load_y(folder, file_id, file_format):
   file = file_format + '_' + str(file_id)
   filename = osp.join(folder, file)
   with open(filename, "r") as f:
      data = f.readlines()

   data = np.asarray(data, dtype=float)
   data = torch.Tensor(data)
   return data



def load_coordinates(folder, file_id, file_id2, file_format):
   """
   folder: the folder that contains the coordinate files. eg. data/GCN_N1P_NEW/CONFIGS
   file_id: index to the coordinate file. eg. 1000
   file_format: the prfix of a cfile. eg. COORD
   """

   file = file_format + '_'+str(file_id)+'_'+str(file_id2)+'.xyz'
   filename = osp.join(folder, file)
   with open(filename, "r") as f:
      data = f.readlines()
   data = data[2:]
   data = [line.split() for line in data]
   # print(data)
   data = [l[1:] for l in data]
   data = np.asarray(data, dtype=float)
   data = torch.tensor(data, dtype=torch.float32)
   # Add 1 to negative coordinates
   mask = data<0
   y = torch.zeros_like(data)
   y[mask] = 1.0
   data = data + y

   for i in data:
      i[0] = i[0]/10.934894
      i[1] = i[1]/11.100272
      i[2] = i[2]/15.619470

   return data


# Calculate the distance between two atoms with fractional coordinates
def dist_fractional(A, B, x_scale=10.934894, y_scale=11.100272, z_scale=15.619470):
   """
   This function calculates the distance between two atoms. 
   Their coordinates are in the fractional coordinate format.

   A: coordinates of atom A. eg [0.105692 0.0273734 0.193402]
   B: coordinates of atom B. eg [0.094583 0.0534993 0.577509]
   scale: the actual length of a side of the cubic box, in angstron
   """

   XA = A[0]
   YA = A[1]
   ZA = A[2]
   XB = B[0]
   YB = B[1]
   ZB = B[2]

   XAB = XA - XB
   XAB = XAB - math.floor(XAB)
   if XAB > 0.5:
       XAB = XAB - 1.0
   elif XAB < -0.5:
       XAB = XAB + 1.0
   YAB = YA - YB
   YAB = YAB - math.floor(YAB)
   if YAB > 0.5:
       YAB = YAB - 1.0
   elif YAB < -0.5:
       YAB = YAB + 1.0
   ZAB = ZA - ZB
   ZAB = ZAB - math.floor(ZAB)
   if ZAB > 0.5:
       ZAB = ZAB - 1.0
   elif ZAB < -0.5:
       ZAB = ZAB + 1.0
   DAB = math.sqrt((XAB*x_scale)**2 + (YAB*y_scale)**2 + (ZAB*z_scale)**2)
   return DAB


# Calculate pair-wise distance matrix of atoms in a molecule
def get_dmatrix(coordinates):
   """
   This function calculates the pair-wise Euclidean distance of atoms in a molecule

   coordinates: coordinates of atoms
   scale: the cubic box size, in angstrom

   """

   # Add 1 to negative coordinates
   mask = coordinates<0
   y = torch.zeros_like(coordinates)
   y[mask] = 1.0
   coordinates = coordinates + y
   dmatrix = torch.tensor(pairwise_distances(coordinates, metric=dist_fractional))
   return dmatrix



# Get the adjacency matrix of a molecule using te dmatrix
def get_adj_matrix(dmatrix, cutoff):
   """
   Using the cutoff distance to determine if there is a chemical bond between a pair of atoms

   dmatrix: pairwise distances of atoms
   cutoff: cutoff distance, in angstron

   """

   adjmatrix = torch.zeros_like(dmatrix, dtype=int)
   mask = dmatrix<=cutoff
   adjmatrix[mask] = 1
   
   adjmatrix.fill_diagonal_(0)

   return adjmatrix
    
    

# Generate edge_index in COO format for PyG Data
def get_edge_index(adj_mat):
   # Convert bt_data (adjacency matrix) to edge_index required by PyG
   # More info: https://pytorch-geometric.readthedocs.io/en/latest/get_started/introduction.html#data-handling-of-graphs
    
   edge_index = adj_mat.nonzero().t().contiguous()
   return edge_index


def atomic_number_one_hot(atom_nums):
   """
   atom_num: the atomic numbers of a molecule
   one hot encode the atom numbers and return a 2D tensor

   """

   atom_type_dict = {
         'La': [1, 0, 0, 0, 0, 0, 0, 0, 0], 
         'Co': [0, 1, 0, 0, 0, 0, 0, 0, 0], 
         'Ni': [0, 0, 1, 0, 0, 0, 0, 0, 0],  
	     'Fe': [0, 0, 0, 1, 0, 0, 0, 0, 0],
         'Mn': [0, 0, 0, 0, 1, 0, 0, 0, 0],
         'Sr': [0, 0, 0, 0, 0, 1, 0, 0, 0],
         'Cr': [0, 0, 0, 0, 0, 0, 1, 0, 0],
         'Ba': [0, 0, 0, 0, 0, 0, 0, 1, 0],
         'Zn': [0, 0, 0, 0, 0, 0, 0, 0, 1],
         'O': [0, 0, 0, 0, 0, 0, 0, 0, 0]   
      }   

   atom_nums_onehotcoded = torch.Tensor([atom_type_dict.get(item) for item in atom_nums])
    
   return atom_nums_onehotcoded

def get_atom_num(folder, file_id, file_id2, file_format):
   file = file_format + '_' + str(file_id) + '_' + str(file_id2) + '.xyz'
   filename = osp.join(folder, file)
   with open(filename, "r") as f:
      data = f.readlines()
   data = data[2:]
   data = [line.split() for line in data]
   atom_list = [line[0] for line in data]
    
   return atom_list


def read_one_molecule(datasrc, file_id, file_id2, y_list):
   """
   This function reads in all the data for one molecule.

   datasrc: the source folder of data. eg. data/JSNN_DATA/LaCoO3_DATA
   """

   # Step 1: read molecule properties (eg. formation energy)
   #folder = osp.join(datasrc, 'ENERGY')
   #file_format = 'ENERGY'
   #y = load_y(folder, id, file_format)
   y = y_list[(file_id-1)*10+file_id2 -1]


   # Step 2: get the adjancy matrix of a molecule
   folder = osp.join(datasrc, 'CONFIGS')
   file_format = 'DEFECT'
   coordinates = load_coordinates(folder, file_id, file_id2, file_format)

   edge_mat = get_dmatrix(coordinates)
   adj_mat = get_adj_matrix(edge_mat, 3)
    
   # step 3: edge attribute
   dis_mat_adj = edge_mat * adj_mat


   atom_type_list = get_atom_num(folder, file_id, file_id2, file_format)
   atom_type_onehotcoded = atomic_number_one_hot(atom_type_list)


   # Step 4: Assemble pieces of data to generate x
   x = atom_type_onehotcoded

   # Step 5: Get the edge index in COO format
   edge_ind = get_edge_index(adj_mat)
   edge_attr = dis_mat_adj[edge_ind[0], edge_ind[1]].unsqueeze(1).to(torch.float)


   # Use x, edge_index, edge_attr and y to create a torch_geometric.data.Data object
   data = Data(x=x, edge_index=edge_ind, edge_attr=edge_attr, y=y)
   
   return data



   
# Split dataset to train, validate and test subsets
def split_dataset(dataset, train_p=70, val_p=15, shuffle=True):
   """
   dataset: dataset to split
   train_p: training set percentage
   val_p: validating set percentage

   """  

   if shuffle:
      ############## Change this if not to seek reproducibility ####
      #random.seed(218632)
      idx = []
      while len(idx) < dataset.len():
         item = random.randint(0, dataset.len()-1)
         if item not in idx:
            idx.append(item)         
   else:
      idx = np.arange(dataset.len())

   len_train = int(len(idx) * train_p)
   len_val = int(len(idx) * val_p)

   
   train_dataset = dataset[idx[0:len_train]]
   val_dataset = dataset[idx[len_train:len_train+len_val]]
   test_dataset = dataset[idx[len_train+len_val:]]

   return train_dataset, val_dataset, test_dataset



















































































































