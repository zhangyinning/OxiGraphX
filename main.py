import time
import numpy as np
import os
import os.path as osp
import torch
from torch_geometric.loader import DataLoader
from torch_geometric.nn import DataParallel
from torch.utils.data import ConcatDataset

from utils import *
from dataset import *
from config import config
from model import *
from train import *


# read in hyperparameters

'''
numDatasets = 6 
post_layers = 1
epochs = 2500
learning_rate = 0.001
train_dataset_idx = [0, 1, 2, 3, 4, 5]
test_dataset_idx = [0, 1, 2, 3, 4, 5]
dispProgress = False
numLayers = 2
batch_size = 32
'''

numDatasets = int(os.getenv('numDatasets')) 
gpuID = os.getenv('gpuID')
post_layers = int(os.getenv('post_layers'))
epochs = int(os.getenv('epochs')) 
learning_rate =float(os.getenv('learning_rate')) 
train_dataset_idx = get_list(os.getenv('train_dataset_idx'))
test_dataset_idx = get_list(os.getenv('test_dataset_idx'))
dispProgress = os.getenv('dispProgress')
numLayers = int(os.getenv('numLayers'))
batch_size = int(os.getenv('batch_size'))


# Parameters needed
# numLayers = 5
seed = 23647
split_shuffle = True 
shuffle = True 
num_workers = 2
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
edge_dim = 1
towers = 1
pre_layers = 1
post_layers = 1
divide_input = False
disp_step = 10
train_p = 0.5
val_p = 0.4

root_list = []
for ds in config.dataset:
   root = osp.join(config.dataset_root, ds)
   root_list.append(root)

combined_train_dataset, combined_val_dataset, combined_test_dataset = \
   prep_datasets(numDatasets, train_dataset_idx, test_dataset_idx, train_p, val_p, config, root_list, shuffle=split_shuffle)


print('combined_train_dataset ', len(combined_train_dataset))
print('combined_val_dataset ', len(combined_val_dataset))
print('combined_test_dataset ', len(combined_test_dataset))

#################################################################


in_channels = combined_train_dataset[0].x.size(dim=-1)
print('in_channels ', in_channels)

out_channels = 75
aggregators = ['sum', 'mean', 'min', 'max', 'std']

# Note: set scalers to ['identify'] to factually by-pass using any degree-based scalers
scalers = ['identity'] #, 'amplification', 'attenuation']
scalers2 = ['identity', 'amplification', 'attenuation']
deg = generate_deg(combined_train_dataset)
print('degree', deg)

# Set Dataloader
g = torch.Generator()
g.manual_seed(seed)
train_data_loader = DataLoader(combined_train_dataset, 
                               batch_size=batch_size, 
                               shuffle=shuffle, 
                               num_workers=num_workers, 
                               worker_init_fn=worker_init_fn, 
                               generator=g)

val_data_loader =   DataLoader(combined_val_dataset,
                               batch_size=batch_size,
                               shuffle=shuffle,
                               num_workers=num_workers,
                               worker_init_fn=worker_init_fn,
                               generator=g)

test_data_loader =  DataLoader(combined_test_dataset,
                               batch_size=batch_size,
                               shuffle=shuffle,
                               num_workers=num_workers,
                               worker_init_fn=worker_init_fn,
                               generator=g)


model = MyCEALNetwork( in_channels,
                        out_channels,
                        aggregators,
                        scalers,
                        deg,
                        edge_dim = edge_dim,
                        towers = towers,
                        numLayers = numLayers,
                        pre_layers = pre_layers,
                        post_layers = post_layers,
                        divide_input = False
                    )

model.to(device)


# Set optimizer
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
# Set learning rate scheduler for the optimizer
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.8, patience=50, min_lr=0.000000001)


train(epochs, model, train_data_loader, val_data_loader, test_data_loader, optimizer, scheduler, device, disp_step, batch_size, learning_rate, numLayers, dispProgress=dispProgress)