import math
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
from torch.nn import Linear, ReLU, Sequential
from torch_geometric.nn import GCNConv, BatchNorm, global_add_pool, global_mean_pool
from tqdm import tqdm
import os.path as osp
from sklearn.metrics import r2_score
import numpy as np
from sklearn.metrics import pairwise_distances

from ceal import *
from utils import *
from dataset import MyDataset
from config import *
from pna import PNAConv


class MyCEALNetwork(torch.nn.Module):
   """
   in_channels,
   out_channels,
   aggregators,

   scalers,
   deg,
   edge_dim = None,
   towers = 1,
   pre_layers = 1,
   post_layers = 1,
   divide_input = False,
   
   """   
   def __init__(self, 
                in_channels, 
                out_channels, 
                aggregators,
                scalers,
                deg,
                numLayers = 2,
                edge_dim = None,
                towers = 1,
                pre_layers = 1,
                post_layers = 1,
                divide_input = False
               ):

      super().__init__()

      self.out_channels = out_channels
      self.numLayers = numLayers

        # The first CEAL layer
      self.ceal_1 = CEALConv( in_channels = in_channels,
                                out_channels = out_channels,
                                aggregators = aggregators,
                                scalers = scalers,
                                deg = deg,
                                edge_dim = edge_dim,
                                towers = 1,
                                pre_layers = 1,
                                post_layers = 1,
                                divide_input = False
                            )
      self.batch_norm_1 = BatchNorm(self.out_channels)

      # This container holds 2nd and more ceal layers
      self.cealConvs = ModuleList()
      self.batch_norms = ModuleList()
      for _ in range(1, self.numLayers):
        cealconv = CEALConv( in_channels = out_channels,
                                out_channels = out_channels,
                                aggregators = aggregators,
                                scalers = scalers,
                                deg = deg,
                                edge_dim = edge_dim,
                                towers = 1,
                                pre_layers = 1,
                                post_layers = 1,
                                divide_input = False
                            )
        batch_norm = BatchNorm(self.out_channels)
        self.cealConvs.append(cealconv)
        self.batch_norms.append(batch_norm)

      self.pre_mlp = Sequential(Linear(in_channels, out_channels//2), ReLU(), Linear(out_channels//2, out_channels))
      self.post_mlp = Sequential(Linear(self.out_channels, 40), ReLU(), Linear(40, 1))
   
   
   def forward(self, batch_data):
      # Note: batch_data is provided by Dataloader
      x, edge_index, edge_attr = batch_data.x, batch_data.edge_index, batch_data.edge_attr
      
      # x = self.pre_mlp(x)
      x = self.ceal_1(x, edge_index, batch_data.batch, edge_attr)
      x = self.batch_norm_1(x)
      x = F.relu(x)
      #x = F.dropout(x, p=0.5, training=self.training)

      for cealconv, batch_norm in zip(self.cealConvs, self.batch_norms):
          x = batch_norm(cealconv(x, edge_index, batch_data.batch, edge_attr))
          x = F.relu(x)
          #x = F.dropout(x, p=0.2,  training=self.training)
      
      x = global_mean_pool(x, batch_data.batch)
      out = self.post_mlp(x)
      return out
























