import os
import os.path as osp
import torch
from tqdm import tqdm

from torch_geometric.data import (
    Data,
    InMemoryDataset,
)

from torch.utils.data import ConcatDataset
from torch_geometric import utils

from config import config
from utils_data_JSNN import *
from sklearn.preprocessing import MinMaxScaler



"""
Custom PyG dataset for molecule property prediction

Note: This is just for ONE compound. 
      If you want to use data from multiple compounds, after using this class, use the MyDataset_combined.

"""



class MyDataset(InMemoryDataset):

   def __init__(self, root, config, dataset_id, transform=None, pre_transform=None, pre_filter=None):
      """
      root: the root of dataset. eg. 'dataset'
      """

      self.config = config
      self.dataset_id = dataset_id
      self.root = root

      super().__init__(root, transform, pre_transform, pre_filter)

      # Set processed data file folder and filename
      # eg. dataset/JSNN_DATA/LaCoO3_DATA/processed/train.pt

      path = osp.join(self.my_processed_dir, 'data.pt')

      # Load self.data (type: torch_geometric.data.data.Data) and self.slices
      self.data, self.slices = torch.load(path)


   # raw files names in root/raw folder if found, skip download
   @property
   def raw_file_names(self):
      return [ ]


   # Eg. dataset/JSNN_DATA/LaCoO3_DATA/processed
   @property
   def my_processed_dir(self):
      return osp.join(self.root, 'processed')

   # processed data file names in root/process_dir, if found skip process
   @property
   def processed_file_names(self):
      return ['data.pt']

   # Maybe later to implement the download method to download raw data files
   def download(self):
      pass

   # Process the raw data and save it to my_processed_dir
   def process(self):
      # The following is to add Data to a list.
      # This is the standard operation when prepare the process() method for Dataset

      # datasrc = osp.join(self.root,self.config.dataset[self.dataset_id])
      datasrc = self.root

      print('datasrc ', datasrc)

      if not os.path.exists(self.my_processed_dir):
         os.makedirs(self.my_processed_dir)
     
      # Set the progress bar
      filepath = osp.join(datasrc, 'CONFIGS')
      numFiles = get_numFiles(filepath)

      pbar = tqdm(total=numFiles)
      description = 'Processing dataset: ' + self.config.dataset[self.dataset_id]
      pbar.set_description(description)

      eng_file = 'DEFECT_ENERGY_EV'
      enrg_list = load_energyList(datasrc, eng_file)
      enrg_list = [(sub-0.698339)/(5.10264-0.698339) for sub in enrg_list]   # normalizing target
      enrg_list = torch.tensor(enrg_list, dtype=torch.float32)
 
      data_list = []
      for id in range(1, 101):  # chaneg this to numFiles
         for x in range(1, 11):
         # Load a molecule's data
            data = read_one_molecule(datasrc, id, x, enrg_list)   
            if self.pre_filter is not None and not self.pre_filter(data):
                continue

            if self.pre_transform is not None:
                data = self.pre_transform(data)

            # Append data to the data_list
            #if data.y.item()< (-2487+2488.818759)/(-2486.123849+2488.818759):
            data_list.append(data)
            pbar.update(1)
            # end of for idx

      pbar.close()

      # Store the processed data
      """
      Because saving a huge python list is really slow, we collate the list into one huge 
      torch_geometric.data.Data object via torch_geometric.data.InMemoryDataset.collate() 
      before saving it.

      The collated data object has concatenated all examples into one big data object and, 
      in addition, returns a slices dictionary to reconstruct single examples from this object. 

      Finally, we need to load these two objects in the constructor into the properties 
      self.data and self.slices. (see the constructor of this class)
      """
      print('Length of data list', len(data_list))
      torch.save(self.collate(data_list), osp.join(self.my_processed_dir, 'data.pt'))


def intersection(lst1, lst2):
    lst3 = [value for value in lst1 if value in lst2]
    return lst3

def prep_datasets(numDatasets, train_dataset_idx, test_dataset_idx, train_p, val_p, config, root_list, shuffle=True):

   dataset_idx = list(range(numDatasets)) 

   trainset_list_reserved = []
   valset_list_reserved = []
   testset_list_reserved = []
   for id in dataset_idx:
      mydataset = MyDataset(root=root_list[id], config=config, dataset_id=id)
      

      '''
      print("point 0\n") # checking dataset
      print(mydataset[0].edge_index)
      print(mydataset[0].edge_index.shape)
      print(mydataset[0].x)
      print(mydataset[0].y)


      print("point 30\n")
      print(mydataset[30].edge_index)
      print(mydataset[30].edge_index.shape)
      print(mydataset[30].x)
      print(mydataset[30].y)
      '''

      tr_dataset, v_dataset, te_dataset = split_dataset(mydataset, train_p, val_p, shuffle=shuffle)
      trainset_list_reserved.append(tr_dataset)
      valset_list_reserved.append(v_dataset)
      testset_list_reserved.append(te_dataset)

   trainset_list = []
   valset_list = []
   testset_list = []
   for id in train_dataset_idx:
      trainset_list.append(trainset_list_reserved[id])
      valset_list.append(valset_list_reserved[id])
   for id in test_dataset_idx:
      if id in train_dataset_idx:
         teset = testset_list_reserved[id]
      else:
         teset = MyDataset(root=root_list[id], config=config, dataset_id=id)
      testset_list.append(teset)
     
   combined_train_dataset  = ConcatDataset(trainset_list)
   combined_val_dataset = ConcatDataset(valset_list)
   combined_test_dataset = ConcatDataset(testset_list)
   return combined_train_dataset, combined_val_dataset, combined_test_dataset












































































