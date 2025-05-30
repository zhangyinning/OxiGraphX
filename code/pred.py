import torch
from model import *
from torch_geometric.loader import DataLoader
from dataset import *
from utils import *
from utils_data_JSNN import *
import time

def main():
    # Initialize network
    model = MyCEALNetwork(in_channels = 9, 
                out_channels = 75, 
                aggregators = ['sum', 'mean', 'min', 'max', 'std'],
                scalers = ['identity'],
                deg = [],
                numLayers = 2,
                edge_dim = 1,
                towers = 1,
                pre_layers = 1,
                post_layers = 1,
                divide_input = False)
    path = '/gpfs/SHPC_Data/home/yzhang56/pred/compounds-ofe/Results/6_20/ChemGNN_mlp_1_post_Layers2_1719012425277976.pth'  # add model path
    model.load_state_dict(torch.load(path, map_location=torch.device('cpu')))
    model.eval()

    pred_dataset0 = MyDataset(root="data/Sr8La24Co7Ni7Mn7Fe7Cr4O96", config=config, dataset_id=0)
    pred_dataset1 = MyDataset(root="data/Ba4Sr4La24Co7Ni7Mn7Fe7Cr4O96", config=config, dataset_id=1)
    pred_dataset2 = MyDataset(root="data/Ba2Sr6La24Co7Ni7Mn7Fe7Cr4O96", config=config, dataset_id=2)
    pred_dataset3 = MyDataset(root="data/Sr8La24Co7Ni7Mn7Fe7Zn4O96", config=config, dataset_id=3)
    pred_dataset4 = MyDataset(root="data/Sr32Co7Ni7Mn7Fe7Cr4O96", config=config, dataset_id=4)
    pred_dataset5 = MyDataset(root="data/Sr32Co8Ni8Mn8Fe8O96", config=config, dataset_id=5)
    pred_dataset6 = MyDataset(root="data/Ba4Sr4La24Co8Ni8Mn8Fe8O96", config=config, dataset_id=6)
    pred_dataset7 = MyDataset(root="data/La32Co8Ni8Mn8Fe8O96", config=config, dataset_id=7)
    pred_dataset8 = MyDataset(root="data/Sr8La24Co8Ni8Mn8Fe8O96", config=config, dataset_id=8)

    pred_data_loader0 = DataLoader(pred_dataset0,
                               batch_size=64,
                               shuffle=False,
                               num_workers=2)
    pred_data_loader1 = DataLoader(pred_dataset1,
                               batch_size=64,
                               shuffle=False,
                               num_workers=2)
    pred_data_loader2 = DataLoader(pred_dataset2,
                               batch_size=64,
                               shuffle=False,
                               num_workers=2)
    pred_data_loader3 = DataLoader(pred_dataset3,
                               batch_size=64,
                               shuffle=False,
                               num_workers=2)
    pred_data_loader4 = DataLoader(pred_dataset4,
                               batch_size=64,
                               shuffle=False,
                               num_workers=2)
    pred_data_loader5 = DataLoader(pred_dataset5,
                               batch_size=64,
                               shuffle=False,
                               num_workers=2)    
    pred_data_loader6 = DataLoader(pred_dataset6,
                               batch_size=64,
                               shuffle=False,
                               num_workers=2)
    pred_data_loader7 = DataLoader(pred_dataset7,
                               batch_size=64,
                               shuffle=False,
                               num_workers=2)
    pred_data_loader8 = DataLoader(pred_dataset8,
                               batch_size=64,
                               shuffle=False,
                               num_workers=2)


    ymax = 5.10264
    ymin = 0.698339

    data_list = [pred_data_loader0, pred_data_loader1, pred_data_loader2, pred_data_loader3, pred_data_loader4, pred_data_loader5, pred_data_loader6,pred_data_loader7, pred_data_loader8]
    x = 0
    for a in data_list:
       my_out = torch.empty(0)
       my_y = torch.empty(0)
       with torch.no_grad():
           for i, data in enumerate(a):
               out, f = model(data)
               my_out = torch.cat((my_out, out), 0)
               my_y = torch.cat((my_y, data.y), 0)
       
       my_out = my_out*(ymax-ymin) +ymin
       my_y = my_y*(ymax-ymin) +ymin
       folder = './Results/6_20/pred/'+str(x)
       x+=1
       create_folder_if_not_exists(folder)
       timestr = str(int(time.time()*1000000))
       pred_file = timestr + 'y_pred.txt'
       save_result(folder, pred_file, test_out = my_out, test_y = my_y)
       plot_regression(folder, pred_file)


if __name__ == "__main__":
    main()
