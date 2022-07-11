## allows us to access the automesh library from outside
import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

## third party
import torch
import torch.nn as nn
import torch_geometric.transforms as T
from torch_geometric.data import LightningDataset
from torch_geometric.nn import GCN, GAT, GraphSAGE, GraphUNet
from pytorch_lightning import Trainer
from pytorch_lightning.plugins import SingleDevicePlugin

## local source
from automesh.data.data import LeftAtriumHeatMapData
from automesh.models.heatmap import HeatMapRegressor
from automesh.loss import AdaptiveWingLoss
from automesh.data.transforms import preprocess_pipeline, augmentation_pipeline
print('Hello')
if __name__ == '__main__':

    transform = T.Compose([
        preprocess_pipeline(),
        augmentation_pipeline(),
        T.Cartesian()
        ])

    train = LeftAtriumHeatMapData(
        root = 'data/GRIPS22/train', 
        sigma = 2.0,
        transform = transform)
    
    x=train[5]
    
    model = HeatMapRegressor(
         base = GAT,
         loss_func = nn.MSELoss(),
         optimizer = torch.optim.Adam,
         lr = 0.0005,
         in_channels = 3,
         edge_dim = 3,
         hidden_channels = 256,
         num_layers = 4,
         out_channels = 8,
         act = torch.relu)
    
    
    
    HeatMapRegressor.predict_points(model(x), x)