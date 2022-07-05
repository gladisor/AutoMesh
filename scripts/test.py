## allows us to access the automesh library from outside
import os
import sys
from typing import Callable, Tuple, Union
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import copy

## third party
import torch_geometric.transforms as T
from torch_geometric.nn import GCN, GAT
from torch_geometric.loader import DataLoader
from torch_geometric.data import Data, Batch, Dataset
from torch.utils.data import Subset
import torch.nn as nn
import torch
import numpy as np
from scipy.spatial.distance import pdist, squareform
import open3d as o3d

## local source
from automesh.data.data import LeftAtriumHeatMapData

class HeatMapRegressor(nn.Module):
    def __init__(self, 
        hidden_channels: int, 
        num_layers: int, 
        num_landmarks: int, 
        dropout: float,
        act: Callable):
        super().__init__()

        self.base = GAT(
            in_channels = 3,
            hidden_channels = hidden_channels,
            num_layers = num_layers,
            out_channels = num_landmarks,
            dropout = dropout,
            act = act)

    def forward(self, x: Union[Data, Batch]) -> torch.tensor:
        # return torch.sigmoid(self.base(x.pos, x. edge_index))
        return self.base(x.pos, x.edge_index)

def split(data: Dataset, p: float) -> Tuple[Subset, Subset]:
    idx = torch.randperm(len(data))
    split_idx = int(len(data) * p)
    train = torch.utils.data.Subset(data, idx[0:split_idx])
    val = torch.utils.data.Subset(data, idx[split_idx:len(data)])

    return (train, val)

def preprocess_pipeline() -> Callable:
    return T.Compose([
        T.FaceToEdge(),
        T.Center(),
        T.NormalizeScale(),
    ])
    
def augmentation_pipeline() -> Callable:
    return T.Compose([
        T.RandomRotate(20, axis = 0),
        T.RandomRotate(20, axis = 1),
        T.RandomRotate(20, axis = 2),
    ])

if __name__ == '__main__':
    data = LeftAtriumHeatMapData(
        root = 'data/GRIPS22',
        sigma = 4.0,
        triangles = 5000,
        transform = T.Compose([
            preprocess_pipeline()
            ])
        )

    train, val = split(data, 0.9)

    model = HeatMapRegressor(256, 4, 8, 0.1, torch.relu)
    opt = torch.optim.Adam(model.parameters(), lr = 0.001)
    # loss_func = nn.BCEWithLogitsLoss(
        # pos_weight = torch.ones(1) * 20, 
        # reduction = 'none')

    loss_func = nn.MSELoss(reduction = 'none')
    loader = DataLoader(train, batch_size = 4, shuffle = True, drop_last = True)

    model.train()
    for epoch in range(10):
        for batch in loader:
            opt.zero_grad()

            y_hat = model(batch)
            # loss = loss_func(y_hat, batch.y[:, 6].unsqueeze(-1)).mean()
            M = torch.zeros(batch.y.shape)
            M[batch.y > 0.2] = 1

            loss = (loss_func(y_hat, batch.y) *  (M * 10 + 1)).mean()

            # loss = (loss_func(y_hat, batch.y) * M).mean()

            # loss = loss_func(y_hat, batch.y[:, [6, 7]]).mean(dim = -1).mean()
            # print(batch.y[:, [6, 7]].max(dim = 0))
        #     break
        # break

            print(loss)
            loss.backward()
            opt.step()

    model.eval()

    val.dataset.visualize_predicted_heat_map(3, model)
    val.dataset.visualize_predicted_heat_map(10, model)
    val.dataset.visualize_predicted_heat_map(5, model)