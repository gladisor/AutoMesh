## allows us to access the automesh library from outside
import os
import sys
from typing import Callable, Union
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import copy

## third party
import torch_geometric.transforms as T
from torch_geometric.nn import GCN
from torch_geometric.loader import DataLoader
from torch_geometric.data import Data, Batch
import torch
import torch.nn as nn
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
        act: Callable):
        super().__init__()

        self.base = GCN(
            in_channels = 3,
            hidden_channels = hidden_channels,
            num_layers = num_layers,
            out_channels = num_landmarks,
            act = act)

    def forward(self, x: Union[Data, Batch]) -> torch.tensor:
        return torch.sigmoid(self.base(x.pos, x. edge_index))

if __name__ == '__main__':
    data = LeftAtriumHeatMapData(
        root = 'data/GRIPS22',
        sigma = 7.0,
        triangles = 5000,
        transform = T.Compose([
            T.FaceToEdge(),
            T.Center(),
            T.RandomRotate(20, axis = 0),
            T.RandomRotate(20, axis = 1),
            T.RandomRotate(20, axis = 2),
            T.NormalizeScale(),
            ]))

    idx = torch.randperm(len(data))
    split = int(len(data) * 0.9)
    train = torch.utils.data.Subset(data, idx[0:split])
    val = torch.utils.data.Subset(data, idx[split:len(data)])

    model = HeatMapRegressor(128, 4, 1, torch.relu)
    opt = torch.optim.Adam(model.parameters(), lr = 0.001)
    loss_func = nn.BCELoss()
    loader = DataLoader(train, batch_size = 4, shuffle = True, drop_last = True)

    for epoch in range(10):
        for batch in loader:
            opt.zero_grad()

            y_hat = model(batch)
            loss = loss_func(y_hat, batch.y[:, 6].unsqueeze(-1))

            print(loss)
            loss.backward()
            opt.step()

    val.dataset.visualize_predicted_heat_map(3, model)
    val.dataset.visualize_predicted_heat_map(10, model)
    val.dataset.visualize_predicted_heat_map(5, model)