## allows us to access the automesh library from outside
import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

## third party
import torch
import torch.nn as nn
import torch_geometric.transforms as T
from torch_geometric.loader import DataLoader
from torch_geometric.nn import MessagePassing, SplineConv, GCN, GAT, GraphSAGE
from torch_geometric.nn.models.basic_gnn import BasicGNN

## local source
from automesh.data.data import LeftAtriumHeatMapData
from automesh.models.heatmap import HeatMapRegressor
from automesh.utils.data import split, preprocess_pipeline, augmentation_pipeline

class SplineNet(BasicGNN):
    def init_conv(self, in_channels: int, out_channels: int,
            **kwargs) -> MessagePassing:
        return SplineConv(in_channels, out_channels, **kwargs)

if __name__ == '__main__':
    data = LeftAtriumHeatMapData(
        root = 'data/GRIPS22',
        sigma = 2.0,
        triangles = 5000,
        transform = T.Compose([
            preprocess_pipeline(),
            augmentation_pipeline(),
            T.Cartesian()
            ]))

    train, val = split(data, 0.9)
    loader = DataLoader(train, batch_size = 4, shuffle = True, drop_last = True)

    model = HeatMapRegressor(
        base = GraphSAGE,
        loss_func = nn.MSELoss(reduction = 'none'),
        lr = 0.001,
        in_channels = 3,
        hidden_channels = 256,
        num_layers = 4,
        out_channels = 8,
        act = torch.relu,
        # edge_dim = 3
        )

    model.train()

    for epoch in range(10):
        for batch in loader:
            model.opt.zero_grad()

            y_hat = model(
                x = batch.pos,
                edge_index = batch.edge_index,
                edge_attr = batch.edge_attr)

            loss = model.calculate_loss(y_hat, batch.y)

            print(f'Train Loss: {loss}')

            loss.backward()
            model.opt.step()

    model.eval()
    val.dataset.visualize_predicted_heat_map(3, model)
    val.dataset.visualize_predicted_heat_map(10, model)
    val.dataset.visualize_predicted_heat_map(5, model)