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

## adapted from:
# https://github.com/elliottzheng/AdaptiveWingLoss/blob/master/adaptive_wing_loss.py
class AdaptiveWingLoss(nn.Module):
    def __init__(
        self, 
        omega: float = 14.0, 
        theta: float = 0.5,
        epsilon: float = 1.0,
        alpha: float = 2.1) -> None:
        super().__init__()

        self.omega = omega
        self.theta = theta
        self.epsilon = epsilon
        self.alpha = alpha
    
    def forward(self, y_hat, y) -> torch.tensor:
        delta_y = (y - y_hat).abs()

        delta_y1 = delta_y[delta_y < self.theta]
        delta_y2 = delta_y[delta_y >= self.theta]

        y1 = y[delta_y < self.theta]
        y2 = y[delta_y >= self.theta]

        loss1 = self.omega * torch.log(1 + torch.pow(delta_y1 / self.omega, self.alpha - y1))
        
        A = self.omega * (1 / (1 + torch.pow(self.theta / self.epsilon, self.alpha - y2))) * (self.alpha - y2) * (
            torch.pow(self.theta / self.epsilon, self.alpha - y2 - 1)) * (1 / self.epsilon)

        C = self.theta * A - self.omega * torch.log(1 + torch.pow(self.theta / self.epsilon, self.alpha - y2))
        loss2 = A * delta_y2 - C

        return (loss1.sum() + loss2.sum()) / (len(loss1) + len(loss2))

if __name__ == '__main__':
    data = LeftAtriumHeatMapData(
        root = 'data/GRIPS22',
        sigma = 2.0,
        triangles = 5000,
        transform = T.Compose([
            preprocess_pipeline(),
            augmentation_pipeline(),
            # T.Cartesian()
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

    loss_func = AdaptiveWingLoss()

    model.train()

    for epoch in range(10):
        for batch in loader:
            model.opt.zero_grad()

            y_hat = model(
                x = batch.pos,
                edge_index = batch.edge_index,
                # edge_attr = batch.edge_attr
                )

            # loss = model.calculate_loss(y_hat, batch.y)
            loss = 0.0
            for c in range(y_hat.shape[1]):
                loss += loss_func(y_hat[:, c], batch.y[:, c])

            print(f'Train Loss: {loss}')

            loss.backward()
            model.opt.step()

    model.eval()
    val.dataset.visualize_predicted_heat_map(3, model)
    val.dataset.visualize_predicted_heat_map(10, model)
    val.dataset.visualize_predicted_heat_map(5, model)