from typing import Callable, Union

import torch.nn as nn
import torch
from torch_geometric.nn import GraphSAGE
from torch_geometric.data import Data, Batch

class HeatMapRegressor(nn.Module):
    def __init__(self, 
        hidden_channels: int, 
        num_layers: int, 
        num_landmarks: int, 
        dropout: float,
        act: Callable,
        lr: float):
        super().__init__()

        self.base = GraphSAGE(
            in_channels = 3,
            hidden_channels = hidden_channels,
            num_layers = num_layers,
            out_channels = num_landmarks,
            dropout = dropout,
            act = act,
            # edge_dim = 4
            )

        self.opt = torch.optim.Adam(self.parameters(), lr = lr)
        self.loss_func = nn.MSELoss(reduction = 'none')

    def forward(self, x: Union[Data, Batch]) -> torch.tensor:
        return self.base(x.pos, x.edge_index)

    def calculate_loss(
            self, 
            y_hat: torch.tensor, 
            y: torch.tensor, 
            weight: float = 10.0,
            threshold: float = 0.2) -> torch.tensor:

        M = torch.zeros(y.shape)
        M[y > threshold] = 1
        loss = (self.loss_func(y_hat, y) *  (M * weight + 1)).mean()
        return loss