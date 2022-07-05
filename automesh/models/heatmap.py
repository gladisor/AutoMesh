from ast import Call
from typing import Callable, Union

import torch
import torch.nn as nn
from torch_geometric.nn import GraphSAGE
from torch_geometric.nn.models.basic_gnn import BasicGNN
from torch_geometric.data import Data, Batch

class HeatMapRegressor(nn.Module):
    def __init__(self,
        base: BasicGNN,
        loss_func: Callable,
        lr: float,
        **kwargs):
    
        super().__init__()

        self.base = base(**kwargs)

        self.opt = torch.optim.Adam(self.parameters(), lr = lr)
        self.loss_func = loss_func

    def forward(self, **kwargs) -> torch.tensor:
        return self.base(**kwargs)

    def calculate_loss(
            self, 
            y_hat: torch.tensor, 
            y: torch.tensor, 
            weight: float = 10.0,
            threshold: float = 0.2) -> torch.tensor:

        # M = torch.zeros(y.shape)
        # M[y > threshold] = 1
        # loss = (self.loss_func(y_hat, y) *  (M * weight + 1)).mean()

        loss = 0.0
        for c in range(y_hat.shape[1]):
            loss += self.loss_func(y_hat[:, c], y[:, c])

        return loss.mean()