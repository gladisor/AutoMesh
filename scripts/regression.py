## allows us to access the automesh library from outside
import os
import sys
from typing import Union
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import torch_geometric.transforms as T
from torch_geometric.nn import GCN, global_mean_pool
from torch_geometric.loader import DataLoader
from torch_geometric.data import Data, Batch
import torch.nn as nn
import torch

## local source
from automesh.data import LeftAtriumData

class GraphRegressor(nn.Module):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__()

        self.base = GCN(*args, **kwargs)

    def forward(self, x: Union[Data, Batch]) -> torch.tensor:
        y = self.base(x.pos, x.edge_index)
        return global_mean_pool(y, x.batch)

        # return global_mean_pool(self.g(v, e).relu(), batch)

if __name__ == '__main__':

    data = LeftAtriumData(
        root = 'data/GRIPS22/',
        transform = T.Compose([
            T.Center(),
            T.NormalizeScale(),
        ]))

    model = GraphRegressor(
        in_channels = 3,
        hidden_channels = 128,
        num_layers = 3,
        out_channels = 24,
        dropout = 0.0,
        act = torch.relu)

    opt = torch.optim.Adam(model.parameters(), lr = 0.001)
    loss_func = nn.MSELoss()

    train_loader = DataLoader(
        dataset = data,
        batch_size = 4,
        shuffle = True,
        drop_last = True)

    for epoch in range(10):
        for batch in train_loader:
            if epoch == 0:
                continue
            
            opt.zero_grad()

            y_hat = model(batch)
            
            branch_points = []
            for graph in batch.to_data_list():
                branch_points.append(graph.pos[graph.y])

            y = torch.cat(branch_points)
            y = y.reshape(batch.num_graphs, -1)
            loss = loss_func(y_hat, y)
            loss.backward()
            opt.step()

            print(loss)