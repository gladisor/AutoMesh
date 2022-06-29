## allows us to access the automesh library from outside
import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import torch_geometric.transforms as T
from torch_geometric.nn import GCNConv, global_mean_pool
from torch_geometric.loader import DataLoader
import torch.nn as nn
import torch

## local source
from automesh.data import LeftAtriumData

class GraphRegressor(nn.Module):
    def __init__(self, in_channels, out_channels) -> None:
        super().__init__()

        self.g = GCNConv(in_channels, out_channels)

    def forward(self, v, e, batch) -> torch.tensor:
        return global_mean_pool(self.g(v, e).relu(), batch)

if __name__ == '__main__':
    data = LeftAtriumData(
        root = 'data/GRIPS22/',
        transform = T.Compose([
            T.Center(),
            T.NormalizeScale(),
        ]))

    train_loader = DataLoader(
        dataset = data,
        batch_size = 4,
        shuffle = True,
        drop_last = True)

    model = GraphRegressor(3, 24)
    opt = torch.optim.Adam(model.parameters(), lr = 0.001)
    loss_func = nn.MSELoss()

    for epoch in range(10):
        for batch in train_loader:
            if epoch == 0:
                continue
            
            opt.zero_grad()
            y_hat = model(batch.pos, batch.edge_index, batch.batch)

            branch_points = []
            for graph in batch.to_data_list():
                branch_points.append(graph.pos[graph.y])

            y = torch.cat(branch_points)
            y = y.reshape(batch.num_graphs, -1)
            loss = loss_func(y_hat, y)
            loss.backward()
            opt.step()

            print(loss)