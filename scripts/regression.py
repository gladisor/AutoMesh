## allows us to access the automesh library from outside
import os
import sys
from typing import Union
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from typing import List

import torch_geometric.transforms as T
from torch_geometric.nn import GCN, global_mean_pool
from torch_geometric.loader import DataLoader
from torch_geometric.data import Data, Batch
import torch.nn as nn
import torch
import open3d as o3d
import numpy as np

## local source
from automesh.data import LeftAtriumData

class GraphRegressor(nn.Module):
    """
    A multiple regression model for processing a graph into a feature vector which
    can be used to predict a set of points.
    """
    def __init__(self, *args, **kwargs) -> None:
        super().__init__()

        self.base = GCN(
            *args, 
            in_channels = 3, 
            out_channels = 24, 
            **kwargs)

    def forward(self, x: Union[Data, Batch]) -> torch.tensor:
        y = self.base(x.pos, x.edge_index)
        return global_mean_pool(y, x.batch)

    def train(self, loader: DataLoader):
        opt = torch.optim.Adam(self.parameters(), lr = 0.0005)
        loss_func = nn.MSELoss()

        for epoch in range(5):
            for batch in loader:
                if epoch == 0:
                    continue
                
                opt.zero_grad()

                y_hat = self(batch)

                branch_points = []
                for graph in batch.to_data_list():
                    branch_points.append(graph.pos[graph.y])

                y = torch.cat(branch_points)
                y = y.reshape(batch.num_graphs, -1)
                loss = loss_func(y_hat, y)
                loss.backward()
                opt.step()

                print(f"Loss: {loss}")

                test_graph = batch.get_example(0)
                with torch.no_grad():
                    predicted_branch_points = self(test_graph).reshape(8, 3)
                    branch_points = test_graph.pos[test_graph.y]

def create_spheres(pc: o3d.geometry.PointCloud, scale: float, color: np.array) -> List[o3d.geometry.TriangleMesh]:
    spheres = []
    for point in pc.points:
        sphere = o3d.geometry.TriangleMesh.create_sphere()
        sphere.scale(scale, center=sphere.get_center())
        sphere.translate(point)
        sphere.paint_uniform_color(color)
        spheres.append(sphere)

    return spheres

if __name__ == '__main__':

    data = LeftAtriumData(
        root = 'data/GRIPS22/',
        transform = T.Compose([
            T.Center(),
            T.NormalizeScale()
            ]))

    model = GraphRegressor(
        hidden_channels = 128,
        num_layers = 3,
        dropout = 0.0,
        act = torch.relu)

    opt = torch.optim.Adam(model.parameters(), lr = 0.0005)
    loss_func = nn.MSELoss()

    train_loader = DataLoader(
        dataset = data,
        batch_size = 4,
        shuffle = True,
        drop_last = True)

    # model.train(train_loader)

    graph = data[0]

    points = model(graph).reshape(8, 3).detach().numpy()
    points = o3d.utility.Vector3dVector(points)
    points = o3d.geometry.PointCloud(points)

    points = create_spheres(points, 0.005, np.array([1.0, 0.0, 0.0]))

    branch_points = graph.pos[graph.y]
    branch_points = o3d.utility.Vector3dVector(branch_points)
    branch_points = o3d.geometry.PointCloud(branch_points)
    points += create_spheres(branch_points, 0.005, np.array([0.0, 1.0, 0.0]))

    o3d.visualization.draw_geometries(points)
    