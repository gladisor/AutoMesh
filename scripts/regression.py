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
from scipy.spatial import distance
from scipy.special import softmax


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
        return torch.tanh(global_mean_pool(y, x.batch))

def create_spheres(pc: o3d.geometry.PointCloud, scale: float, color: np.array) -> List[o3d.geometry.TriangleMesh]:
    spheres = []
    for point in pc.points:
        sphere = o3d.geometry.TriangleMesh.create_sphere()
        sphere.scale(scale, center=sphere.get_center())
        sphere.translate(point)
        sphere.paint_uniform_color(color)
        spheres.append(sphere)

    return spheres

def batch_branch_points(batch: Batch) -> torch.tensor:

        branch_points = []
        for graph in batch.to_data_list():
            branch_points.append(graph.pos[graph.y])

        y = torch.cat(branch_points)
        y = y.reshape(batch.num_graphs, -1)

        return y

class RandomScaleAxis(T.BaseTransform):
    def __init__(self, scale: float, axis: int) -> None:
        super().__init__()

        self.scale = scale
        self.axis = 0

    def __call__(self, data: Data) -> Data:
        torch.random.uniform(-self.scale, self.scale)

if __name__ == '__main__':

    data = LeftAtriumData(
        root = 'data/GRIPS22/',
        transform = T.Compose([
            T.ToUndirected(),
            T.Center(),
            T.RandomRotate(180, axis = 0),
            T.RandomRotate(180, axis = 1),
            T.RandomRotate(180, axis = 2),
            T.NormalizeScale(),
            ]))

    idx = torch.randperm(len(data))
    split = int(len(data) * 0.9)

    train_data = DataLoader(
        dataset = torch.utils.data.Subset(data, idx[:split]),
        batch_size = 4,
        shuffle = True,
        drop_last = True)

    val_data = torch.utils.data.Subset(data, idx[split:])
    val_batch = Batch.from_data_list(list(val_data[:len(val_data)]))

    model = GraphRegressor(
        hidden_channels = 128,
        num_layers = 3,
        dropout = 0.0,
        act = torch.relu)

    opt = torch.optim.SGD(model.parameters(), lr = 0.001)
    loss_func = nn.MSELoss()

    for epoch in range(5):
        for batch in train_data:
            opt.zero_grad()

            y_hat = model(batch)
            y = batch_branch_points(batch)

            loss = loss_func(y_hat, y)
            loss.backward()
            opt.step()

            with torch.no_grad():
                val_y_hat = model(val_batch)
                val_y = batch_branch_points(val_batch)
                val_loss = loss_func(val_y_hat, val_y)

            print(f"Train Loss: {loss}, Val Loss: {val_loss}")

    # # graph = data[0]

    # # points = model(graph).reshape(8, 3).detach().numpy()
    # # points = o3d.utility.Vector3dVector(points)
    # # points = o3d.geometry.PointCloud(points)

    # # points = create_spheres(points, 0.005, np.array([1.0, 0.0, 0.0]))

    # # branch_points = graph.pos[graph.y]
    # # branch_points = o3d.utility.Vector3dVector(branch_points)
    # # branch_points = o3d.geometry.PointCloud(branch_points)
    # # points += create_spheres(branch_points, 0.005, np.array([0.0, 1.0, 0.0]))

    # # o3d.visualization.draw_geometries(points)
    