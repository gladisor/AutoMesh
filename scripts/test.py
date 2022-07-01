## allows us to access the automesh library from outside
import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import copy

## third party
import torch_geometric.transforms as T
import open3d as o3d
import numpy as np

## local source
from automesh.data.data import LeftAtriumData, LeftAtriumHeatMapData

if __name__ == '__main__':
    data = LeftAtriumHeatMapData(
        root = 'data/GRIPS22',
        sigma = 1,
        triangles = 5000,
        transform = T.Compose([
            T.FaceToEdge(),
            T.Center(),
            T.RandomRotate(180, axis = 0),
            T.RandomRotate(180, axis = 1),
            T.RandomRotate(180, axis = 2),
            T.NormalizeScale(),
            ]))

    data.display(5)