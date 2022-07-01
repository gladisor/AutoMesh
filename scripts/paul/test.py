## allows us to access the automesh library from outside
import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

## third party
from torch_geometric.nn import GCN
from torch_geometric.loader import DataLoader
import torch_geometric.transforms as T
import torch.nn as nn
import torch

## local source
from automesh.data.data import LeftAtriumData
from automesh.data.data_meshcnn import MeshCNNLeftAtriumData
from automesh.models.meshcnn.base_options import BaseOptions

if __name__ == '__main__':

    data = LeftAtriumData(
        root = 'data/GRIPS22/',
        transform = T.Compose([
            T.Center(),
            T.NormalizeScale(),
        ]))
    meta = MeshCNNLeftAtriumDatadef ( BaseOptions, mesh_paths)
    
print (meta)