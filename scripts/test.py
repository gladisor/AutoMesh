## allows us to access the automesh library from outside
import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

## third party
import open3d as o3d
import numpy as np
from torch.utils.data import Dataset
import torch_geometric.nn as gnn
from torch_geometric.nn import GCNConv
from pathlib import Path

## local source
from automesh.data import LeftAtriumData



if __name__ == '__main__':
    data = LeftAtriumData('data/GRIPS22/')

    m = GCNConv(3, 1)

    v, e, b = data[0]
    y = m(v, e.T)
    
    print(len(y))