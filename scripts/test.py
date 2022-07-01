## allows us to access the automesh library from outside
import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

## third party
import torch_geometric.transforms as T

## local source
from automesh.data import LeftAtriumData, LeftAtriumHeatMapData

if __name__ == '__main__':
    data = LeftAtriumHeatMapData(
        root = 'data/GRIPS22',
        sigma = 1.0,
        transform = T.Compose([
            T.FaceToEdge(),
            T.Center(),
            T.RandomRotate(180, axis = 0),
            T.RandomRotate(180, axis = 1),
            T.RandomRotate(180, axis = 2),
            T.NormalizeScale(),
            ]))

    data.display(5)