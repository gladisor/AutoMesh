import torch
from typing import Callable

import torch_geometric.transforms as T

from torch_geometric.data import Data

def preprocess_pipeline() -> Callable:
    return T.Compose([
        T.FaceToEdge(remove_faces = False),
        T.Center(),
        T.NormalizeScale(),
    ])
    
def rotation_pipeline(degrees: float = 20) -> Callable:
    return T.Compose([
        T.RandomRotate(degrees, axis = 0),
        T.RandomRotate(degrees, axis = 1),
        T.RandomRotate(degrees, axis = 2),
    ])

class AutoMeshVirtualNode(T.VirtualNode):
    def __call__(self, data: Data) -> Data:
        data=super().__call__(data)
        
        data.pos=torch.cat((data.pos, torch.zeros(1,data.pos.shape[1])),0)
        data.y=torch.cat((data.y, torch.zeros(1,data.y.shape[1])),0)
        return data
