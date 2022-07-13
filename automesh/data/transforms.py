from typing import Callable

import torch_geometric.transforms as T

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