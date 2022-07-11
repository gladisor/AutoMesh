from typing import Callable

import torch_geometric.transforms as T

def preprocess_pipeline() -> Callable:
    return T.Compose([
        T.FaceToEdge(),
        T.Center(),
        T.NormalizeScale(),
    ])
    
def augmentation_pipeline() -> Callable:
    return T.Compose([
        T.RandomRotate(20, axis = 0),
        T.RandomRotate(20, axis = 1),
        T.RandomRotate(20, axis = 2),
    ])