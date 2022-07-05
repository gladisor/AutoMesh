from typing import Tuple, Callable

import torch
from torch.utils.data import Subset
from torch_geometric.data import Dataset
import torch_geometric.transforms as T

def split(data: Dataset, p: float) -> Tuple[Subset, Subset]:
    idx = torch.randperm(len(data))
    split_idx = int(len(data) * p)
    train = torch.utils.data.Subset(data, idx[0:split_idx])
    val = torch.utils.data.Subset(data, idx[split_idx:len(data)])

    return (train, val)

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