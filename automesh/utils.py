from typing import Tuple

import torch
from torch.utils.data import Subset
from torch_geometric.data import Dataset

def split(data: Dataset, p: float) -> Tuple[Subset, Subset]:
    idx = torch.randperm(len(data))
    split_idx = int(len(data) * p)
    train = torch.utils.data.Subset(data, idx[0:split_idx])
    val = torch.utils.data.Subset(data, idx[split_idx:len(data)])

    return (train, val)