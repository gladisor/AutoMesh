import torch.nn as nn
from torch_geometric.nn import GCN
from torch_geometric.data import Data, Batch

class BaseGCN(GCN):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

    def forward(self, x):
        if type(x) == Data:
            return self(x.x, x.edge_index)
        elif type(x) == Batch:
            return self(x['pos'], x['edge_index'])
        else:
            raise AttributeError("Incorrect data type. Expected Data or Batch")

class PointClassifier(BaseGCN):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

if __name__ == '__main__':
    model = BaseGCN(
        in_channels = 3,
        hidden_channels = 128,
        num_layers = 2,
        out_channels = 1,
        act = nn.ReLU())

    print(model)

