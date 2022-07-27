## allows us to access the automesh library from outside
import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from torch_geometric.nn import GIN, GraphNorm, GraphSizeNorm
import torch.nn as nn

if __name__ == '__main__':

    model = GIN(
        in_channels = 3,
        hidden_channels = 100,
        num_layers = 2,
        out_channels = 8,
        dropout = 0.20,
        act = nn.LeakyReLU,
        act_kwargs = {'negative_slope': 0.001},
        norm = GraphSizeNorm())

    print(model)
