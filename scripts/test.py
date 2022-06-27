## allows us to access the automesh library from outside
from json import load
import os
from random import shuffle
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

## third party
from torch_geometric.nn import GCN
from torch_geometric.loader import DataLoader
import torch_geometric.transforms as T
import torch.nn as nn
import torch

## local source
from automesh.data import LeftAtriumData

if __name__ == '__main__':

    data = LeftAtriumData(
        root = 'data/GRIPS22/', 
        transform = T.Compose([
            T.Center(),
            T.NormalizeScale(),
        ]))

    loader = DataLoader(
        dataset = data,
        batch_size = 5,
        shuffle = True,
        drop_last = True)
    
    model = GCN(
        in_channels = 3,
        hidden_channels = 128,
        num_layers = 2,
        out_channels = 1,
        act = nn.ReLU)

    opt = torch.optim.Adam(model.parameters(), lr = 0.001)
    sigmoid = nn.Sigmoid()

    for i, epoch in enumerate(range(10)):
        for batch in loader:
            if i == 0:
                print("skipping first batch")
                break

            opt.zero_grad()

            logits = model(batch['pos'], batch['edge_index']).squeeze(-1)

            y = torch.zeros(logits.shape)
            y[batch['y']] = 1

            p = sigmoid(logits)
            loss = - y.dot(torch.log(p)) - (1 - y) * torch.log(1 - p)
            loss.backward()
            opt.step()

            print(loss.detach())
        

