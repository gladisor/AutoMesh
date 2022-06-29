## allows us to access the automesh library from outside
import os
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
        
    # data.display(30)
    # idx = torch.randperm(len(data))
    # split = 0.90
    # end = int(len(idx) * split)

    # train = torch.utils.data.Subset(data, idx[0:end])
    # val = torch.utils.data.Subset(data, idx[end:len(idx)])

    # train_loader = DataLoader(
    #     dataset = train,
    #     batch_size = 5,
    #     shuffle = True,
    #     drop_last = True)
    
    # model = GCN(
    #     in_channels = 3,
    #     hidden_channels = 128,
    #     num_layers = 2,
    #     out_channels = 1,
    #     act = nn.ReLU)

    # opt = torch.optim.Adam(model.parameters(), lr = 0.001)
    # sigmoid = nn.Sigmoid()
    
    # # loss_func = nn.BCEWithLogitsLoss()
    # loss_func = nn.HuberLoss()

    # for i, epoch in enumerate(range(10)):
    #     for batch in train_loader:
    #         if i == 0:
    #             print("skipping first batch")
    #             break

    #         opt.zero_grad()

    #         logits = model(batch['pos'], batch['edge_index']).squeeze(-1)

    #         ## create target
    #         y = torch.zeros(logits.shape)
    #         y[batch['y']] = 100

    #         loss = loss_func(logits, y)

    #         loss.backward()
    #         opt.step()
    #         print(loss)

    #         for i in range(len(val)):
    #             val_graph = val[i]

    #             logits = model(val_graph.x, val_graph.edge_index).squeeze(-1).detach()

    #             point_preds = logits.topk(k = 8)
    #             point_preds.indices
    #             # print(logits.view(val_batch.num_graphs, -1).shape)

    #             # # top_8 = logits.topk(k = 8)

    #             # branch_verts = val_batch['x'][val_batch['y']]

    #             # print(val_batch.num_graphs)

    #             # num_correct = (p == y).sum()
    #             # num_guesses = p.sum()

    #             # print(num_correct, num_guesses)