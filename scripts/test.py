## allows us to access the automesh library from outside
import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

## third party
import torch
import torch.nn as nn
import torch_geometric.transforms as T
from torch_geometric.data import LightningDataset
from torch_geometric.nn import GCN, GAT, GraphSAGE, GraphUNet
from pytorch_lightning import Trainer
from pytorch_lightning.plugins import SingleDevicePlugin
from pytorch_lightning.loggers import CSVLogger

## local source
from automesh.data.data import LeftAtriumHeatMapData
from automesh.models.heatmap import HeatMapRegressor
from automesh.loss import AdaptiveWingLoss
from automesh.data.transforms import preprocess_pipeline, augmentation_pipeline

if __name__ == '__main__':

    transform = T.Compose([
        preprocess_pipeline(),
        augmentation_pipeline(),
        # T.Cartesian()
        ])

    train = LeftAtriumHeatMapData(
        root = 'data/GRIPS22/train', 
        sigma = 2.0,
        transform = transform)

    val = LeftAtriumHeatMapData(
        root = 'data/GRIPS22/val', 
        sigma = 2.0,
        transform = transform)

    batch_size = 4

    data = LightningDataset(
        train_dataset = train,
        val_dataset = val,
        batch_size = batch_size,
        drop_last = True,
        # num_workers = 4
        )

    # model = HeatMapRegressor(
    #     base = GraphSAGE,
    #     loss_func = AdaptiveWingLoss(),
    #     optimizer = torch.optim.Adam,
    #     lr = 0.001,
    #     in_channels = 3,
    #     # edge_dim = 3,
    #     hidden_channels = 256,
    #     num_layers = 4,
    #     out_channels = 8,
    #     act = torch.relu)

    # logger = CSVLogger(save_dir = 'results', name = 'HeatMapRegressor')

    # trainer = Trainer(
    #     strategy = SingleDevicePlugin(),
    #     max_epochs = 4,
    #     logger = logger,
    #     log_every_n_steps = int(len(train) / batch_size)
    #     )

    # trainer.fit(model, data)

    model = HeatMapRegressor.load_from_checkpoint('results/HeatMapRegressor/version_0/checkpoints/epoch=3-step=64.ckpt')

    for i in range(len(val)):
        val.visualize_predicted_heat_map(i, model)
        val.visualize_predicted_points(i, model)