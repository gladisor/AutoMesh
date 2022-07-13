## allows us to access the automesh library from outside
import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

## third party
import torch
import torch.nn as nn
import torch_geometric.transforms as T
from torch_geometric.data import LightningDataset
from torch_geometric.nn import GraphSAGE, GraphNorm, GCNConv, SAGEConv, GATConv
from pytorch_lightning import Trainer
from pytorch_lightning.plugins import DDPSpawnPlugin
from pytorch_lightning.loggers import CSVLogger

## local source
from automesh.models.architectures import ParamGCN
from automesh.data.data import LeftAtriumHeatMapData
from automesh.models.heatmap import HeatMapRegressor
from automesh.loss import (
    AdaptiveWingLoss, DiceLoss, BCEDiceLoss, 
    JaccardLoss, FocalLoss, TverskyLoss, FocalTverskyLoss)
from automesh.data.transforms import preprocess_pipeline, rotation_pipeline

if __name__ == '__main__':
    transform = T.Compose([
        preprocess_pipeline(), 
        rotation_pipeline(),
        ])

    train = LeftAtriumHeatMapData(root = 'data/GRIPS22/train', sigma = 2.0, transform = transform)
    val = LeftAtriumHeatMapData(root = 'data/GRIPS22/val', sigma = 2.0, transform = transform)

    batch_size = 1
    data = LightningDataset(
        train_dataset = train,
        val_dataset = val,
        batch_size = batch_size,
        num_workers = 4)

    model = HeatMapRegressor(
        base = ParamGCN,
        base_kwargs = {
            'conv_layer': SAGEConv,
            # 'conv_layer': FeastConv,
            # 'conv_kwargs': {},
            # 'pool_layer': TopKPooling
            # 'pool_kwargs: {'ratio': 0.5},
            'in_channels': 3,
            'hidden_channels': 128,
            'num_layers': 4,
            'out_channels': 8,
            'act': nn.GELU,
            # 'act_kwargs': {'negative_slope': 0.01},
            'norm': GraphNorm(128)},
        loss_func = FocalLoss,
        loss_func_kwargs = {},
        opt = torch.optim.Adam,
        opt_kwargs = {'lr': 0.0005}
        )

    logger = CSVLogger(save_dir = 'results', name = 'testing')

    devices = 4
    num_batches = int(len(train) / batch_size) // devices

    trainer = Trainer(
        accelerator = 'gpu',
        strategy = DDPSpawnPlugin(find_unused_parameters = False),
        devices = devices,
        max_epochs = 2,
        logger = logger,
        log_every_n_steps = num_batches,
        )
    
    trainer.fit(model, data)
    
    # model = HeatMapRegressor.load_from_checkpoint('results/testing/version_6/checkpoints/epoch=19-step=320.ckpt')

    # for i in range(len(val)):
    #     val.visualize_predicted_heat_map(i, model)
    #     val.visualize_predicted_points(i, model)
    #     val.display(i)
