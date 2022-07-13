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
from optuna import Trial, create_study, create_trial
import pandas as pd

## local source
from automesh.models.architectures import ParamGCN
from automesh.data.data import LeftAtriumHeatMapData
from automesh.models.heatmap import HeatMapRegressor
from automesh.loss import (
    AdaptiveWingLoss, DiceLoss, BCEDiceLoss, 
    JaccardLoss, FocalLoss, TverskyLoss, FocalTverskyLoss)
from automesh.data.transforms import preprocess_pipeline, rotation_pipeline

def heatmap_regressor(trial: Trial):
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
        num_workers = 4,
        persistent_workers = True)

    hidden_channels = trial.suggest_int('hidden_channels', 128, 1024)
    num_layers = trial.suggest_int('num_layers', 2, 10)
    lr = trial.suggest_float('lr', 0.0001, 0.001)

    model = HeatMapRegressor(
        base = ParamGCN,
        base_kwargs = {
            'conv_layer': SAGEConv,
            'conv_kwargs': {},
            'in_channels': 3,
            'hidden_channels': hidden_channels,
            'num_layers': num_layers,
            'out_channels': 8,
            'act': nn.GELU,
            # 'act_kwargs': {'negative_slope': 0.01},
            'norm': GraphNorm(hidden_channels)},
        loss_func = FocalLoss,
        loss_func_kwargs = {},
        opt = torch.optim.Adam,
        opt_kwargs = {'lr': lr}
        )

    logger = CSVLogger(save_dir = 'results', name = 'testing')

    devices = 2
    num_batches = int(len(train) / batch_size) // devices

    trainer = Trainer(
        accelerator = 'gpu',
        strategy = DDPSpawnPlugin(find_unused_parameters = False),
        devices = devices,
        max_epochs = 150,
        logger = logger,
        log_every_n_steps = num_batches,
        )
    
    trainer.fit(model, data)

    path = os.path.join(logger.save_dir, logger.name, 'version_' + str(logger.version - 1), 'metrics.csv')
    history = pd.read_csv(path)
    return history['nme'].min()

if __name__ == '__main__':
    study = create_study()
    study.optimize(heatmap_regressor, n_trials = 100, direction = 'minimize')