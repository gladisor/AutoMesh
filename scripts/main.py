## allows us to access the automesh library from outside
import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import pickle
# import socketserver
# socketserver.TCPServer.allow_reuse_address = True

## third party
import torch
import torch.nn as nn
import torch_geometric.transforms as T
from torch_geometric.data import LightningDataset
from torch_geometric.nn import GraphNorm, SAGEConv, GATConv
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.plugins import DDPSpawnPlugin, environments
from pytorch_lightning.loggers import CSVLogger
from optuna import Trial, create_study, create_trial
from optuna.trial import FixedTrial
from optuna.integration import PyTorchLightningPruningCallback
from optuna.pruners import HyperbandPruner
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
    seed_everything(42)

    transform = T.Compose([
        preprocess_pipeline(), 
        rotation_pipeline(degrees=50),
        ])

    train = LeftAtriumHeatMapData(root = 'data/GRIPS22/train', sigma = 2.0, transform = transform)
    val = LeftAtriumHeatMapData(root = 'data/GRIPS22/val', sigma = 2.0, transform = transform)

    batch_size = 1
    data = LightningDataset(
        train_dataset = train,
        val_dataset = val,
        batch_size = batch_size,
        num_workers = 1,
        persistent_workers = True
	)

    hidden_channels = trial.suggest_int('hidden_channels', 32, 256)
    num_layers = trial.suggest_int('num_layers', 3, 8)
    lr = trial.suggest_float('lr', 0.00001, 0.001)

    params = {
        'base': ParamGCN,
        'base_kwargs': {
            'conv_layer': SAGEConv,
            'conv_kwargs': {},
            'in_channels': 3,
            'hidden_channels': hidden_channels,
            'num_layers': num_layers,
            'out_channels': 8,
            'act': nn.GELU,
            'act_kwargs': {},
            'norm': GraphNorm(hidden_channels)
        },
        'loss_func': FocalLoss,
        'loss_func_kwargs': {},
        'opt': torch.optim.Adam,
        'opt_kwargs': {'lr': lr}
    }

    model = HeatMapRegressor(
        base = params['base'],
        base_kwargs = params['base_kwargs'],
        loss_func = params['loss_func'],
        loss_func_kwargs = params['loss_func_kwargs'],
        opt = params['opt'],
        opt_kwargs = params['opt_kwargs'])

    logger = CSVLogger(save_dir = 'results', name = 'optuna')
    devices = 1

    ddp_spawn_plugin = DDPSpawnPlugin(find_unused_parameters = False) 

    trainer = Trainer(
        accelerator = 'gpu',
        strategy = ddp_spawn_plugin,
        devices = devices,
        max_epochs = 50,
        logger = logger,
        # callbacks = [PyTorchLightningPruningCallback(trial, monitor='val_nme')]
        )

    trainer.fit(model, data)

    path = os.path.join(logger.save_dir, logger.name, 'version_' + str(logger.version - 1), 'metrics.csv')
    history = pd.read_csv(path)
    return history['val_nme'].min()

if __name__ == '__main__':
    study = create_study(
        direction = 'minimize')

    study.optimize(heatmap_regressor, n_trials = 100)

    with open('study.pkl', 'wb') as f:
        pickle.dump(study, f)

    # trial = FixedTrial({'hidden_channels': 64, 'num_layers': 3, 'lr': 0.0001})
    # heatmap_regressor(trial)
