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
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.plugins import DDPSpawnPlugin
from pytorch_lightning.loggers import CSVLogger, NeptuneLogger, CometLogger, WandbLogger
# from pytorch_lightning.callbacks import ModelCheckpoint
from optuna import Trial, create_study, create_trial
from optuna.trial import FixedTrial
import pandas as pd
import wandb

## local source
from automesh.models.architectures import ParamGCN
from automesh.data.data import LeftAtriumHeatMapData
from automesh.models.heatmap import HeatMapRegressor
from automesh.loss import (
    AdaptiveWingLoss, DiceLoss, BCEDiceLoss, 
    JaccardLoss, FocalLoss, TverskyLoss, FocalTverskyLoss)
from automesh.data.transforms import preprocess_pipeline, rotation_pipeline

# os.environ['COMET_URL_OVERRIDE'] = 'https://www.comet.com/clientlib/'
os.environ['WANDB_API_KEY'] = '9a6992594ce0851dbccb151860b2751420a558a3'
os.environ['WANDB_MODE'] = 'offline'
os.environ['WANDB_ENTITY'] = 'tshah'

wandb.init()

def heatmap_regressor(trial: Trial):
    seed_everything(42, workers = True)

    transform = T.Compose([
        preprocess_pipeline(), 
        rotation_pipeline(degrees=50),
        ])

    train = LeftAtriumHeatMapData(root = 'data/GRIPS22/train', sigma = 2.0, transform = transform)
    val = LeftAtriumHeatMapData(root = 'data/GRIPS22/val', sigma = 2.0, transform = transform)

    batch_size = 2
    data = LightningDataset(
        train_dataset = train,
        val_dataset = val,
        batch_size = batch_size,
        num_workers = 4,
        persistent_workers = True
	)

    hidden_channels = trial.suggest_int('hidden_channels', 128, 1024)
    num_layers = trial.suggest_int('num_layers', 2, 10)
    lr = trial.suggest_float('lr', 0.0001, 0.001)

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

    # logger = CSVLogger(save_dir = 'results', name = 'best_numerical_params')
    wandb_logger = WandbLogger()

#    neptune_logger = NeptuneLogger(
#        api_key = 'eyJhcGlfYWRkcmVzcyI6Imh0dHBzOi8vYXBwLm5lcHR1bmUuYWkiLCJhcGlfdXJsIjoiaHR0cHM6Ly9hcHAubmVwdHVuZS5haSIsImFwaV9rZXkiOiJjOTU0NTAwNi0wMWQyLTRlMDgtOThiZS1kZDE0NjE2YzA2MDMifQ==',
#        project = 'tristan.shah/automesh',
#	mode = 'offline',
#	log_model_checkpoints = False,
#	)


#    cl = CometLogger(
#        save_dir = '.',
#        workspace = 'gladisor',  # Optional
#        project_name = 'automesh',  # Optional
#        api_key = '78RvPfsgNnp3mE3JVYZmKyngw',
#        rest_api_key = '78RvPfsgNnp3mE3JVYZmKyngw',
#        offline = True
#        )

    devices = 3
    num_batches = int(len(train) / batch_size) // devices

    trainer = Trainer(
        num_nodes = 1,
        accelerator = 'gpu',
        strategy = DDPSpawnPlugin(find_unused_parameters = False),
        devices = devices,
        max_epochs = 50,
        logger = wandb_logger,
        log_every_n_steps = num_batches,
	# enable_checkpointing = False
        )

    trainer.fit(model, data)

    path = os.path.join(logger.save_dir, logger.name, 'version_' + str(logger.version - 1), 'metrics.csv')
    # print(neptune_logger.save_dir, neptune_logger.name, neptune_logger.version)
    history = pd.read_csv(path)
    return history['val_nme'].min()

if __name__ == '__main__':
    # study = create_study()
    # study.optimize(heatmap_regressor, n_trials = 100)

    trial = FixedTrial({'hidden_channels': 832, 'num_layers': 8, 'lr': 0.0007})

    heatmap_regressor(trial)
