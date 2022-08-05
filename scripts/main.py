## allows us to access the automesh library from outside
import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import sqlite3
import warnings
from pprint import pprint

## third party
import torch
import torch.nn as nn

torch.multiprocessing.set_sharing_strategy('file_system')

import torch_geometric.transforms as T
from torch_geometric.data import LightningDataset
from torch_geometric.nn import GraphNorm, GraphSAGE, GraphUNet
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.plugins import DDPSpawnPlugin
from pytorch_lightning.loggers import CSVLogger

from pytorch_lightning.utilities.warnings import PossibleUserWarning, LightningDeprecationWarning
warnings.filterwarnings('ignore', category = PossibleUserWarning)
warnings.filterwarnings('ignore', category = LightningDeprecationWarning)

from optuna import Trial, create_study, samplers, pruners
from optuna.exceptions import TrialPruned
from optuna.trial import FixedTrial

## local source
from automesh.data.data import LeftAtriumHeatMapData
from automesh.models.heatmap import HeatMapRegressor
from automesh.data.transforms import preprocess_pipeline, rotation_pipeline
from automesh.callbacks import OptimalMetric, AutoMeshPruning
from automesh.config.param_selector import Selector
import automesh.models.architectures

def heatmap_regressor(trial: Trial, num_epochs = 100):
    seed_everything(42)

    transform = T.Compose([
        preprocess_pipeline(), 
        rotation_pipeline(degrees=25),
	T.GenerateMeshNormals(),
        T.PointPairFeatures(),
        ])

    sigma = trial.suggest_float('sigma', 0.5, 3.0)
    train = LeftAtriumHeatMapData(root = 'data/GRIPS22/train', sigma = sigma, transform = transform)
    val = LeftAtriumHeatMapData(root = 'data/GRIPS22/val', sigma = sigma, transform = transform)

    batch_size = trial.suggest_int('batch_size', 1, 7)
    data = LightningDataset(
        train_dataset = train,
        val_dataset = val,
        batch_size = batch_size,
        num_workers = 4)

    selector = Selector(trial, ['model', 'loss_func', 'opt'])
    params = selector.params()

    pprint(selector.params())

    if 'norm' in params['model_kwargs'].keys():
        params['model_kwargs']['norm'] = params['model_kwargs']['norm'](params['model_kwargs']['hidden_channels'])
        params['model_kwargs'].pop('norm_kwargs')

    model = HeatMapRegressor(**selector.params())

    ## callbacks
    tracker = OptimalMetric('minimize', 'val_nme')
    pruner = AutoMeshPruning(trial, 'val_nme')

    logger = CSVLogger(save_dir = 'results', name = 'best')

    trainer = Trainer(
        num_sanity_val_steps=0,
        accelerator = 'auto',
        strategy = DDPSpawnPlugin(find_unused_parameters = False),
        devices = 4,
        max_epochs = num_epochs,
        logger = logger,
        callbacks = [
            tracker,
            pruner
        ])

    try:
        trainer.fit(model, data)
    except Exception:
        pass

    if trainer.callback_metrics.get('pruned', False):
        raise TrialPruned()

    return trainer.callback_metrics[tracker.name]

if __name__ == '__main__':

    db_name = 'big.db'
    db = sqlite3.connect(db_name)

    study = create_study(
        direction = 'minimize',
        sampler = samplers.TPESampler(),
        storage = f'sqlite:///{db_name}')

    study.optimize(heatmap_regressor, n_trials = 500)

    # ## For evaluating a fixed trial
    # trial = FixedTrial({
    #     ## model
    #     'model': 'ParamGCN',
    #     'conv_layer': 'GATv2Conv',
    #     'act': 'LeakyReLU',
    #     'negative_slope': 0.04876889,
    #     'dropout': 0.1,
    #     'in_channels': 3,
    #     'hidden_channels': 128,
    #     'num_layers': 2,
    #     'out_channels': 8,
    #     'norm': 'GraphNorm',
    #     'heads': 1,
    #     'add_self_loops': True,

    #     'loss_func': 'FocalTverskyLoss',
    #     'alpha_t': 0.03401540046,
    #     'gamma_ft': 1.5437116,

    #     ## optimizer
    #     'opt': 'Adam',
    #     'lr': 0.000751094,
    #     'weight_decay': 4e-05
    # })

    # heatmap_regressor(trial)
