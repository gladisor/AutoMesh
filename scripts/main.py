## allows us to access the automesh library from outside
import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import sqlite3
import warnings

## third party
import torch
import torch.nn as nn
import torch_geometric.transforms as T
from torch_geometric.data import LightningDataset
from torch_geometric.nn import GraphNorm, SAGEConv
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.plugins import DDPSpawnPlugin
from pytorch_lightning.loggers import CSVLogger

from pytorch_lightning.utilities.warnings import PossibleUserWarning, LightningDeprecationWarning
warnings.filterwarnings('ignore', category = PossibleUserWarning)
warnings.filterwarnings('ignore', category = LightningDeprecationWarning)

from optuna import Trial, create_study, samplers, pruners
from optuna.exceptions import TrialPruned

## local source
from automesh.models.architectures import ParamGCN
from automesh.data.data import LeftAtriumHeatMapData
from automesh.models.heatmap import HeatMapRegressor
from automesh.loss import FocalLoss
from automesh.data.transforms import preprocess_pipeline, rotation_pipeline
from automesh.callbacks import OptimalMetric, AutoMeshPruning
from automesh.config.param_selector import ParamSelector


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
        num_workers = 4)

    param_selector = ParamSelector(trial)
    
    
    basic_params=param_selector.get_basic_params('basic')
    conv_layer, conv_layer_kwargs = param_selector.select_params('conv_layer')
    act, act_kwargs = param_selector.select_params('act')
    norm, norm_kwargs = param_selector.select_params('norm')
    loss_func, loss_func_kwargs = param_selector.select_params('loss_func')
    opt, opt_kwargs = param_selector.select_params('opt')

    params = {
        'base': ParamGCN,
        'base_kwargs': {
            'conv_layer': conv_layer,
            'conv_layer_kwargs': conv_layer_kwargs,
            'in_channels': basic_params['in_channels'],
            'hidden_channels': basic_params['hidden_channels'],
            'num_layers': basic_params['num_layers'],
            'out_channels': basic_params['out_channels'],
            'act': act,
            'act_kwargs': act_kwargs,
            'norm': norm(basic_params['hidden_channels'])
        },
        'loss_func': loss_func,
        'loss_func_kwargs': loss_func_kwargs,
        'opt': opt,
        'opt_kwargs': {'lr' : basic_params['lr'], **opt_kwargs}
    }
    
    print('Params', params)
    model = HeatMapRegressor(
        base = params['base'],
        base_kwargs = params['base_kwargs'],
        loss_func = params['loss_func'],
        loss_func_kwargs = params['loss_func_kwargs'],
        opt = params['opt'],
        opt_kwargs = params['opt_kwargs'])

    logger = CSVLogger(save_dir = 'results', name = 'database')
    tracker = OptimalMetric('minimize', 'val_nme')
    pruner = AutoMeshPruning(trial, 'val_nme')

    trainer = Trainer(
        num_sanity_val_steps=0,
        accelerator = 'gpu',
        strategy = DDPSpawnPlugin(find_unused_parameters = False),
        devices = 4,
        max_epochs = 150,
        logger = logger,
        callbacks = [
            tracker, 
            pruner
        ])

    trainer.fit(model, data)

    if trainer.callback_metrics['pruned']:
        raise TrialPruned()

    return trainer.callback_metrics[tracker.name]

if __name__ == '__main__':

    db_name = 'database.db'
    db = sqlite3.connect(db_name)

    study = create_study(
        #study_name= 'test_4',
        direction = 'minimize',
        # sampler = samplers.RandomSampler(),
        pruner = pruners.HyperbandPruner(),
        storage = f'sqlite:///{db_name}')

    study.optimize(heatmap_regressor, n_trials = 500)
