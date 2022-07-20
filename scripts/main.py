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
from pytorch_lightning.callbacks import Callback
from optuna import Trial, TrialPruned, create_study
from optuna.trial import FixedTrial
from optuna.integration import PyTorchLightningPruningCallback

from optuna.samplers import RandomSampler
from optuna.pruners import HyperbandPruner, MedianPruner, PercentilePruner, BasePruner
import pandas as pd

## local source
from automesh.models.architectures import ParamGCN
from automesh.data.data import LeftAtriumHeatMapData
from automesh.models.heatmap import HeatMapRegressor
from automesh.loss import (
    AdaptiveWingLoss, DiceLoss, BCEDiceLoss, 
    JaccardLoss, FocalLoss, TverskyLoss, FocalTverskyLoss)
from automesh.data.transforms import preprocess_pipeline, rotation_pipeline

class AutoMeshPruning(Callback):
    def __init__(self, trial: Trial, monitor: str) -> None:
        super().__init__()
        self.trial = trial
        self.monitor = monitor

    def on_validation_epoch_end(self, trainer: Trainer, pl_module: HeatMapRegressor) -> None:
        epoch = pl_module.current_epoch
        current_score = trainer.callback_metrics.get(self.monitor)
        self.trial.report(current_score, step = epoch)

        trial = self.trial.storage.get_trial(self.trial._trial_id)

        should_prune = self.trial.study.pruner.prune(self.trial.study, trial)

        print(f'Should Prune? : {should_prune}')

        should_stop = should_prune or self.trial.study.pruner.prune(self.trial.study, self.trial)
        should_stop = trainer.strategy.reduce_boolean_decision(should_stop)
        trainer.should_stop = trainer.should_stop or should_stop

        if trainer.should_stop:
            message = "Trial was pruned at epoch {}.".format(epoch)
            print(message)

class AlwaysPrune(BasePruner):
    def __init__(self) -> None:
        super().__init__()

    def prune(self, study, trial: Trial) -> bool:
        return True

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
        num_workers = 3)

    hidden_channels = trial.suggest_int('hidden_channels', 10, 20)
    num_layers = trial.suggest_int('num_layers', 2, 3)
    # lr = trial.suggest_float('lr', 0.00001, 0.001)
    lr = 0.00001

    params = {
        'base': ParamGCN,
        'base_kwargs': {
            'conv_layer': SAGEConv,
            'conv_kwargs': {},
            'in_channels': 3,
            'hidden_channels': hidden_channels,
            'num_layers': num_layers,
            'out_channels': 8,
            # 'act': trial.suggest_categorical('act', [nn.GELU, nn.ELU, nn.ReLU, nn.LeakyReLU]),
            'act': nn.ReLU,
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

    trainer = Trainer(
        num_sanity_val_steps=0,
        accelerator = 'cpu',
        strategy = DDPSpawnPlugin(find_unused_parameters = False),
        devices = 3,
        max_epochs = 5,
        logger = logger,
        callbacks = [AutoMeshPruning(trial, monitor='val_nme')]
        )

    trainer.fit(model, data)

    path = os.path.join(logger.save_dir, logger.name, 'version_' + str(logger.version - 1), 'metrics.csv')
    history = pd.read_csv(path)

    return history['val_nme'].min()

if __name__ == '__main__':
    study = create_study(
        direction = 'minimize',
        sampler = RandomSampler(),
        # pruner = PercentilePruner(percentile = 1.0, n_startup_trials = 1)
        pruner = AlwaysPrune()
        )

    study.optimize(heatmap_regressor, n_trials = 10)

    with open('study.pkl', 'wb') as f:
        pickle.dump(study, f)
    
    # trial = FixedTrial({'hidden_channels': 64, 'num_layers': 3, 'lr': 0.0001, 'act': nn.ReLU})
    # heatmap_regressor(trial)

    # print(trial.intermediate_values)
    