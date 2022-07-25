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
from pytorch_lightning.callbacks import Callback, LambdaCallback

from pytorch_lightning.utilities.warnings import PossibleUserWarning, LightningDeprecationWarning
warnings.filterwarnings('ignore', category = PossibleUserWarning)
warnings.filterwarnings('ignore', category = LightningDeprecationWarning)

from optuna import Trial, create_study, samplers, pruners
from optuna.exceptions import TrialPruned
from optuna.trial import FixedTrial
import pandas as pd

## local source
from automesh.models.architectures import ParamGCN
from automesh.data.data import LeftAtriumHeatMapData
from automesh.models.heatmap import HeatMapRegressor
from automesh.loss import FocalLoss
from automesh.data.transforms import preprocess_pipeline, rotation_pipeline

class OptimalMetric(Callback):
    def __init__(self, direction: str, monitor: str):
        super().__init__()
        assert direction == 'maximize' or direction == 'minimize'

        self.direction = direction
        self.monitor = monitor
        self.name = self.direction + '_' + self.monitor

    def on_validation_end(self, trainer: Trainer, _: HeatMapRegressor):
        if not trainer.is_global_zero:
            return

        ## grab current value from rank 0 trainer
        current_value = trainer.callback_metrics.get(self.monitor).item()

        ## best value has not been set yet
        if self.name not in trainer.callback_metrics:
            trainer.callback_metrics[self.name] = current_value
        else:

            ## get best value
            best_value = trainer.callback_metrics[self.name]
            maximum_optimal = self.direction == 'maximize' and current_value > best_value
            minimum_optimal = self.direction == 'minimize' and current_value < best_value

            ## update previous best value if better
            if maximum_optimal or minimum_optimal:
                trainer.callback_metrics[self.name] = current_value

class AutoMeshPruning(Callback):
    def __init__(self, trial: Trial, metric: str):
        super().__init__()

        self.trial = trial
        self.metric = metric
        self.pruned = False

    def on_validation_end(self, trainer: Trainer, pl_module: HeatMapRegressor):
        score = trainer.callback_metrics.get(self.metric)
        epoch = pl_module.current_epoch

        should_stop = False
        if trainer.is_global_zero:
            self.trial.report(score, epoch)
            should_stop = self.trial.should_prune()
            self.pruned = should_stop
            trainer.callback_metrics['pruned'] = should_stop

        trainer.should_stop = trainer.training_type_plugin.broadcast(should_stop)
    
    # def on_fit_end(self, trainer: Trainer, pl_module: HeatMapRegressor):
    #     if self.pruned:
    #         raise TrialPruned()

class AlwaysPrune(pruners.BasePruner):
    def __init__(self) -> None:
        super().__init__()

    def prune(self, study, trial) -> bool:
        return True

activations = {
    'nn.GELU': nn.GELU,
    'nn.ELU': nn.ELU,
    'nn.ReLU': nn.ReLU,
    'nn.LeakyReLU': nn.LeakyReLU
}

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
    num_layers = trial.suggest_int('num_layers', 2, 4)
    act = trial.suggest_categorical('act', list(activations.keys()))
    act = activations[act]
    lr = 0.0005

    params = {
        'base': ParamGCN,
        'base_kwargs': {
            'conv_layer': SAGEConv,
            'conv_kwargs': {},
            'in_channels': 3,
            'hidden_channels': hidden_channels,
            'num_layers': num_layers,
            'out_channels': 8,
            'act': act,
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

    logger = CSVLogger(save_dir = 'results', name = 'database')
    tracker = OptimalMetric('minimize', 'val_nme')
    pruner = AutoMeshPruning(trial, 'val_nme')

    trainer = Trainer(
        num_sanity_val_steps=0,
        accelerator = 'auto',
        strategy = DDPSpawnPlugin(find_unused_parameters = False),
        devices = 4,
        max_epochs = 10,
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
        direction = 'minimize',
        sampler = samplers.RandomSampler(),
        pruner = pruners.ThresholdPruner(upper = 50.0),
        storage = f'sqlite:///{db_name}')

    study.optimize(heatmap_regressor, n_trials = 30)

    # trial = FixedTrial({'hidden_channels': 15, 'num_layers': 2, 'act': 'nn.LeakyReLU'})
    # heatmap_regressor(trial)