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
from torch_geometric.nn import GraphNorm, SAGEConv, GATConv, FeaStConv, GCNConv
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.plugins import DDPSpawnPlugin, environments
from pytorch_lightning.loggers import CSVLogger
from pytorch_lightning.callbacks import Callback
from optuna import Trial, create_study, create_trial
from optuna.trial import FixedTrial
import pandas as pd

## local source
from automesh.models.architectures import ParamGCN, ParamGraphUNet
from automesh.data.data import LeftAtriumHeatMapData
from automesh.models.heatmap import HeatMapRegressor
from automesh.loss import (
    AdaptiveWingLoss, DiceLoss, BCEDiceLoss, 
    JaccardLoss, FocalLoss, TverskyLoss, FocalTverskyLoss)
from automesh.data.transforms import preprocess_pipeline, rotation_pipeline
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
        num_workers = 10,
        #pin_memory=False,
        #persistent_workers = False,
	)
    

    print("before base")
    param_selector = ParamSelector(trial)
    
    # hidden_channels = trial.suggest_int('hidden_channels', 32, 256)
    # num_layers = trial.suggest_int('num_layers', 3, 5)
    # lr = trial.suggest_float('lr', 0.00001, 0.001)
    
    
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
        'opt_kwargs': {'lr' : basic_params['lr']}
    }
    print('Params', params)
    model = HeatMapRegressor(
        base = params['base'],
        base_kwargs = params['base_kwargs'],
        loss_func = params['loss_func'],
        loss_func_kwargs = params['loss_func_kwargs'],
        opt = params['opt'],
        opt_kwargs = params['opt_kwargs'])

    logger = CSVLogger(save_dir = 'results', name = 'optuna')


    devices = -1



    ddp_spawn_plugin = DDPSpawnPlugin(
#        num_nodes = num_nodes,
#        cluster_environment = environments.SLURMEnvironment(),
        find_unused_parameters = False) 

    trainer = Trainer(
#       num_nodes = num_nodes,
        accelerator = 'gpu',
        strategy = ddp_spawn_plugin,
        devices = devices,


        max_epochs = 150,

        logger = logger,
        )

    trainer.fit(model, data)

    path = os.path.join(logger.save_dir, logger.name, 'version_' + str(logger.version - 1), 'metrics.csv')
    history = pd.read_csv(path)
    return history['val_nme'].min()

if __name__ == '__main__':
    # global startingtime
    # startingtime=0.0
    
    study = create_study()
    study.optimize(heatmap_regressor, n_trials = 100)

    with open('study.pkl', 'wb') as f:
        pickle.dump(study, f)

   
    # trial = FixedTrial({'hidden_channels': 832, 'num_layers': 8, 'lr': 0.0007})
    # heatmap_regressor(trial)

