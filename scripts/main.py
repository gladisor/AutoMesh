## allows us to access the automesh library from outside
import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import sqlite3
import warnings
import pprint
## third party
import torch
import torch.nn as nn
import torch_geometric.transforms as T
from torch_geometric.data import LightningDataset
from torch_geometric.nn import GraphNorm, SAGEConv
#from torch_geometric.transforms.add_positional_encoding import AddRandomWalkPE
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
from automesh.models.architectures import ParamGCN
from automesh.data.data import LeftAtriumHeatMapData
from automesh.models.heatmap import HeatMapRegressor
from automesh.loss import FocalLoss, JaccardLoss
from automesh.data.transforms import preprocess_pipeline, rotation_pipeline, AutoMeshVirtualNode
from automesh.callbacks import OptimalMetric, AutoMeshPruning
from automesh.config.param_selector import Selector


from torch.nn import ReLU
from torch_geometric.nn import GATConv
from torch.optim import Adam



def heatmap_regressor(trial: Trial):
    seed_everything(42)
    
    transform = T.Compose([
        preprocess_pipeline(),
        rotation_pipeline(degrees=50),
        T.GenerateMeshNormals(),
        T.PointPairFeatures(),
        AutoMeshVirtualNode()
        ])

    train = LeftAtriumHeatMapData(root = 'data/GRIPS22/trainone', sigma = 2.0, transform = transform, )
    val = LeftAtriumHeatMapData(root = 'data/GRIPS22/valone', sigma = 2.0, transform = transform)
    

    batch_size = 1
    data = LightningDataset(
        train_dataset = train,
        val_dataset = val,
        batch_size = batch_size,
        num_workers = 4)


    selector = Selector(trial, ['model', 'loss_func', 'opt'])
    pprint.pprint(selector.params())
    params = selector.params()
    
    if 'norm' in params['model_kwargs'].keys():
        params['model_kwargs']['norm'] = params['model_kwargs']['norm'](params['model_kwargs']['hidden_channels'])
        
    model = HeatMapRegressor(**selector.params())
    # print(model)
    
    
    
    logger = CSVLogger(save_dir = 'results', name = 'database')
    tracker = OptimalMetric('minimize', 'val_nme')
    pruner = AutoMeshPruning(trial, 'val_nme')

    trainer = Trainer(
        num_sanity_val_steps=0,
        accelerator = 'gpu',
        strategy = DDPSpawnPlugin(find_unused_parameters = False),
        devices = 4,
        max_epochs = 100,
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
    
    # trial =FixedTrial({'act': 'ReLU',
    #                     'add_self_loops': False,
    #                     'concat': False,
    #                     'conv_layer': 'GATConv',
    #                     'dropout': 0.0012250542474682713,
    #                     'heads': 2,
    #                     'hidden_channels': 170,
    #                     'loss_func': 'JaccardLoss',
    #                     'lr': 0.00031583021936346486,
    #                     'norm': 'GraphNorm',
    #                     'num_layers': 8,
    #                     'opt': 'Adam',
    #                     'weight_decay': 0.0})
    # heatmap_regressor(trial)
    
    # study = create_study(direction = 'minimize')

    # study.optimize(heatmap_regressor, n_trials = 3)


