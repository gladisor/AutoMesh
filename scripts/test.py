## allows us to access the automesh library from outside
import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import yaml
from inspect import signature
from typing import Tuple, Dict

import torch.nn as nn
from torch_geometric.nn import SAGEConv, GATConv, GCNConv
import pandas as pd
from optuna import Trial

def select_params(trial: Trial, path: str) -> Tuple[object, Dict]:
    categorical_option = trial.suggest_categorical([])

    options[categorical_option]

    params = {}

    for name, limits in options[categorical_option]:
        if type(limits) == List:
            params[name] = trial.suggest_categorical(name, limits)
        elif 
        
    return (categorical_option, params)


if __name__ == '__main__':
    # layers = {
    #     'GCNConv': {
    #         'layer': GCNConv,
    #         'params': {
    #             'improved': [True, False],
    #             'add_self_loops': [True, False],
    #             'normalize': [True, False]
    #         }
    #     },
    #     'SAGEConv': {
    #         'layer': SAGEConv,
    #         'params': {
    #             'normalize': [True, False],
    #             'project': [True, False]
    #         }
    #     },
    #     'GATConv': {
    #         'layer': GATConv,
    #         'params': {
    #             'heads': (1, 8),
    #             'concat': False,
    #             'dropout': (0.0, 0.5),
    #             'add_self_loops': [True, False],
    #         }
    #     }
    # }

    # with open('layers.yml', 'w') as f:
    #     yaml.dump(layers, f)