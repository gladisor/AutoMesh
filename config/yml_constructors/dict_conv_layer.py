## allows us to access the automesh library from outside
import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import yaml
from torch_geometric.nn import SAGEConv, GATConv, GCNConv



#tuple for range list for categorical
if __name__ == '__main__':

    conv_layer = {
        'GCNConv': {
            'obj': GCNConv,
            'params': {
                'improved': [True, False],
                'add_self_loops': [True, False],
                'normalize': [True, False]
            }
        },
        'SAGEConv': {
            'obj': SAGEConv,
            'params': {
                'normalize': [True, False]
            }
        },
        'GATConv': {
            'obj': GATConv,
            'params': {
                'heads': (1, 8),
                'concat': [False],
                'dropout': (0.0, 0.5),
                'add_self_loops': [True, False],
            }
        }
    }

    with open('config/conv_layer.yml', 'w') as f:
        yaml.dump(conv_layer, f)