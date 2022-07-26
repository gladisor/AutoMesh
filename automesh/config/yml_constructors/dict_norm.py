## allows us to access the automesh library from outside
import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import yaml
from torch_geometric.nn import GraphNorm, LayerNorm


#tuple for range list for categorical


if __name__ == '__main__':
    norms = {
        'GraphNorm': {
            'obj': GraphNorm
            },
        'LayerNorm':{
            'obj': LayerNorm
            }
    }

    with open('automesh/config/norm.yml', 'w') as f:
        yaml.dump(norms, f)
        
