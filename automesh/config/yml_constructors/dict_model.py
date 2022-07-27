## allows us to access the automesh library from outside
import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../')))
import yaml
from torch.optim import Adam, SGD, Adagrad
from automesh.models.architectures import ParamGCN, GraphUNet


#tuple for range list for categorical

if __name__ == '__main__':
    model = {
        'ParamGCN': {
            'obj': ParamGCN,
            'params': {
                'in_channels': 'basic',
                'out_channels': 'basic',
                'hidden_channels': 'basic',
                'num_layers': 'basic',
                'lr':'basic',
                'conv_layer': 'not_basic',
                'norm': 'not_basic',
                'act':'not_basic'
            }
    },
       
        'GraphUNet': {
            'obj': GraphUNet,
            'params': {
                'in_channels': 'basic',
                'out_channels': 'basic',
                'hidden_channels': 'basic',
                'depth':(3,6),
                'pool_ratios': (0.5, 0.9),
                'lr':'basic',
                'norm': 'not_basic',
                'act':'not_basic'
            }
    }
}
    
    with open('automesh/config/model.yml', 'w') as f:
        yaml.dump(model, f)