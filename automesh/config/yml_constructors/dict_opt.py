## allows us to access the automesh library from outside
import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../')))
import yaml
from torch.optim import Adam, SGD, Adagrad


#tuple for range list for categorical

if __name__ == '__main__':
    opt = {
        'Adam': {
            'obj': Adam,
            'params': {
                'weight_decay': (0.0, 0.1),
                'lr':'basic'
            }
        },
        # 'SGD': {
        #     'obj': SGD,
        #     'params': {
        #         'momentum': (0.0, 1.0), 
        #         'weight_decay': (0.0, 0.1),
                # 'lr':'basic'
        #     }
        # },
        # 'Adagrad': {
        #     'obj': Adagrad,
        #     'params': {
        #         'lr_decay': (0.0, 0.7), 
        #         'weight_decay': (0.0, 0.1),
                # 'lr':'basic'
        #     }
        # }
    }

    with open('automesh/config/opt.yml', 'w') as f:
        yaml.dump(opt, f)