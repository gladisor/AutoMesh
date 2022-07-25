## allows us to access the automesh library from outside
import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import yaml

from automesh.loss import (
    AdaptiveWingLoss, DiceLoss, BCEDiceLoss, 
    JaccardLoss, FocalLoss, TverskyLoss, FocalTverskyLoss)

#tuple for range list for categorical

if __name__ == '__main__':
    loss_func = {
        'AdaptiveWingLoss': {
            'obj': AdaptiveWingLoss,
            'params': {
                'omega': (8.0, 30.0),
                'theta': (0.2, 0.8)
            }
        },
        ####smoothing for both Dice could be added as parameter: default=1e-6
        'DiceLoss': {
            'obj': DiceLoss
        },
        'BCEDiceLoss': {
                'obj': BCEDiceLoss
        },
        'JaccardLoss': {
            'obj': JaccardLoss
        },
        'FocalLoss': {
            'obj': FocalLoss,
            'params': {
                'alpha_f': (0.6, 1.0),
                'gamma_f': (0.0, 5.0)
            }
        },
        'TverskyLoss': {
            'obj': TverskyLoss,
            'params': {
                'alpha_t': (0.0, 1.0)
            }
        },
        'FocalTverskyLoss': {
            'obj': FocalTverskyLoss,
            'params': {
                'gamma_ft': (1.0, 3.0)
            }
        },
    }

    with open('automesh/config/loss_func.yml', 'w') as f:
        yaml.dump(loss_func, f)