import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import yaml
from torch.nn import ReLU, GELU, ELU, Sigmoid, LeakyReLU, Tanh

if __name__ == '__main__':
    act = {
        'ReLU':{
            'obj' : ReLU
            },
        'GELU':{
            'obj': GELU
            },
        'ELU':{
            'obj' : ELU
            },
        'Sigmoid':{
            'obj' : Sigmoid
            },
        'LeakyReLU':{
            'obj' : LeakyReLU,
            'params':{
                'negative_slope':(0.01,0.1)
                }
            },
        'Tanh':{
            'obj' : Tanh
            },
    }
        
    with open('config/act.yml', 'w') as f:
        yaml.dump(act, f)