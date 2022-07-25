## allows us to access the automesh library from outside
import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import yaml


#tuple for range list for categorical

if __name__ == '__main__':
    basic = {
        'hidden_channels': (32,256),
        'num_layers': (3,5),
        'lr': (0.00001,0.001),
        'in_channels': 3, #nr of different features
        'out_channels': 8 #nr of branching points
        }
    with open('automesh/config/basic.yml', 'w') as f:
        yaml.dump(basic, f)
        