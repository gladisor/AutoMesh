## allows us to access the automesh library from outside
import os
import sys
import pprint
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from optuna import create_study

from automesh.config.param_selector import Selector
import automesh.loss

def objective(trial):
    
    selector = Selector(trial, ['model', 'loss_func', 'opt'])
    
    pprint.pprint(selector.params())
    
    return 1.0

if __name__ == '__main__':
        study = create_study(direction = 'minimize')

        study.optimize(objective, n_trials = 3)