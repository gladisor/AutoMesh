import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import optuna
from optuna.trial import FixedTrial
from main import heatmap_regressor

if __name__ == '__main__':

    study = optuna.load_study(study_name = 'no-name-836a7af2-5fab-4947-aa61-6e7243da6e80', storage = 'sqlite:///big.db')

    datasets = [
        'data/cv_0',
        'data/cv_1',
        'data/cv_2',
        'data/cv_3',
        'data/cv_4']

    for dataset in datasets:
        print(dataset)
        heatmap_regressor(study.best_trial, path = dataset, num_epochs = 200)
