## allows us to access the automesh library from outside
import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import pickle

import torch_geometric.transforms as T
import optuna
import matplotlib.pyplot as plt
import pandas as pd

from automesh.data.data import LeftAtriumHeatMapData
from automesh.models.heatmap import HeatMapRegressor
from automesh.data.transforms import preprocess_pipeline, rotation_pipeline


if __name__ == '__main__':

    data = pd.read_csv('results/data/version_5/metrics.csv')
    val = data[['epoch', 'val_nme', 'val_loss']].dropna()
    train = data[['epoch', 'train_loss']].dropna()

    fig, ax = plt.subplots(2, 1)
    fig.set_size_inches(10, 8)

    ax[0].plot(val['epoch'], val['val_nme'])
    ax[1].plot(train['epoch'], train['train_loss'])
    ax[1].plot(val['epoch'], val['val_loss'])
    plt.show()

    # transform = T.Compose([
    #     preprocess_pipeline(), 
    #     rotation_pipeline(degrees=50),
    #     T.GenerateMeshNormals(),
    #     T.PointPairFeatures()
    #     ])

    # val = LeftAtriumHeatMapData(root = 'data/GRIPS22/val', sigma = 2.0, transform = transform)

    # model = HeatMapRegressor.load_from_checkpoint('results/database/version_8/checkpoints/epoch=99-step=2200.ckpt')

    # for i in range(len(val)):
    #     val.visualize_predicted_heat_map(i, model)
    #     val.visualize_predicted_points(i, model)
    #     val.display(i)

    # study = pickle.load(open('study.pkl', 'rb'))
    # for trial in study.trials:
    #     intermediate_values = trial.storage.get_trial(trial._trial_id).intermediate_values
    #     print(intermediate_values)

    # fig = optuna.visualization.plot_intermediate_values(study)
    # fig.show()

