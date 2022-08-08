## allows us to access the automesh library from outside
import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import pickle
import glob

import torch_geometric.transforms as T
import optuna
import matplotlib.pyplot as plt
import pandas as pd

from automesh.data.data import LeftAtriumHeatMapData
from automesh.models.heatmap import HeatMapRegressor
from automesh.data.transforms import preprocess_pipeline, rotation_pipeline

def combine(data):
    data = pd.concat(data, axis = 1).mean(axis = 'columns')
    data = data.reset_index(drop = True)
    data = data.to_frame(name = 'data')

    return data

if __name__ == '__main__':

    val_losses = []
    val_nmes = []
    train_losses = []

    for path in glob.glob('results/validation/*'):
        data = pd.read_csv(os.path.join(path, 'metrics.csv'))
        val = data[['epoch', 'val_nme', 'val_loss']].dropna()
        train = data[['epoch', 'train_loss']].dropna()

        val_losses.append(val['val_loss'])
        val_nmes.append(val['val_nme'])
        train_losses.append(train['train_loss'])

    val_losses = combine(val_losses)
    val_nmes = combine(val_nmes)
    train_losses = combine(train_losses)

    fig, ax = plt.subplots(2, 1)
    fig.set_size_inches(10, 8)

    ax[0].plot(val_nmes.index, val_nmes['data'])
    ax[0].set_ylabel('Normalized Mean Error')
    ax[1].plot(train_losses.index, train_losses['data'])
    ax[1].plot(val_losses.index, val_losses['data'])
    plt.show()

#    transform = T.Compose([
#        preprocess_pipeline(), 
#        rotation_pipeline(degrees=50),
#        T.GenerateMeshNormals(),
#        T.PointPairFeatures()
#        ])

#    val = LeftAtriumHeatMapData(root = 'data/GRIPS22/val', sigma = 1.0, transform = transform)

#    model = HeatMapRegressor.load_from_checkpoint('results/edge_features/version_4/checkpoints/epoch=499-step=4000.ckpt')

#    for i in range(len(val)):
#        val.visualize_predicted_heat_map(i, model)
#        val.visualize_predicted_points(i, model)
#        val.display(i)

    # study = pickle.load(open('study.pkl', 'rb'))
    # for trial in study.trials:
    #     intermediate_values = trial.storage.get_trial(trial._trial_id).intermediate_values
    #     print(intermediate_values)

    # fig = optuna.visualization.plot_intermediate_values(study)
    # fig.show()

