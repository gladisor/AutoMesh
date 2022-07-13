## allows us to access the automesh library from outside
import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import torch_geometric.transforms as T

from automesh.data.data import LeftAtriumHeatMapData
from automesh.models.heatmap import HeatMapRegressor
from automesh.data.transforms import preprocess_pipeline, rotation_pipeline

if __name__ == '__main__':
    transform = T.Compose([
        preprocess_pipeline(), 
        rotation_pipeline(),
        ])
    val = LeftAtriumHeatMapData(root = 'data/GRIPS22/val', sigma = 2.0, transform = transform)

    model = HeatMapRegressor.load_from_checkpoint('epoch=149-step=4800.ckpt')

    for i in range(len(val)):
        val.visualize_predicted_heat_map(i, model)
        val.visualize_predicted_points(i, model)
        val.display(i)