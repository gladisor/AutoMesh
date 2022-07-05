## allows us to access the automesh library from outside
import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

## third party
import torch_geometric.transforms as T
from torch_geometric.loader import DataLoader
import torch

## local source
from automesh.data.data import LeftAtriumHeatMapData
from automesh.models.heatmap import HeatMapRegressor
from automesh.utils.data import split, preprocess_pipeline, augmentation_pipeline

if __name__ == '__main__':
    data = LeftAtriumHeatMapData(
        root = 'data/GRIPS22',
        sigma = 2.0,
        triangles = 5000,
        transform = T.Compose([
            preprocess_pipeline(),
            augmentation_pipeline()
            ]))

    train, val = split(data, 0.9)
    loader = DataLoader(train, batch_size = 4, shuffle = True, drop_last = True)
    model = HeatMapRegressor(256, 4, 8, 0.1, torch.relu, 0.001)

    # model.train()

    # for epoch in range(10):
    #     for batch in loader:
    #         model.opt.zero_grad()

    #         y_hat = model(batch)

    #         loss = model.calculate_loss(y_hat, batch.y)

    #         print(f'Train Loss: {loss}')

    #         loss.backward()
    #         model.opt.step()

    # model.eval()
    # val.dataset.visualize_predicted_heat_map(3, model)
    # val.dataset.visualize_predicted_heat_map(10, model)
    # val.dataset.visualize_predicted_heat_map(5, model)