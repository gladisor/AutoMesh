from typing import Callable, Union

import torch
import torch.nn as nn
from torch.optim import Optimizer
from torch_geometric.data import Data, Batch
from pytorch_lightning import LightningModule

from automesh.models.architectures import BaseArchitecture

class HeatMapRegressor(LightningModule):
    def __init__(
            self,
            base: BaseArchitecture,
            optimizer: Optimizer,
            lr: float,
            loss_func: Callable,
            **kwargs) -> None:

        super().__init__()
        self.base = base(**kwargs)
        self.optimizer = optimizer
        self.lr = lr
        self.loss_func = loss_func

    @staticmethod
    def predict_points(heatmap: torch.tensor, points: torch.tensor) -> torch.tensor:
        idx = heatmap.argmax(dim = 0) # get max value index for each landmark
        return points[idx, :] # extract the coordinates
    
    @staticmethod
    def normalized_mean_error(pred_points: torch.tensor, true_points: torch.tensor) -> torch.tensor:
        return (true_points - pred_points).norm(dim = 1).mean()
    
    @staticmethod
    def evaluate_heatmap(heatmap: torch.tensor, data: Data) -> torch.tensor:
        pred_points = HeatMapRegressor.predict_points(heatmap, data.x)
        true_points = HeatMapRegressor.predict_points(data.y, data.x)
        return HeatMapRegressor.normalized_mean_error(pred_points, true_points)
    
    def forward(self, x: Union[Data, Batch]) -> torch.tensor:
        if x.edge_attr != None:
            return self.base(x.pos, x.edge_index, x.edge_attr)
        else:
            return self.base(x.pos, x.edge_index)
    
    def configure_optimizers(self):
        opt = self.optimizer(self.base.parameters(), self.lr)
        return opt

    def landmark_loss(self, y_hat, y) -> torch.tensor:
        ## compute landmark loss on each channel
        loss = 0.0
        for c in range(y_hat.shape[1]):
            loss += self.loss_func(y_hat[:, c], y[:, c])

        return loss

    def training_step(self, batch: Batch, batch_idx) -> torch.tensor:
        ## compute landmark loss on each channel
        return self.landmark_loss(self(batch), batch.y)

    def validation_step(self, batch, batch_idx):

        distance = 0.0
        for i in range(batch.num_graphs):
            data = batch.get_example(i)
            distance += HeatMapRegressor.evaluate_heatmap(self(data), data)

        distance /= batch.num_graphs

        val_loss = self.landmark_loss(self(batch), batch.y)
        print(f'{batch_idx} -> Val loss: {val_loss}, Mean distance: {distance}')