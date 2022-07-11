from typing import Callable, Union, Dict, Any

import numpy as np
import torch
import torch.nn as nn
from torch.optim import Optimizer
from torch_geometric.data import Data, Batch
from pytorch_lightning import LightningModule

class HeatMapRegressor(LightningModule):
    """
    Predicts landmarks on a mesh by generating heatmap distributions per channel.
    """
    def __init__(
            self,
            base: nn.Module,
            opt: Optimizer,
            opt_kwargs: Dict[str, Any],
            loss_func: nn.Module,
            loss_func_kwargs: Dict[str, Any],
            **kwargs) -> None:
        super().__init__()

        ## constructing graph neural network
        self.base = base(**kwargs)
        self.opt = opt
        self.opt_kwargs = opt_kwargs
        self.loss_func = loss_func(**loss_func_kwargs)

        self.save_hyperparameters()

    @staticmethod
    def predict_points(heatmap: torch.tensor, points: torch.tensor) -> torch.tensor:
        idx = heatmap.argmax(dim = 0) # get max value index for each landmark
        return points[idx, :] # extract the coordinates

    @staticmethod
    def normalized_mean_error(pred_points: torch.tensor, true_points: torch.tensor) -> torch.tensor:
        ## mean euclidan distance between points
        return (true_points - pred_points).norm(dim = 1).mean()
    
    @staticmethod
    def evaluate_heatmap(heatmap: torch.tensor, data: Data) -> torch.tensor:
        ## get points with highest predicted probability
        pred_points = HeatMapRegressor.predict_points(heatmap, data.x)
        ## extract true highest probability points from ground truth heatmap
        true_points = HeatMapRegressor.predict_points(data.y, data.x)
        ## computing average distance between points
        return HeatMapRegressor.normalized_mean_error(pred_points, true_points)

    def forward(self, x: Union[Data, Batch]) -> torch.tensor:
        if x.edge_attr != None:
            return self.base(x.pos, x.edge_index, x.edge_attr)
        else:
            return self.base(x.pos, x.edge_index)
    
    def configure_optimizers(self):
        return self.opt(self.base.parameters(), **self.opt_kwargs)

    def landmark_loss(self, y_hat, y) -> torch.tensor:
        ## compute landmark loss on each channel
        loss = 0.0
        for c in range(y_hat.shape[1]):
            loss += self.loss_func(y_hat[:, c], y[:, c])

        return loss

    def training_step(self, batch: Batch, batch_idx) -> torch.tensor:
        ## compute landmark loss on each channel
        loss = self.landmark_loss(self(batch), batch.y)

        self.log(
            'train_loss', loss, 
            batch_size = batch.num_graphs, 
            )
        return loss

    def validation_step(self, batch, batch_idx):
        ## we dont want to differentiate
        with torch.no_grad():
            distance = 0.0

            ## calculate evaluation metric independently on each graph
            for i in range(batch.num_graphs):
                data = batch.get_example(i)
                distance += HeatMapRegressor.evaluate_heatmap(self(data), data)

            ## average over number of graphs
            distance /= batch.num_graphs

            ## compute loss on validation batch as well
            val_loss = self.landmark_loss(self(batch), batch.y)
        
        self.log(
            'val_performance', {'val_loss': val_loss, 'distance': distance}, 
            batch_size = batch.num_graphs)