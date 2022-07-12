from typing import Callable, Union, Dict, Any, Optional

import numpy as np
import torch
import torch.nn as nn
from torch.optim import Optimizer
from torch_geometric.data import Data, Batch
from pytorch_lightning import LightningModule
import yaml

from automesh.metric import NormalizedMeanError

class HeatMapRegressor(LightningModule):
    """
    Predicts landmarks on a mesh by generating heatmap distributions per channel.
    """
    def __init__(
            self,
            base: nn.Module,
            base_kwargs: Dict[str, Any],
            opt: Optimizer,
            opt_kwargs: Dict[str, Any],
            loss_func: nn.Module,
            loss_func_kwargs: Optional[Dict[str, Any]] = {}):
        super().__init__()

        ## constructing graph neural network
        self.base = base(**base_kwargs)
        self.opt = opt
        self.opt_kwargs = opt_kwargs
        self.loss_func = loss_func(**loss_func_kwargs)
        self.nme = NormalizedMeanError()

        ## saving state
        self.save_hyperparameters(ignore = ['norm'])

    # @staticmethod
    # def load_from_config(path: str):
    #     with open('config.yml', 'r') as stream:
    #         config = yaml.full_load(stream)

        
    #     return

    @staticmethod
    def predict_points(heatmap: torch.tensor, points: torch.tensor) -> torch.tensor:
        idx = heatmap.argmax(dim = 0) # get max value index for each landmark
        return points[idx, :] # extract the coordinates

    def forward(self, x: Union[Data, Batch]) -> torch.tensor:
        ## use edge attributes in forward pass if they exist
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

        self.log('train_loss', loss, batch_size = batch.num_graphs)
        return loss

    def validation_step(self, batch, batch_idx):
        ## calculate evaluation metric independently on each graph
        for i in range(batch.num_graphs):
            data = batch.get_example(i)
            heatmap = self(data)
            pred_points = HeatMapRegressor.predict_points(heatmap, data.x)
            true_points = HeatMapRegressor.predict_points(data.y, data.x)

            self.nme.update(pred_points, true_points)

        ## compute loss on validation batch as well
        val_loss = self.landmark_loss(self(batch), batch.y)
        
        self.log(
            'nme',
            self.nme,
            batch_size = batch.num_graphs,
            sync_dist = True)

        return {'nme': self.nme, 'val_loss': val_loss}