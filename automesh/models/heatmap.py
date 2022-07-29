from typing import Callable, Union, Dict, Any, Optional

import numpy as np
import torch
import torch.nn as nn
from torch.optim import Optimizer
from torch_geometric.data import Data, Batch
from pytorch_lightning import LightningModule

from automesh.metric import NormalizedMeanError
from automesh.loss import ChannelWiseLoss

class HeatMapRegressor(LightningModule):
    """
    Predicts landmarks on a mesh by generating heatmap distributions per channel.
    """
    def __init__(
            self,
            model: nn.Module,
            model_kwargs: Dict[str, Any],
            opt: Optimizer,
            opt_kwargs: Dict[str, Any],
            loss_func: nn.Module,
            loss_func_kwargs: Optional[Dict[str, Any]] = {}):
        super().__init__()

        ## constructing graph neural network
        self.model = model(**model_kwargs)
        self.opt = opt
        self.opt_kwargs = opt_kwargs
        self.loss_func = ChannelWiseLoss(loss_func(**loss_func_kwargs))
        self.nme = NormalizedMeanError()

        ## saving state
        self.save_hyperparameters(ignore = ['norm'])

    @staticmethod
    def predict_points(heatmap: torch.tensor, points: torch.tensor) -> torch.tensor:
        idx = heatmap.argmax(dim = 0) # get max value index for each landmark
        return points[idx, :] # extract the coordinates

    def forward(self, x: Union[Data, Batch]) -> torch.tensor:
        ## use edge attributes in forward pass if they exist
        x_pos = x.pos.clone()
        x_edge_index = x.edge_index.clone()

        has_edge_attr = False
        if x.edge_attr != None:
            has_edge_attr = True
            x_edge_attr = x.edge_attr.clone()

        del x

        if has_edge_attr:
            return self.model(x_pos, x_edge_index, x_edge_attr)
        else:
            return self.model(x_pos, x_edge_index)

    def configure_optimizers(self):
        return self.opt(self.model.parameters(), **self.opt_kwargs)

    def landmark_loss(self, y_hat: torch.Tensor, y: torch.Tensor):

        loss = self.loss_func(y_hat, y)
        
        # assert y_hat.shape == y.shape
                
        # loss = 0.0
        # for c in range(y_hat.shape[1]):
        #     loss += self.loss_func(y_hat[:, c], y[:, c])

        return loss

    def training_step(self, batch: Batch, batch_idx) -> torch.tensor:
        ## compute landmark loss on each channel
        loss = self.landmark_loss(self(batch), batch.y)

        self.log(
            'train_loss', 
            loss, 
            batch_size = batch.num_graphs,
            on_step=False,
            on_epoch=True, 
            sync_dist = True)

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
            'val_nme',
            self.nme,
            batch_size = batch.num_graphs,
            on_step=False,
            on_epoch=True,
            sync_dist = True
            )

        self.log(
            'val_loss',
            val_loss,
            batch_size = batch.num_graphs,
            sync_dist = True
            )

        return {'val_nme': self.nme, 'val_loss': val_loss}