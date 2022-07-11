from typing import Callable, Union, Dict, Any

import numpy as np
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
            # opt: Optimizer,
            # opt_kwargs: Dict[str, Any],
            lr: float,
            loss_func: nn.Module,
            # loss_kwargs: Dict[str, Any],
            **kwargs) -> None:
        super().__init__()

        self.base = base(**kwargs)
        self.optimizer = optimizer
        # self.opt = opt(**opt_kwargs)
        # self.opt = opt
        # self.opt_kwargs = opt_kwargs
        self.lr = lr
        self.loss_func = loss_func
        # self.loss_func = loss_func(**loss_kwargs)

        self.save_hyperparameters()

    @staticmethod
    def predict_points(heatmap: torch.tensor, points: torch.tensor) -> torch.tensor:
        idx = heatmap.argmax(dim = 0) # get max value index for each landmark
        return points[idx, :] # extract the coordinates


    #not yet finalized verify that index comes out correctly
    #get highest neighborhood average. center vertex is considered to be branch point
    @staticmethod
    def predict_smoothed_points(heatmap: torch.tensor, data: Data) -> torch.tensor:
         # get max value index for each landmark
        avg=torch.zeros(heatmap.shape)
        for j in range(avg.size(1)):
            for i in range(avg.size(0)):
                neighborhood=[]
                neighborhood.append(heatmap[i,j])
                data.edge_index
                for x in range(data.edge_index.size(1)):
                    if data.edge_index[0][x]==i:
                        neighbornode=data.edge_index[1][x]
                        neighborhood.append(heatmap[neighbornode.item()][j])
                avg[i][j]=sum(neighborhood)/len(neighborhood) 
        idx = avg.argmax(dim = 0)
        
        print ("idx: ------------ ", idx)  ###check steps here
        
        return data.x[idx, :]# extract the coordinates
    
    @staticmethod 
    def predict_confidence():
        pass 
    
    
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
        loss = self.landmark_loss(self(batch), batch.y)
        self.log('train_loss', loss, batch_size = batch.num_graphs)
        return loss

    def validation_step(self, batch, batch_idx):

        with torch.no_grad():
            distance = 0.0
            for i in range(batch.num_graphs):
                data = batch.get_example(i)
                distance += HeatMapRegressor.evaluate_heatmap(self(data), data)

            distance /= batch.num_graphs

            val_loss = self.landmark_loss(self(batch), batch.y)
        
        self.log('val_performance', {'val_loss': val_loss, 'distance': distance}, batch_size = batch.num_graphs)