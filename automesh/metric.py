import torch
from torchmetrics import Metric
from pytorch_lightning.callbacks import Callback
from pytorch_lightning import Trainer, LightningModule

class NormalizedMeanError(Metric):
    '''
    Normalized mean error measures the average euclidian distance between pairs of points.
    We use this metric to evaluate the distance between predicted branching points and true
    branching points. This allows for robust comparison of different models even when they use 
    different loss functions.
    '''

    full_state_update = False
    def __init__(self):
        super().__init__()
        self.add_state('distance', default = torch.tensor(0.0), dist_reduce_fx = 'sum')
        self.add_state('total', default = torch.tensor(0), dist_reduce_fx = 'sum')
    
    def update(self, pred_points: torch.Tensor, true_points: torch.Tensor) -> torch.Tensor:
        assert pred_points.shape == true_points.shape
        self.distance += (true_points - pred_points).norm(dim = 1).sum()
        self.total += true_points.shape[0]

    def compute(self) -> torch.Tensor:
        return self.distance / self.total