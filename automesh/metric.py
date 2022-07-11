from torchmetrics import Metric
import torch


class NormalizedMeanError(Metric):
    def __init__(self):
        super().__init__()
        self.add_state('distance', default = torch.tensor(0.0), dist_reduce_fx = 'sum')
        self.add_state('total', default = torch.tensor(0), dist_reduce_fx = 'sum')
    
    def update(self, pred_points: torch.Tensor, true_points: torch.Tensor) -> torch.Tensor:
        self.distance += (true_points - pred_points).norm(dim = 1).sum()
        self.total += true_points.shape[0]

    def compute(self) -> torch.Tensor:
        return self.distance / self.total
