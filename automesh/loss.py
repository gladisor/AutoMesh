import torch
import torch.nn as nn

## adapted from:
# https://github.com/elliottzheng/AdaptiveWingLoss/blob/master/adaptive_wing_loss.py
class AdaptiveWingLoss(nn.Module):
    def __init__(
        self, 
        omega: float = 14.0,
        theta: float = 0.5,
        epsilon: float = 1.0,
        alpha: float = 2.1) -> None:
        super().__init__()

        self.omega = omega
        self.theta = theta
        self.epsilon = epsilon
        self.alpha = alpha
    
    def forward(self, y_hat, y) -> torch.tensor:
        delta_y = (y - y_hat).abs()

        delta_y1 = delta_y[delta_y < self.theta]
        delta_y2 = delta_y[delta_y >= self.theta]

        y1 = y[delta_y < self.theta]
        y2 = y[delta_y >= self.theta]

        loss1 = self.omega * torch.log(1 + torch.pow(delta_y1 / self.omega, self.alpha - y1))
        
        A = self.omega * (1 / (1 + torch.pow(self.theta / self.epsilon, self.alpha - y2))) * (self.alpha - y2) * (
            torch.pow(self.theta / self.epsilon, self.alpha - y2 - 1)) * (1 / self.epsilon)

        C = self.theta * A - self.omega * torch.log(1 + torch.pow(self.theta / self.epsilon, self.alpha - y2))
        loss2 = A * delta_y2 - C

        return (loss1.sum() + loss2.sum()) / (len(loss1) + len(loss2))