import torch
from torch import Tensor
import torch.nn as nn

class ChannelWiseLoss(nn.Module):
    def __init__(self, loss_func: nn.Module):
        super().__init__()
        self.loss_func = loss_func

    def forward(self, y_hat: Tensor, y: Tensor):

        assert y_hat.shape == y.shape
        
        loss = 0.0
        for c in range(y_hat.shape[1]):
            loss += self.loss_func(y_hat[:, c], y[:, c])

        return loss

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
    
    def forward(self, y_hat: Tensor, y: Tensor) -> Tensor:
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

## Adapted from
# https://www.kaggle.com/code/bigironsphere/loss-function-library-keras-pytorch/notebook
class DiceLoss(nn.Module):
    def __init__(self, smooth: float = 1e-6):
        super().__init__()
        self.smooth = smooth

    def forward(self, y_hat: Tensor, y: Tensor) -> Tensor:
        p = torch.sigmoid(y_hat)

        intersection = (p * y).sum()
        dice = (2.0 * intersection + self.smooth) / (p.sum() + y.sum() + self.smooth)
        return 1 - dice

class BCEDiceLoss(DiceLoss):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def forward(self, y_hat: Tensor, y: Tensor) -> Tensor:
        p = torch.sigmoid(y_hat)

        intersection = (p * y).sum()
        dice = 1 - (2.0 * intersection + self.smooth) / (p.sum() + y.sum() + self.smooth)
        bce = nn.functional.binary_cross_entropy(p, y, reduction = 'mean')

        return dice + bce

class JaccardLoss(DiceLoss):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def forward(self, y_hat: Tensor, y: Tensor) -> Tensor:
        p = torch.sigmoid(y_hat)

        intersection = (p * y).sum()
        total = (p + y).sum()
        union = total - intersection
        J = 1 - (intersection + self.smooth) / (union + self.smooth)

        return J

## https://arxiv.org/abs/1708.02002
class FocalLoss(nn.Module):
    def __init__(self, alpha_f: float = 0.8, gamma_f: float = 2.0):
        super().__init__()
        self.alpha = alpha_f
        self.gamma = gamma_f

    def forward(self, y_hat: Tensor, y: Tensor) -> Tensor:
        p = torch.sigmoid(y_hat)

        bce = nn.functional.binary_cross_entropy(p, y, reduction = 'none')
        bce_exp = torch.exp(-bce)
        focal = (self.alpha * (1 - bce_exp) ** self.gamma) * bce
        return focal.mean()

## https://arxiv.org/abs/1706.05721
class TverskyLoss(DiceLoss):
    """
    alpha controls how much to penalize false positives vs valse negatives
    high alpha (1.0) means more penalty towards false positives while low
    alpha (0.0) means more penalty towards false negatives.

    alpha == beta == 0.5 is equivalent to dice loss. 
    """
    def __init__(self, alpha_t: float = 0.5, **kwargs):
        super().__init__(**kwargs)
        assert 0.0 <= alpha_t <= 1.0
        self.alpha = alpha_t
        self.beta = 1.0 - alpha_t

    def forward(self, y_hat: Tensor, y: Tensor) -> Tensor:
        p = torch.sigmoid(y_hat)

        tp = (p * y).sum()
        fp = ((1.0 - y) * p).sum()
        fn = (y * (1.0 - p)).sum()

        tv = (tp + self.smooth) / (tp + self.alpha * fp + self.beta * fn)

        return 1.0 - tv

class FocalTverskyLoss(TverskyLoss):
    def __init__(self, gamma_ft: float = 2.0, **kwargs):
        super().__init__(**kwargs)
        self.gamma = gamma_ft

    def forward(self, y_hat: Tensor, y: Tensor) -> Tensor:
        return super().forward(y_hat, y) ** (1/self.gamma)
