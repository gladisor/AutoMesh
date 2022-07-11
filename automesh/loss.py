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

class DiceLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, y_hat: torch.tensor, y: torch.tensor) -> torch.tensor:
        p = torch.sigmoid(y_hat)

        intersection = p.dot(y)

        dice = (2 * intersection + 1) / (p.sum() + y.sum() + 1)

        return 1 - dice

class BCEDiceLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, y_hat: torch.tensor, y: torch.tensor) -> torch.tensor:
        p = torch.sigmoid(y_hat)

        intersection = p.dot(y)

        dice = 1 - (2 * intersection + 1) / (p.sum() + y.sum() + 1)
        bce = nn.functional.binary_cross_entropy(p, y)

        return dice + bce

class JaccardLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, y_hat: torch.tensor, y: torch.tensor) -> torch.tensor:
        p = torch.sigmoid(y_hat)
        intersection = (p * y).sum()
        total = (p + y).sum()
        union = total - intersection
        J = 1 - (intersection + 1) / (union + 1)

        return J

class FocalLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, y_hat: torch.tensor, y: torch.tensor) -> torch.tensor:
        p = torch.sigmoid(y_hat)
        bce = nn.functional.binary_cross_entropy(p, y)
        bce_exp = torch.exp(-bce)
        focal = 0.8 * (1 - bce_exp) ** 2 * bce
        return focal

class TverskyLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, y_hat: torch.tensor, y: torch.tensor) -> torch.tensor:
        # p = torch.sigmoid(y_hat)
        # p = y_hat

        # #True Positives, False Positives & False Negatives
        # TP = (p * y).sum()    
        # FP = ((1-y) * p).sum()
        # FN = (y * (1-p)).sum()
       
        # Tversky = (TP + 1) / (TP + 0.5*FP + 0.5*FN + 1)
        
        # return 1 - Tversky
        return None