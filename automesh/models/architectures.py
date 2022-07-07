from abc import abstractclassmethod
from typing import Callable

import torch.nn as nn

class BaseArchitecture(nn.Module):
    def __init__(self, **kwargs):
        self.optimizer_args = kwargs

    @abstractclassmethod
    def optimizer_init(self):
        """
        Follow the pytorch lightning doccumentation to define optimizers for the archetecture.
        https://pytorch-lightning.readthedocs.io/en/stable/starter/converting.html
        """
        return