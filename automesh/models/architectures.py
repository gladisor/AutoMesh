import torch.nn as nn
from typing import Dict, Any
from torch_geometric.nn.models.basic_gnn import BasicGNN
from torch_geometric.nn.conv import MessagePassing

class ParamGCN(BasicGNN):
    '''
    Stacks a graph neural network layer a variable number of times with activation and normalization
    in between.  
    '''
    def __init__(self,conv_layer: MessagePassing, conv_layer_kwargs: Dict[str, Any], **kwargs):
        self.conv_layer = conv_layer
        self.conv_layer_kwargs = conv_layer_kwargs
        super().__init__(**kwargs)

    def init_conv(self, in_channels: int, out_channels: int, **kwargs) -> MessagePassing:
        return self.conv_layer(in_channels, out_channels, **self.conv_layer_kwargs)