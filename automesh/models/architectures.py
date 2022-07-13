import torch.nn as nn
from typing import Dict, Any
from torch_geometric.nn.models import GraphUNet
from torch_geometric.nn.models.basic_gnn import BasicGNN
from torch_geometric.nn.conv import MessagePassing

class ParamGCN(BasicGNN):
    def __init__(self,conv_layer: MessagePassing, 
                 conv_kwargs: Dict[str, Any], 
                 **kwargs):
        self.conv_layer=conv_layer
        self.conv_kwargs=conv_kwargs
        super().__init__(**kwargs)
  
    def init_conv(self, in_channels: int, out_channels: int,
                  **kwargs) -> MessagePassing:
        return self.conv_layer(in_channels, out_channels,**self.conv_kwargs)
    
class ParamGraphUNet(GraphUNet):
    def __init__(self, 
                 conv_layer: MessagePassing, 
                 conv_kwargs: Dict[str, Any],
                 pool_layer: nn.Module , 
                 pool_kwargs: Dict[str, Any], 
                 act: nn.Module,
                 act_kwargs: Dict[str, Any],
                 in_channels, 
                 hidden_channels, 
                 out_channels, 
                 depth: int, 
                 sum_res=True, 
                 ):
        super().__init__(in_channels, hidden_channels, out_channels, depth, sum_res)
        self.conv_layer=conv_layer
        self.conv_kwargs=conv_kwargs
        self.pool_layer=pool_layer  
        self.act=act(**act_kwargs)
        self.pool_kwargs=pool_kwargs                 

        channels = hidden_channels

        self.down_convs = nn.ModuleList()
        self.pools = nn.ModuleList()
        self.down_convs.append(conv_layer(in_channels, channels, **conv_kwargs))
        for i in range(depth):
            self.pools.append(pool_layer(channels, **pool_kwargs) ) # self.pool_ratios[i]**pooling_kwargs
            self.down_convs.append(conv_layer(channels, channels, **conv_kwargs))

        in_channels = channels if sum_res else 2 * channels

        self.up_convs = nn.ModuleList()
        for i in range(depth - 1):
            self.up_convs.append(conv_layer(in_channels, channels, **conv_kwargs))
        self.up_convs.append(conv_layer(in_channels, out_channels, **conv_kwargs))
        self.reset_parameters()