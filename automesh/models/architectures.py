import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn.models import GraphUNet
from torch_geometric.nn.models.basic_gnn import BasicGNN
from torch_geometric.nn.conv import MessagePassing



class Param_GCN(BasicGNN):
    
    def __init__(self,convlayer: MessagePassing, **kwargs):
        self.convlayer=convlayer
        super().__init__(**kwargs)
  
    def init_conv(self, in_channels: int, out_channels: int,
                  **kwargs) -> MessagePassing:
        return self.convlayer(in_channels, out_channels, **kwargs)
    
    
    
class Param_GraphUNet(GraphUNet):
   
    def __init__(self, convlayer: MessagePassing, poolinglayer: nn.Module , 
                 in_channels, hidden_channels, out_channels, depth, 
                 pool_ratios=0.5, sum_res=True, act=F.relu):
        
        self.convlayer=convlayer
        self.poolinglayer=poolinglayer                        
        super(GraphUNet, self).__init__(in_channels, hidden_channels, out_channels, 
                                        depth, pool_ratios=0.5, sum_res=True, act=F.relu )
        
        channels = hidden_channels

        self.down_convs = nn.ModuleList()
        self.pools = nn.ModuleList()
        self.down_convs.append(convlayer(in_channels, channels, improved=True))
        for i in range(depth):
            self.pools.append(poolinglayer(channels, self.pool_ratios[i]))
            self.down_convs.append(convlayer(channels, channels, improved=True))

        in_channels = channels if sum_res else 2 * channels

        self.up_convs = nn.ModuleList()
        for i in range(depth - 1):
            self.up_convs.append(convlayer(in_channels, channels, improved=True))
        self.up_convs.append(convlayer(in_channels, out_channels, improved=True))

        self.reset_parameters()

    


class BaseArchitecture(nn.Module):
    def __init__(self, **kwargs):
        self.optimizer_args = kwargs

    # @abstractclassmethod
    def optimizer_init(self):
        """
        Follow the pytorch lightning doccumentation to define optimizers for the archetecture.
        https://pytorch-lightning.readthedocs.io/en/stable/starter/converting.html
        """
        return