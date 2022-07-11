from torch_geometric.nn.models.basic_gnn import BasicGNN
from torch_geometric.nn.conv import MessagePassing

class ParamGCN(BasicGNN):
    def __init__(self,convlayer: MessagePassing, **kwargs):
        self.convlayer=convlayer
        super().__init__(**kwargs)
  
    def init_conv(self, in_channels: int, out_channels: int,
                  **kwargs) -> MessagePassing:
        return self.convlayer(in_channels, out_channels, **kwargs)