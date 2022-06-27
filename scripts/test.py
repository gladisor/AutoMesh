## allows us to access the automesh library from outside
import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

## third party
from torch_geometric.nn import GCNConv

## local source
from automesh.data import LeftAtriumData

if __name__ == '__main__':
    data = LeftAtriumData('data/GRIPS22/')
    v, e, b = data[0]
    m = GCNConv(3, 1)