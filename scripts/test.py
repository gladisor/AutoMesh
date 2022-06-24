## allows us to access the automesh library from outside
import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

## third party
import open3d as o3d
import numpy as np
from torch.utils.data import Dataset
from pathlib import Path

## local source
from automesh.data import LeftAtriumData

if __name__ == '__main__':

    data = LeftAtriumData('data/GRIPS22/')

    data.visualize(0)