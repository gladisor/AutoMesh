## allows us to access the automesh library from outside
import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from glob import glob
from pprint import pprint

from automesh.data.data import LeftAtriumHeatMapData

if __name__ == '__main__':
    paths = glob('data/full_dataset/*.ply')
    mesh, branch = LeftAtriumHeatMapData.get_ordered_paths(paths)

    for m, b in zip(mesh, branch):
        print(m, b)
    
