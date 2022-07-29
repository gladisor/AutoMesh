## allows us to access the automesh library from outside
import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from glob import glob
import random
import shutil

from automesh.data.data import LeftAtriumHeatMapData

def build_cross_validation_datasets(source: str, dest: str, sets: int, split: float):
    paths = glob(os.path.join(source, '/*.ply'))
    mesh, branch = LeftAtriumHeatMapData.get_ordered_paths(paths)
    split_idx = int(len(mesh) * split)

    print(mesh)

    for i in range(sets):
        set_dest = os.path.join(dest, f'cv_{i}/')
        os.makedirs(set_dest, exist_ok = True)
        data = list(zip(mesh, branch))

        print(mesh, branch)
        random.shuffle(data)

        for x, y in zip(*data):
            print(x)
            shutil.copy(x, dest)
        
if __name__ == '__main__':

    build_cross_validation_datasets(
        source = 'data/full_dataset', 
        dest = 'data',
        sets = 2, 
        split = 0.9)