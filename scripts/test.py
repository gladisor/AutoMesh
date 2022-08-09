## allows us to access the automesh library from outside
import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from glob import glob
import random
import shutil

from automesh.data.data import LeftAtriumHeatMapData

def build_cross_validation_datasets(source: str, dest: str, sets: int, split: float):
    paths = glob(os.path.join(source, '*.ply'))
    mesh, branch = LeftAtriumHeatMapData.get_ordered_paths(paths)
    split_idx = int(len(mesh) * split)

    for i in range(sets):
        set_dest = os.path.join(dest, f'cv_{i}/')
        train_dest = os.path.join(set_dest, 'train/raw')
        val_dest = os.path.join(set_dest, 'val/raw')

        os.makedirs(set_dest)
        os.makedirs(train_dest)
        os.makedirs(val_dest)

        data = list(zip(mesh, branch))
        random.shuffle(data)

        for x, y in data[0:split_idx]:
            shutil.copy(x, train_dest)
            shutil.copy(y, train_dest)
        
        for x, y in data[split_idx:]:
            shutil.copy(x, val_dest)
            shutil.copy(y, val_dest)

if __name__ == '__main__':

    build_cross_validation_datasets(
        source = 'data/full_dataset', 
        dest = 'data',
        sets = 5,
        split = 0.95)
