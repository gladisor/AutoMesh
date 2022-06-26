from typing import List, Tuple
import torch
from torch.utils.data import Dataset
import numpy as np
from pathlib import Path
import glob
import open3d

def parse_heart_id(path: str) -> int:
    if '_LeftAtrium.ply' or 'LeftAtriumBranchPoints.ply' in path:
        heart_id = path.split('_LeftAtrium')[-2].split('-')[-1]
        return int(heart_id)
    else:
        return None
        
class LeftAtriumData(Dataset):
    def __init__(self, path: str) -> None:
        super().__init__()

        ## grab all .ply files
        paths = Path(path).glob('*.ply')
        paths = [x.as_posix() for x in paths]

        ## filter by data content
        mesh_paths = [x for x in paths if 'LeftAtrium.ply' in x]
        branch_paths = [x for x in paths if 'LeftAtriumBranchPoints.ply' in x]

        ## sort by heart id
        self.mesh_paths = sorted(mesh_paths, key = parse_heart_id)
        self.branch_paths = sorted(branch_paths, key = parse_heart_id)

        ## load dataset into memory
        self.mesh = [open3d.io.read_triangle_mesh(x) for x in self.mesh_paths]
        self.branch = [open3d.io.read_point_cloud(x) for x in self.branch_paths]

    def visualize(self, idx) -> None:
        ## visualizes the selected heart mesh
        self.mesh[idx].compute_vertex_normals()
        open3d.visualization.draw_geometries([self.mesh[idx]])

    def __getitem__(self, idx) -> Tuple[torch.tensor, torch.tensor, torch.tensor]:
        ## scale mesh between zero and one
        self.mesh[idx].compute_vertex_normals()
        
        v = torch.tensor(np.array(self.mesh[idx].vertex_normals))
        t = torch.tensor(np.array(self.mesh[idx].triangles))
        b = torch.tensor(np.array(self.branch[idx].points))
        return (v, t, b)

    def __len__(self) -> int:
        return len(self.mesh)