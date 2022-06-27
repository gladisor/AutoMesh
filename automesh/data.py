from typing import List, Tuple
import torch
from torch.utils.data import Dataset
import numpy as np
from pathlib import Path
import glob
import open3d as o3d
import copy

def parse_heart_id(path: str) -> int:
    if '_LeftAtrium.ply' or 'LeftAtriumBranchPoints.ply' in path:
        heart_id = path.split('_LeftAtrium')[-2].split('-')[-1]
        return int(heart_id)
    else:
        return None

def normalize_geometry(geom: o3d.geometry.Geometry3D) -> o3d.geometry.Geometry3D:
    x = copy.deepcopy(geom)
    x.translate(x.get_center() * -1)
    scale_coef = 1 / np.linalg.norm(x.get_max_bound())
    x.scale(scale_coef, x.get_center())

    return x

def get_edges_from_triangles(triangles: np.array) -> np.array:
    edges = np.concatenate([
        triangles[:, [0, 1]],
        triangles[:, [1, 2]]])

    edges = np.sort(edges, axis = 1)
    edges = np.unique(edges, axis = 0)

    return edges
        
class LeftAtriumData(Dataset):
    def __init__(self, path: str, normalize = True) -> None:
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
        self.mesh = [o3d.io.read_triangle_mesh(x) for x in self.mesh_paths]
        self.branch = [o3d.io.read_point_cloud(x) for x in self.branch_paths]

        ## scale geometry between -1 and 1
        if normalize:
            self.mesh = [normalize_geometry(mesh) for mesh in self.mesh]
            self.branch = [normalize_geometry(branch) for branch in self.branch]

    def visualize(self, idx) -> None:
        ## visualizes the selected heart mesh
        self.mesh[idx].compute_vertex_normals()
        o3d.visualization.draw_geometries([self.mesh[idx]])

    def __getitem__(self, idx) -> Tuple[torch.tensor, torch.tensor, torch.tensor]:
        ## scale mesh between zero and one

        vert = torch.tensor(np.array(self.mesh[idx].vertices), dtype = torch.float32)
        tri = np.array(self.mesh[idx].triangles)
        edge = torch.tensor(get_edges_from_triangles(tri), dtype = torch.long)
        branch = torch.tensor(np.array(self.branch[idx].points))

        return (vert, edge, branch)

    def __len__(self) -> int:
        return len(self.mesh)