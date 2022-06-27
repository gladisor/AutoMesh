from typing import List, Tuple
import copy
from pathlib import Path
import os
import sys

import open3d as o3d
import numpy as np
import torch
from torch_geometric.data import Dataset

class LeftAtriumData(Dataset):
    def __init__(self, root: str, transform = None, pre_transform=None, pre_filter=None) -> None:
        super().__init__(root, transform, pre_transform, pre_filter)

        if not os.path.isdir(self.raw_dir):
            raise NotADirectoryError("No raw data folder!")
            sys.exit(1)

        self.mesh_paths, self.branch_paths = LeftAtriumData.get_ordered_paths(self.processed_file_names)

    @property
    def raw_file_names(self):
        paths = Path(self.raw_dir).glob('*.ply')
        paths = [x.as_posix() for x in paths]
        return paths

    @property
    def processed_file_names(self):
        paths = Path(self.raw_dir).glob('*.ply')
        paths = [Path(self.processed_dir) / path.name for path in paths]
        paths = [x.as_posix() for x in paths]
        return paths

    @staticmethod
    def get_ordered_paths(paths: List[str]) -> Tuple[List, List]:
        ## filter by data content
        mesh_paths = [x for x in paths if 'LeftAtrium.ply' in x]
        branch_paths = [x for x in paths if 'LeftAtriumBranchPoints.ply' in x]

        ## sort by heart id
        mesh_paths = sorted(mesh_paths, key = LeftAtriumData.parse_heart_id)
        branch_paths = sorted(branch_paths, key = LeftAtriumData.parse_heart_id)

        return (mesh_paths, branch_paths)

    @staticmethod
    def parse_heart_id(path: str) -> int:
        if '_LeftAtrium.ply' or 'LeftAtriumBranchPoints.ply' in path:
            heart_id = path.split('_LeftAtrium')[-2].split('-')[-1]
            return int(heart_id)
        else:
            return None

    @staticmethod
    def normalize_geometry(geom: o3d.geometry.Geometry3D) -> o3d.geometry.Geometry3D:
        x = copy.deepcopy(geom)
        x.translate(x.get_center() * -1)
        scale_coef = 1 / np.linalg.norm(x.get_max_bound())
        x.scale(scale_coef, x.get_center())

        return x

    @staticmethod
    def get_edges_from_triangles(triangles: np.array) -> np.array:
        edges = np.concatenate([
            triangles[:, [0, 1]],
            triangles[:, [1, 2]]])

        edges = np.sort(edges, axis = 1)
        edges = np.unique(edges, axis = 0)

        return edges

    def process(self):

        raw_paths = LeftAtriumData.get_ordered_paths(self.raw_file_names)
        processed_paths = LeftAtriumData.get_ordered_paths(self.processed_file_names)

        if len(raw_paths[0]) != len(raw_paths[1]):
            raise RuntimeWarning("Number of heart meshes does not match number of branching points!")
        
        for r_m_path, r_b_path, p_m_path, p_b_path in zip(*raw_paths, *processed_paths):
            ## load dataset into memory
            mesh = o3d.io.read_triangle_mesh(r_m_path)
            branch = o3d.io.read_point_cloud(r_b_path)

            ## normalizing mesh between -1 and 1 and centering
            mesh = LeftAtriumData.normalize_geometry(mesh)
            branch = LeftAtriumData.normalize_geometry(branch)

            ## save transformed data
            o3d.io.write_triangle_mesh(p_m_path, mesh)
            o3d.io.write_point_cloud(p_b_path, branch)

    def len(self) -> int:
        return len(self.processed_file_names)

    def get(self, idx):

        mesh = o3d.io.read_triangle_mesh(self.mesh_paths[idx])
        branch = o3d.io.read_point_cloud(self.branch_paths[idx])

        vertices = torch.tensor(np.array(mesh.vertices), dtype = torch.float32)
        
        edges = LeftAtriumData.get_edges_from_triangles(np.array(mesh.triangles))
        edges = torch.tensor(edges, dtype = torch.long)
        
        return (vertices, edges)

