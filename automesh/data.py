from typing import List, Tuple
import copy
from pathlib import Path
import os
import sys

import open3d as o3d
import numpy as np
import torch
from torch_geometric.data import Data, Dataset
from scipy.spatial import distance

class LeftAtriumData(Dataset):
    def __init__(self, root: str, closest_k: int = 1, transform = None, pre_transform=None, pre_filter=None) -> None:
        super().__init__(root, transform, pre_transform, pre_filter)

        if not os.path.exists(self.raw_dir):
            raise NotADirectoryError("No raw data folder or data is missing.")
            sys.exit(1)

        self.mesh_paths, self.branch_paths = LeftAtriumData.get_ordered_paths(self.raw_file_names)
        self.closest_k = closest_k

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
    def get_edges_from_triangles(triangles: np.array) -> torch.tensor:
        edges = np.concatenate([
            triangles[:, [0, 1]],
            triangles[:, [1, 2]],
            triangles[:, [0, 2]]], axis = 0)

        edges = np.sort(edges, axis = 1)
        edges = np.unique(edges, axis = 0)

        return torch.tensor(edges, dtype = torch.long).T

    def len(self) -> int:
        return len(self.mesh_paths)

    def get(self, idx):
        ## loading from file
        mesh = o3d.io.read_triangle_mesh(self.mesh_paths[idx])
        branch = o3d.io.read_point_cloud(self.branch_paths[idx])

        ## extracting points
        vertices = np.array(mesh.vertices)
        orig_branch_points = np.array(branch.points)
        _, branch_points_idx = np.unique(orig_branch_points, axis = 0, return_index = True)
        branch_points = orig_branch_points[np.sort(branch_points_idx)]

        ## computing nearest neibhors on mesh
        d = distance.cdist(branch_points, vertices)
        mesh_branch_points = torch.tensor(d).topk(k = self.closest_k, dim = 1, largest = False).indices.squeeze(-1)
        vertices = torch.tensor(vertices, dtype = torch.float32)

        data = Data(
            x = vertices,
            pos = vertices,
            edge_index = LeftAtriumData.get_edges_from_triangles(np.array(mesh.triangles)),
            y = mesh_branch_points)

        return data

    def display(self, idx):
        mesh = copy.deepcopy(o3d.io.read_triangle_mesh(self.mesh_paths[idx]))
        branch = o3d.io.read_point_cloud(self.branch_paths[idx])

        for point in np.unique(branch.points, axis = 0):
            sphere = o3d.geometry.TriangleMesh.create_sphere()
            sphere.scale(1, center=sphere.get_center())
            sphere.translate(point)
            mesh += sphere

        mesh.compute_vertex_normals()
        o3d.visualization.draw_geometries([mesh])