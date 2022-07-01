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
from scipy.special import softmax

class LeftAtriumData(Dataset):
    """
    Extracts information from a set of .ply files located in {root}/raw. Handles
    preprocessing and construction of a torch_geometric graph automatically. Will
    also apply any transformations specified. Labels for each data point will be
    the indexes of verticies in the mesh which are closest to the branching points
    contained in the LeftAtriumBranchPoints.ply files.
    """
    def __init__(self, 
            root: str, 
            triangles: int = 2000, 
            transform = None, 
            pre_transform = None,
            pre_filter = None) -> None:

        self.root = root
        ## check for incorrect file structure
        if not os.path.exists(self.raw_dir):
            raise NotADirectoryError("No raw data folder or data is missing.")
            sys.exit(1)

        self.triangles = triangles
        ## grab the paths to the meshes and branch points in order
        self.mesh_paths, self.branch_paths = LeftAtriumData.get_ordered_paths(self.processed_file_names)
        super().__init__(root, transform, pre_transform, pre_filter)

    @property
    def raw_file_names(self) -> List[str]:
        paths = Path(self.raw_dir).glob('*.ply')
        paths = [x.as_posix() for x in paths]
        return paths

    @property
    def processed_file_names(self) -> List[str]:
        paths = Path(self.raw_dir).glob('*.ply')
        paths = [Path(self.processed_dir) / path.name for path in paths]
        paths = [x.as_posix() for x in paths]
        return paths

    @staticmethod
    def parse_heart_id(path: str) -> int:
        ## extract the id from each file name
        if '_LeftAtrium.ply' or 'LeftAtriumBranchPoints.ply' in path:
            heart_id = path.split('_LeftAtrium')[-2].split('-')[-1]
            return int(heart_id)
        else:
            return None

    @staticmethod
    def get_ordered_paths(paths: List[str]) -> Tuple[List, List]:
        ## filter by data content
        mesh_paths = [x for x in paths if 'LeftAtrium.ply' in x]
        branch_paths = [x for x in paths if 'LeftAtriumBranchPoints.ply' in x]

        ## sort by heart id
        mesh_paths = sorted(mesh_paths, key = LeftAtriumData.parse_heart_id)
        branch_paths = sorted(branch_paths, key = LeftAtriumData.parse_heart_id)
        return (mesh_paths, branch_paths)

    def process(self) -> None:

        if not os.path.isdir(self.processed_dir) or not os.listdir(self.processed_dir):
            raw_mesh_paths, raw_branch_paths = LeftAtriumData.get_ordered_paths(self.raw_file_names)

            for idx in range(len(raw_mesh_paths)):
                ## loading from file
                mesh = o3d.io.read_triangle_mesh(raw_mesh_paths[idx])
                branch = o3d.io.read_point_cloud(raw_branch_paths[idx])

                if self.triangles != None:
                    ## re mesh with less faces if desired
                    mesh = mesh.simplify_quadric_decimation(self.triangles)

                o3d.io.write_triangle_mesh(self.mesh_paths[idx], mesh)
                o3d.io.write_point_cloud(self.branch_paths[idx], branch)

    def len(self) -> int:
        return len(self.mesh_paths)

    def get(self, idx) -> Data:
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
        mesh_branch_points = torch.tensor(d.argmin(axis = 1), dtype = torch.long)
        vertices = torch.tensor(vertices, dtype = torch.float32)
        faces = torch.tensor(np.array(mesh.triangles), dtype = torch.long).T

        ## create a torch_geometric Data object which holds information about a graph
        return Data(x = vertices, pos = vertices, face = faces, y = mesh_branch_points)

    def display(self, idx) -> None:
        ## function which renders the mesh and branching points of a particular data point
        mesh = copy.deepcopy(o3d.io.read_triangle_mesh(self.mesh_paths[idx]))
        branch = o3d.io.read_point_cloud(self.branch_paths[idx])

        for point in np.unique(branch.points, axis = 0):
            sphere = o3d.geometry.TriangleMesh.create_sphere()
            sphere.scale(1, center=sphere.get_center())
            sphere.translate(point)
            mesh += sphere

        mesh.compute_vertex_normals()
        o3d.visualization.draw_geometries([mesh])

class LeftAtriumHeatMapData(LeftAtriumData):
    def __init__(self, 
            root: str, 
            sigma: float,
            triangles: int = 2000,
            transform=None, 
            pre_transform=None, 
            pre_filter=None) -> None:

        super().__init__(root, triangles, transform, pre_transform, pre_filter)
        self.sigma = sigma

    def process(self) -> None:
        return super().process()

    def get(self, idx: int) -> Data:
        data = super().get(idx)

        branch_points = data.pos[data.y]

        D = distance.cdist(data.pos, branch_points)
        H = np.exp(- np.power(D, 2) / (2 * self.sigma))

        data.y = H

        return data