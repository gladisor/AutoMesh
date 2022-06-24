from torch.utils.data import Dataset
from pathlib import Path
import glob
import open3d as o3d

class LeftAtriumData(Dataset):
    def __init__(self, path: str) -> None:
        super().__init__()

        ## grab all .ply files
        paths = Path(path).glob('*.ply')
        paths = map(lambda x: x.as_posix(), paths)

        ## filter by data content
        mesh_paths = filter(lambda x: 'LeftAtrium.ply' in x, paths)
        branch_paths = filter(lambda x: 'LeftAtriumBranchPoints.ply' in x, paths)

        ## load dataset into memory
        self.mesh = list(map(lambda x: o3d.io.read_triangle_mesh(x), mesh_paths))
        self.branch = list(map(lambda x: o3d.io.read_triangle_mesh(x), branch_paths))

    def visualize(self, idx) -> None:
        self.mesh[idx].compute_vertex_normals()
        self.branch[idx].compute_vertex_normals()
        o3d.visualization.draw_geometries([self.mesh[idx], self.branch[idx]])