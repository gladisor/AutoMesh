## allows us to access the automesh library from outside
import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

## third party
import open3d as o3d
import numpy as np

## local source
import automesh

data = o3d.io.read_triangle_mesh("/Users/tristanshah/Desktop/1000shapes/AutoMesh/data/GRIPS22/DE-UKU-02-0001_LeftAtrium.ply")
print(data)
print("HELLO PPL")

# points = np.array(pcd.points)
# normals = np.array(pcd.normals)
# covariances = np.array(pcd.covariances)

# print(covariances)