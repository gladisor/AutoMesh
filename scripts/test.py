## Allows us to access the automesh library from outside
import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from automesh import data
import open3d as o3d

pcd = o3d.io.read_point_cloud("/Users/tristanshah/Desktop/1000shapes/AutoMesh/data/GRIPS22/DE-UKU-02-0001_LeftAtrium.ply")
print(pcd)