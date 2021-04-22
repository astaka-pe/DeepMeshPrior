import csv
import open3d as o3d
import numpy as np
import torch


input_file = 'datasets/bumpy.ply'
input_mesh = o3d.io.read_triangle_mesh(input_file)
rand_vs = np.random.randn(np.asarray(input_mesh.vertices).shape[0], 3)
input_mesh.vertices = o3d.utility.Vector3dVector(rand_vs)
o3d.io.write_triangle_mesh('datasets/bumpy_rand.ply', input_mesh)