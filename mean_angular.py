import open3d as o3d
import numpy as np
import sys
from numpy import linalg as LA


argv = sys.argv
argc = len(argv)
if argc != 3:
    print('Usage: python gcnconv.py "input_dir"')
    quit()

path1 = argv[1]
path2 = argv[2]

mesh1 = o3d.io.read_triangle_mesh(path1)
mesh2 = o3d.io.read_triangle_mesh(path2)

mesh1.compute_triangle_normals()
mesh2.compute_triangle_normals()

normal1 = np.asarray(mesh1.triangle_normals)
normal2 = np.asarray(mesh2.triangle_normals)

inner = [np.inner(normal1[i], normal2[i]) for i in range(normal1.shape[0])]
#norm = [LA.norm(normal1[i]) * LA.norm(normal2[i]) for i in range(normal1.shape[0])]
sad = np.rad2deg(np.arccos(np.clip(inner, -1.0, 1.0)))
mad = np.sum(sad) / len(sad)
print("mean_angular_difference: ", mad)