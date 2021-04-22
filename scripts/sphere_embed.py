import os
import sys
import argparse

import numpy as np
import scipy as sp
import scipy.sparse
import scipy.optimize

current_dir_path = os.path.dirname(os.path.abspath(__file__))
module_root_path = os.path.abspath(os.path.join(current_dir_path, os.pardir))
sys.path.append(module_root_path)

from util.objmesh import ObjMesh


def main(filename, maxiter):
    # open mesh file
    mesh = ObjMesh(filename)
    vertex = mesh.vertices
    faces = mesh.indices.reshape((-1, 3))
    num_verts = len(vertex)
    num_faces = len(faces)
    print('#vertex: %d' % (num_verts))
    print(' #faces: %d' % (num_faces))

    # check vertex neighbors
    neighbors = [set() for _ in range(num_verts)]
    for f in faces:
        neighbors[f[0]] = neighbors[f[0]].union(set(f))
        neighbors[f[1]] = neighbors[f[1]].union(set(f))
        neighbors[f[2]] = neighbors[f[2]].union(set(f))

    for i in range(num_verts):
        neighbors[i] = list(neighbors[i].difference(set([i])))

    # construct Laplacian matrix
    inds = []
    jnds = []
    vals = []
    for i in range(num_verts):
        nn = len(neighbors[i])
        inds.extend([i] * nn)
        jnds.extend(neighbors[i])
        vals.extend([-1.0] * nn)

    W = sp.sparse.csr_matrix((vals, (inds, jnds)), shape=(num_verts, num_verts))
    diags = np.asarray(np.sum(W, axis=1)).reshape(-1)
    D = sp.sparse.dia_matrix((diags, 0), shape=(num_verts, num_verts))
    Lw = -D + W

    # utility methods for optimization
    def lossfun(x_and_a, num_verts, Lw):
        x = x_and_a[:num_verts * 3].reshape((3, num_verts))
        a = x_and_a[num_verts * 3:].reshape((1, num_verts))

        r2 = np.sum(x * x, axis=0, keepdims=True)
        l_sph = np.sum((r2 - 1.0)**2.0)

        residue = a * x - sp.sparse.csr_matrix.dot(x, Lw.T)
        l_res = np.sum(residue**2.0)

        return l_sph + l_res

    def gradfun(x_and_a, num_verts, Lw):
        x = x_and_a[:num_verts * 3].reshape((3, num_verts))
        a = x_and_a[num_verts * 3:].reshape((num_verts))

        r2 = np.sum(x * x, axis=0, keepdims=True)
        grad_sph = 4.0 * (r2 - 1.0) * x

        A = sp.sparse.dia_matrix((a, 0), shape=(num_verts, num_verts))
        A_minus_Lw = A - Lw
        residue = sp.sparse.csr_matrix.dot(x, A_minus_Lw.T)
        grad_res_x = 2.0 * sp.sparse.csr_matrix.dot(residue, A_minus_Lw)
        grad_res_a = 2.0 * np.sum(x * residue, axis=0)

        grad_x = grad_sph + grad_res_x
        grad_a = grad_res_a
        grad = np.concatenate([grad_x.reshape(-1), grad_a.reshape(-1)])

        return grad

    def hessfun(x_and_a, num_verts, Lw):
        x = x_and_a[:num_verts * 3].reshape((3, num_verts))
        a = x_and_a[num_verts * 3:].reshape((num_verts))

        hess_sph_diags = np.einsum('xn,yn->nxy', x, x) * 8.0
        inds = []
        jnds = []
        vals = []
        for i in range(num_verts):
            for k in range(3):
                for l in range(3):
                    inds.append(k * num_verts + i)
                    jnds.append(l * num_verts + i)
                    vals.append(hess_sph_diags[i, k, l])

        hess_sph_x = sp.sparse.csr_matrix((vals, (inds, jnds)), shape=(num_verts * 3, num_verts * 3))

        r2 = np.sum(x * x, axis=0, keepdims=True)
        r2 = np.tile(r2, (3, 1)).reshape(-1)
        hess_sph_r2 = sp.sparse.dia_matrix((4.0 * (r2 - 1.0), 0), shape=(num_verts * 3, num_verts * 3))
        hess_sph_x = hess_sph_x + hess_sph_r2

        A = sp.sparse.dia_matrix((a, 0), shape=(num_verts, num_verts))
        A_minus_Lw = A - Lw

        hess_res_x = 2.0 * sp.sparse.csr_matrix.dot(A_minus_Lw.T, A_minus_Lw)
        hess_res_x = sp.sparse.block_diag((hess_res_x, ) * 3)
        hess_res_a = 2.0 * np.sum(x * x, axis=0)
        hess_res_a = sp.sparse.dia_matrix((hess_res_a, 0), shape=(num_verts, num_verts))

        hess_x = hess_sph_x + hess_res_x
        hess_a = hess_res_a
        hess = sp.sparse.block_diag((hess_x, hess_a))

        return hess

    # optimization
    x0 = np.transpose(vertex, axes=(1, 0)).flatten()
    a0 = np.zeros(num_verts)
    x_and_a = np.concatenate([x0, a0])

    # trust region method
    options = {'maxiter': maxiter, 'verbose': 2, 'disp': True}
    res = sp.optimize.minimize(lossfun,
                               x0=x_and_a,
                               jac=gradfun,
                               hess=hessfun,
                               args=(num_verts, Lw),
                               options=options,
                               method='trust-constr')

    # process output data (scaled to unit bounding box of MeshLab)
    out_vertex = res['x'][:num_verts * 3].reshape((3, num_verts))
    out_vertex = np.transpose(out_vertex, axes=(1, 0)) * 0.5

    # save output mesh
    dirname = os.path.dirname(filename)
    base, ext = os.path.splitext(os.path.basename(filename))
    base = base.split('_')[0]
    outfile = os.path.join(dirname, base + '_sphere_map' + ext)

    # scale
    mesh.vertices = out_vertex
    mesh.save(outfile)
    print('Mesh saved: %s' % (outfile))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Spherical embedding')
    parser.add_argument('-i', '--input', type=str, required=True, help='Input mesh file (*.obj)')
    parser.add_argument('-n', '--maxiter', type=int, default=1000, help='# of max iterations')
    args = parser.parse_args()

    main(args.input, args.maxiter)
