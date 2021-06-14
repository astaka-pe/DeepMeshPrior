import numpy as np
import torch
from functools import reduce

class Dataset:
    def __init__(self, data):
        self.keys = data.keys
        self.num_nodes = data.num_nodes
        self.num_edges = data.num_edges
        self.num_node_features = data.num_node_features
        self.contains_isolated_nodes = data.contains_isolated_nodes()
        self.contains_self_loops = data.contains_self_loops()
        self.x = data['x']
        self.y = data['y']
        self.x_pos = data['x_pos']
        self.edge_index = data['edge_index']

class Mesh:
    def __init__(self, path):
        self.path = path
        self.vs, self.faces = self.fill_from_file(path)
        self.device = 'cpu'
        self.build_gemm() #self.edges, self.ve
    
    def fill_from_file(self, path):
        vs, faces = [], []
        f = open(path)
        for line in f:
            line = line.strip()
            splitted_line = line.split()
            if not splitted_line:
                continue
            elif splitted_line[0] == 'v':
                vs.append([float(v) for v in splitted_line[1:4]])
            elif splitted_line[0] == 'f':
                face_vertex_ids = [int(c.split('/')[0]) for c in splitted_line[1:]]
                assert len(face_vertex_ids) == 3
                face_vertex_ids = [(ind - 1) if (ind >= 0) else (len(vs) + ind) for ind in face_vertex_ids]
                faces.append(face_vertex_ids)
        f.close()
        vs = np.asarray(vs)
        faces = np.asarray(faces, dtype=int)

        assert np.logical_and(faces >= 0, faces < len(vs)).all()
        return vs, faces

    def build_gemm(self):
        self.ve = [[] for _ in self.vs]
        self.vei = [[] for _ in self.vs]
        edge_nb = []
        sides = []
        edge2key = dict()
        edges = []
        edges_count = 0
        nb_count = []
        for face_id, face in enumerate(self.faces):
            faces_edges = []
            for i in range(3):
                cur_edge = (face[i], face[(i + 1) % 3])
                faces_edges.append(cur_edge)
            for idx, edge in enumerate(faces_edges):
                edge = tuple(sorted(list(edge)))
                faces_edges[idx] = edge
                if edge not in edge2key:
                    edge2key[edge] = edges_count
                    edges.append(list(edge))
                    edge_nb.append([-1, -1, -1, -1])
                    sides.append([-1, -1, -1, -1])
                    self.ve[edge[0]].append(edges_count)
                    self.ve[edge[1]].append(edges_count)
                    self.vei[edge[0]].append(0)
                    self.vei[edge[1]].append(1)
                    nb_count.append(0)
                    edges_count += 1
            for idx, edge in enumerate(faces_edges):
                edge_key = edge2key[edge]
                edge_nb[edge_key][nb_count[edge_key]] = edge2key[faces_edges[(idx + 1) % 3]]
                edge_nb[edge_key][nb_count[edge_key] + 1] = edge2key[faces_edges[(idx + 2) % 3]]
                nb_count[edge_key] += 2
            for idx, edge in enumerate(faces_edges):
                edge_key = edge2key[edge]
                sides[edge_key][nb_count[edge_key] - 2] = nb_count[edge2key[faces_edges[(idx + 1) % 3]]] - 1
                sides[edge_key][nb_count[edge_key] - 1] = nb_count[edge2key[faces_edges[(idx + 2) % 3]]] - 2
        self.edges = np.array(edges, dtype=np.int32)
        self.gemm_edges = np.array(edge_nb, dtype=np.int64)
        self.sides = np.array(sides, dtype=np.int64)
        self.edges_count = edges_count
        # lots of DS for loss

        self.nvs, self.nvsi, self.nvsin, self.ve_in = [], [], [], []
        for i, e in enumerate(self.ve):
            self.nvs.append(len(e))
            self.nvsi += len(e) * [i]
            self.nvsin += list(range(len(e)))
            self.ve_in += e
        self.vei = reduce(lambda a, b: a + b, self.vei, [])
        self.vei = torch.from_numpy(np.array(self.vei).ravel()).to(self.device).long()
        self.nvsi = torch.from_numpy(np.array(self.nvsi).ravel()).to(self.device).long()
        self.nvsin = torch.from_numpy(np.array(self.nvsin).ravel()).to(self.device).long()
        self.ve_in = torch.from_numpy(np.array(self.ve_in).ravel()).to(self.device).long()

        self.max_nvs = max(self.nvs)
        self.nvs = torch.Tensor(self.nvs).to(self.device).float()
        self.edge2key = edge2key

    def compute_face_normals(self):
        face_normals = np.cross(self.vs[self.faces[:, 1]] - self.vs[self.faces[:, 0]], self.vs[self.faces[:, 2]] - self.vs[self.faces[:, 1]])
        norm = np.sqrt(np.sum(np.square(face_normals), 1))
        face_normals /= np.tile(norm, (3, 1)).T

        return face_normals