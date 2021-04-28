import csv
import open3d as o3d
import numpy as np
import torch
import copy
import torch.nn.functional as F
import torch.nn as nn
import datetime
import os
import sys
import glob

from torch.utils.tensorboard import SummaryWriter
from torch_geometric.data import Data
from torch_geometric.nn import GCNConv
from torch_scatter import scatter_max
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
        self.edge_index = data['edge_index']

    def check_graph(self, data):
        '''グラフ情報を表示'''
        print("グラフ構造:", data)
        print("グラフのキー: ", data.keys)
        print("ノード数:", data.num_nodes)
        print("エッジ数:", data.num_edges)
        print("ノードの特徴量数:", data.num_node_features)
        print("孤立したノードの有無:", data.contains_isolated_nodes())
        print("自己ループの有無:", data.contains_self_loops())
        print("====== ノードの特徴量:x ======")
        print(data['x'])
        print("====== ノードのクラス:y ======")
        print(data['y'])
        print("========= エッジ形状 =========")
        print(data['edge_index'])

class Mesh:
    def __init__(self, mesh):
        self.device = 'cpu'
        self.vs = self.mesh2ply(mesh)
        self.faces = np.asarray(mesh.triangles)
        #self.edges = self.edges_list(self.faces)
        self.build_gemm()
        
    def mesh2ply(self, mesh):
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(np.asarray(mesh.vertices))
        return np.asarray(pcd.points)
    
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

    def edges_list(self, faces):
        edges = []
        for v in faces:
            e1 = sorted([v[0], v[1]])
            e2 = sorted([v[1], v[2]])
            e3 = sorted([v[2], v[0]])
            if not e1 in edges:
                edges.append(e1)
            if not e2 in edges:
                edges.append(e2)
            if not e3 in edges:
                edges.append(e3)
            if not e1[::-1] in edges:
                edges.append(e1[::-1])
            if not e2[::-1] in edges:
                edges.append(e2[::-1])
            if not e3[::-1] in edges:
                edges.append(e3[::-1])
        edges = np.asarray(edges)
        return edges

argv = sys.argv
argc = len(argv)
if argc != 2:
    print('Usage: python gcnconv.py "input_dir"')
    quit()
file_path = argv[1]
file_list = glob.glob(file_path + '/*.obj')
file_list = sorted(file_list)

input_file = file_list[1]
label_file = file_list[0]
mesh_name = input_file.split('/')[1]
gt_file = 'datasets/groundtruth/' + mesh_name + '.obj'
gt_mesh = o3d.io.read_triangle_mesh(gt_file)

input_mesh = o3d.io.read_triangle_mesh(input_file)
label_mesh = o3d.io.read_triangle_mesh(label_file)

i_mesh = Mesh(input_mesh)
l_mesh = Mesh(label_mesh)

# ノードの特徴量
np.random.seed(42)
x = np.random.normal(size=(i_mesh.vs.shape[0], 16))
x = torch.tensor(x, dtype=torch.float, requires_grad=True)
x_pos = torch.tensor(i_mesh.vs, dtype=torch.float)
# ラベル
y = torch.tensor(l_mesh.vs, dtype=torch.float)

# エッジ
edge_index = torch.tensor(i_mesh.edges.T, dtype=torch.long)
edge_index = torch.cat([edge_index, edge_index[[1,0],:]], dim=1)

data = Data(x=x, y=y, edge_index=edge_index)

dataset = Dataset(data)
print(dataset.check_graph(data))


class Net(torch.nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        """
        # sharow
        hidden_size = [32, 64, 128, 256, 128, 64]
        self.conv1 = GCNConv(dataset.num_node_features, hidden_size[0])
        self.conv2 = GCNConv(hidden_size[0], hidden_size[1])
        self.conv3 = GCNConv(hidden_size[1], hidden_size[2])
        self.conv4 = GCNConv(hidden_size[2], hidden_size[3])
        self.conv5 = GCNConv(hidden_size[3], hidden_size[4])
        self.linear1 = torch.nn.Linear(hidden_size[4], hidden_size[5])
        self.linear2 = torch.nn.Linear(hidden_size[5], 3)
        self.bn1 = torch.nn.BatchNorm1d(hidden_size[0])
        self.bn2 = torch.nn.BatchNorm1d(hidden_size[1])
        self.bn3 = torch.nn.BatchNorm1d(hidden_size[2])
        self.bn4 = torch.nn.BatchNorm1d(hidden_size[3])
        self.bn5 = torch.nn.BatchNorm1d(hidden_size[4])
        self.l_relu = nn.LeakyReLU(0.01)
        """
        # deep gcn
        hidden_size = [32, 64, 128, 256, 256, 512, 512, 256, 256, 128, 64]
        self.conv1 = GCNConv(dataset.num_node_features, hidden_size[0])
        self.conv2 = GCNConv(hidden_size[0], hidden_size[1])
        self.conv3 = GCNConv(hidden_size[1], hidden_size[2])
        self.conv4 = GCNConv(hidden_size[2], hidden_size[3])
        self.conv5 = GCNConv(hidden_size[3], hidden_size[4])
        self.conv6 = GCNConv(hidden_size[4], hidden_size[5])
        self.conv7 = GCNConv(hidden_size[5], hidden_size[6])
        self.conv8 = GCNConv(hidden_size[6], hidden_size[7])
        self.conv9 = GCNConv(hidden_size[7], hidden_size[8])
        self.conv10 = GCNConv(hidden_size[8], hidden_size[9])
        self.linear1 = torch.nn.Linear(hidden_size[9], hidden_size[10])
        self.linear2 = torch.nn.Linear(hidden_size[10], 3)
        self.bn1 = torch.nn.BatchNorm1d(hidden_size[0])
        self.bn2 = torch.nn.BatchNorm1d(hidden_size[1])
        self.bn3 = torch.nn.BatchNorm1d(hidden_size[2])
        self.bn4 = torch.nn.BatchNorm1d(hidden_size[3])
        self.bn5 = torch.nn.BatchNorm1d(hidden_size[4])
        self.bn6 = torch.nn.BatchNorm1d(hidden_size[5])
        self.bn7 = torch.nn.BatchNorm1d(hidden_size[6])
        self.bn8 = torch.nn.BatchNorm1d(hidden_size[7])
        self.bn9 = torch.nn.BatchNorm1d(hidden_size[8])
        self.bn10 = torch.nn.BatchNorm1d(hidden_size[9])
        self.l_relu = nn.LeakyReLU(0.01)

    def forward(self, data):
        x, edge_index = dataset.x.to(device), dataset.edge_index.to(device)

        dx = self.conv1(x, edge_index)
        dx = self.bn1(dx)
        dx = self.l_relu(dx)

        dx = self.conv2(dx, edge_index)
        dx = self.bn2(dx)
        dx = self.l_relu(dx)

        dx = self.conv3(dx, edge_index)
        dx = self.bn3(dx)
        dx = self.l_relu(dx)

        dx = self.conv4(dx, edge_index)
        dx = self.bn4(dx)
        dx = self.l_relu(dx)

        dx = self.conv5(dx, edge_index)
        dx = self.bn5(dx)
        dx = self.l_relu(dx)

        dx = self.conv6(dx, edge_index)
        dx = self.bn6(dx)
        dx = self.l_relu(dx)

        dx = self.conv7(dx, edge_index)
        dx = self.bn7(dx)
        dx = self.l_relu(dx)

        dx = self.conv8(dx, edge_index)
        dx = self.bn8(dx)
        dx = self.l_relu(dx)

        dx = self.conv9(dx, edge_index)
        dx = self.bn9(dx)
        dx = self.l_relu(dx)

        dx = self.conv10(dx, edge_index)
        dx = self.bn10(dx)
        dx = self.l_relu(dx)

        # dx, _ = scatter_max(x, dataset.batch, dim=0)
        dx = self.linear1(dx)
        dx = self.l_relu(dx)
        dx = self.linear2(dx)

        return x_pos.to(device) + dx

def mae_loss(pred_pos, real_pos):
    """mean-absolute error for vertex positions"""
    # mean absolute difference for vertex positions
    diff_pos = torch.abs(real_pos - pred_pos)
    diff_pos = torch.sum(diff_pos.squeeze(), dim=1)
    mae_pos = torch.sum(diff_pos) / len(diff_pos)
    return mae_pos

def mse_loss(pred_pos, real_pos):
    """mean-square error for vertex positions"""
    diff_pos = torch.abs(real_pos - pred_pos)
    diff_pos = diff_pos ** 2
    diff_pos = torch.sum(diff_pos.squeeze(), dim=1)
    diff_pos = torch.sqrt(diff_pos)
    mse_loss = torch.sum(diff_pos) / len(diff_pos)
    return mse_loss

def mae_loss_edge_lengths(pred_pos, real_pos, edges):
    """mean-absolute error for edge lengths"""
    pred_edge_pos = pred_pos[edges,:].clone().detach()
    real_edge_pos = real_pos[edges,:].clone().detach()

    pred_edge_lens = torch.abs(pred_edge_pos[:,0,:]-pred_edge_pos[:,1,:])
    real_edge_lens = torch.abs(real_edge_pos[:,0,:]-real_edge_pos[:,1,:])

    pred_edge_lens = torch.sum(pred_edge_lens, dim=1)
    real_edge_lens = torch.sum(real_edge_lens, dim=1)
    
    diff_edge_lens = torch.abs(real_edge_lens - pred_edge_lens)
    mae_edge_lens = torch.mean(diff_edge_lens)

    return mae_edge_lens

def var_edge_lengths(pred_pos, edges):
    """variance of edge lengths"""
    pred_edge_pos = pred_pos[edges,:].clone().detach()

    pred_edge_lens = torch.abs(pred_edge_pos[:,0,:]-pred_edge_pos[:,1,:])

    pred_edge_lens = torch.sum(pred_edge_lens, dim=1)
    
    mean_edge_len = torch.mean(pred_edge_lens, dim=0, keepdim=True)
    var_edge_len = torch.pow(pred_edge_lens - mean_edge_len, 2.0)
    var_edge_len = torch.mean(var_edge_len)

    return var_edge_len

def mesh_laplacian_loss(pred_pos, ve, edges):
    """simple laplacian for output meshes"""
    pred_pos = pred_pos.T
    sub_mesh_vv = [edges[v_e, :].reshape(-1) for v_e in ve]
    sub_mesh_vv = [set(vv.tolist()).difference(set([i])) for i, vv in enumerate(sub_mesh_vv)]

    num_verts = pred_pos.size(1)
    mat_rows = [np.array([i] * len(vv), dtype=np.long) for i, vv in enumerate(sub_mesh_vv)]
    mat_rows = np.concatenate(mat_rows)
    mat_cols = [np.array(list(vv), dtype=np.long) for vv in sub_mesh_vv]
    mat_cols = np.concatenate(mat_cols)

    mat_rows = torch.from_numpy(mat_rows).long().to(pred_pos.device)
    mat_cols = torch.from_numpy(mat_cols).long().to(pred_pos.device)
    mat_vals = torch.ones_like(mat_rows).float()
    neig_mat = torch.sparse.FloatTensor(torch.stack([mat_rows, mat_cols], dim=0),
                                        mat_vals,
                                        size=torch.Size([num_verts, num_verts]))
    pred_pos = pred_pos.T
    sum_neigs = torch.sparse.mm(neig_mat, pred_pos)
    sum_count = torch.sparse.mm(neig_mat, torch.ones((num_verts, 1)).type_as(pred_pos))
    nnz_mask = (sum_count != 0).squeeze()
    lap_vals = sum_count[nnz_mask, :] * pred_pos[nnz_mask, :] - sum_neigs[nnz_mask, :]
    lap_vals = torch.sqrt(torch.sum(lap_vals * lap_vals, dim=1) + 1.0e-6)
    lap_loss = torch.sum(lap_vals) / torch.sum(nnz_mask)

    return lap_loss

def mad(mesh1, mesh2):
    mesh1.compute_triangle_normals()
    mesh2.compute_triangle_normals()

    normal1 = np.asarray(mesh1.triangle_normals)
    normal2 = np.asarray(mesh2.triangle_normals)

    inner = [np.inner(normal1[i], normal2[i]) for i in range(normal1.shape[0])]
    sad = np.rad2deg(np.arccos(np.clip(inner, -1.0, 1.0)))
    mad = np.sum(sad) / len(sad)

    return mad

init_mad = mad(label_mesh, gt_mesh)
print("initial_mad_value: ", init_mad)

# デバイス設定
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# モデルのインスタンス生成
model = Net().to(device)

# モデルを訓練モードに設定
model.train()
dataset.x = dataset.x.to(device)
dataset.y = dataset.y.to(device)
dataset.edge_index = dataset.edge_index.to(device)

# optimizer
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
loss_weight = [1.0, 0.0, 0.0, 0.2]

dt_now = datetime.datetime.now()
log_dir = "./logs/" + mesh_name + dt_now.isoformat()
writer = SummaryWriter(log_dir=log_dir)
os.mkdir("datasets/output/" + mesh_name + dt_now.isoformat())
log_file = log_dir + "/log.txt"
with open(log_file, mode="w") as f:
    f.write("Input: ")
    f.write(input_file)
    f.write("\n")
    f.write("Label: ")
    f.write(label_file)
    f.write("\n")
    f.write("Groundtruth: ")
    f.write(gt_file)
    f.write("\n")
    f.write(str(model))
    f.write("\n")
    f.write(str(optimizer))
    f.write("\n")
    f.write(str(loss_weight))
    f.write("\n")
    f.write(str(init_mad))


# learning loop
for epoch in range(1, 4001):
    optimizer.zero_grad()
    out = model(dataset)
    loss1 = mse_loss(out, dataset.y)
    loss4 = mesh_laplacian_loss(out, l_mesh.ve, l_mesh.edges)
    loss = loss_weight[0] * loss1 + loss_weight[3] * loss4
    loss.backward()
    optimizer.step()
    writer.add_scalar("total_loss", loss, epoch)
    writer.add_scalar("mse_loss", loss1, epoch)
    writer.add_scalar("laplacian_loss", loss4, epoch)
    if epoch % 10 == 0:
        print('Epoch %d | Loss: %.4f' % (epoch, loss.item()))
    if epoch % 50 == 0:
        input_mesh.vertices = o3d.utility.Vector3dVector(out.to('cpu').detach().numpy().copy())
        input_mesh.triangle_normals = o3d.utility.Vector3dVector([])
        o3d.io.write_triangle_mesh('datasets/output/' + mesh_name + dt_now.isoformat() + '/' + str(epoch) + '_output.obj', input_mesh)
        mad_value = mad(input_mesh, gt_mesh)
        print("mad_value: ", mad_value)
        writer.add_scalar("mean_angle_difference", mad_value, epoch)
writer.close()