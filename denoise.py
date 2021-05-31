import numpy as np
import torch
import copy
import datetime
import os
import sys
import glob
import argparse
import json
from util.objmesh import ObjMesh
from util.models import Dataset, Mesh
from util.networks import Net
import util.loss as Loss

from torch.utils.tensorboard import SummaryWriter
from torch_geometric.data import Data

parser = argparse.ArgumentParser(description='Deep mesh prior for denoising')
parser.add_argument('-i', '--input', type=str, required=True)
parser.add_argument('--lr', type=float, default=0.01)
parser.add_argument('--iter', type=int, default=5000)
parser.add_argument('--skip', type=bool, default=False)
parser.add_argument('--lap', type=float, default=3.0)
FLAGS = parser.parse_args()

for k, v in vars(FLAGS).items():
    print('{:10s}: {}'.format(k, v))

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

file_path = FLAGS.input
file_list = glob.glob(file_path + '/*.obj')
file_list = sorted(file_list)
input_file = file_list[1]
label_file = file_list[0]
mesh_name = input_file.split('/')[-2]
gt_file = 'datasets/groundtruth/' + mesh_name + '.obj'

l_mesh = Mesh(label_file)
i_mesh = Mesh(input_file)
g_mesh = Mesh(gt_file)

# node-features and edge-index
np.random.seed(42)
x = np.random.normal(size=(l_mesh.vs.shape[0], 16))
x = torch.tensor(x, dtype=torch.float, requires_grad=True)
x_pos = torch.tensor(i_mesh.vs, dtype=torch.float)
y = torch.tensor(l_mesh.vs, dtype=torch.float)

edge_index = torch.tensor(l_mesh.edges.T, dtype=torch.long)
edge_index = torch.cat([edge_index, edge_index[[1,0],:]], dim=1)
init_mad = Loss.mad(l_mesh, g_mesh)

data = Data(x=x, y=y, x_pos=x_pos, edge_index=edge_index)
dataset = Dataset(data)
dataset.x = dataset.x.to(device)
dataset.y = dataset.y.to(device)
dataset.edge_index = dataset.edge_index.to(device)
print(dataset.check_graph(data))

# create model instance
model = Net(FLAGS.skip).to(device)
model.train()

# output experimental conditions
dt_now = datetime.datetime.now()
log_dir = "./logs/denoise/" + mesh_name + dt_now.isoformat()
writer = SummaryWriter(log_dir=log_dir)
out_dir = "./datasets/d_output/" + mesh_name + dt_now.isoformat()
os.mkdir(out_dir)
log_file = out_dir + "/condition.json"
condition = {"input":input_file, "label":label_file, "gt": gt_file, "iter": FLAGS.iter ,"lap": FLAGS.lap, "skip": FLAGS.skip, "init_mad": init_mad, "lr": FLAGS.lr}

with open(log_file, mode="w") as f:
    l = json.dumps(condition, indent=2)
    f.write(l)

# learning loop
min_mad = 1000
print("initial_mad_value: ", init_mad)

optimizer = torch.optim.Adam(model.parameters(), lr=FLAGS.lr)

for epoch in range(1, 10001):
    optimizer.zero_grad()
    out = model(dataset)
    loss1 = Loss.mse_loss(out, dataset.y)
    loss2 = Loss.mesh_laplacian_loss(out, l_mesh.ve, l_mesh.edges)
    loss = loss1 + FLAGS.lap * loss2
    loss.backward()
    optimizer.step()
    writer.add_scalar("total_loss", loss, epoch)
    writer.add_scalar("mse_loss", loss1, epoch)
    writer.add_scalar("laplacian_loss", loss2, epoch)
    if epoch % 10 == 0:
        print('Epoch %d | Loss: %.4f' % (epoch, loss.item()))
    if epoch % 50 == 0:
        o_mesh = ObjMesh(input_file)
        o_mesh.vs = o_mesh.vertices = out.to('cpu').detach().numpy().copy()
        o_mesh.faces = i_mesh.faces
        o_mesh.save(out_dir + '/' + str(epoch) + '_output.obj')
        mad_value = Loss.mad(o_mesh, g_mesh)
        min_mad = min(mad_value, min_mad)
        print("mad_value: ", mad_value, "min_mad: ", min_mad)
        writer.add_scalar("mean_angle_difference", mad_value, epoch)

writer.close()