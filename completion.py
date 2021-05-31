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

parser = argparse.ArgumentParser(description='Deep mesh prior for completion')
parser.add_argument('-i', '--input', type=str, required=True)
parser.add_argument('--lr', type=float, default=0.01)
parser.add_argument('--iter', type=int, default=5000)
parser.add_argument('--skip', type=bool, default=False)
parser.add_argument('--lap', type=float, default=0.3)
FLAGS = parser.parse_args()

for k, v in vars(FLAGS).items():
    print('{:10s}: {}'.format(k, v))

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

file_path = FLAGS.input
file_list = glob.glob(file_path + '/*.obj')
file_list = sorted(file_list)
vert_list = glob.glob(file_path + '/*.txt')
input_file = file_list[1] # xx_init.obj
label_file = file_list[0] # xx.obj
mesh_name = input_file.split('/')[-2]
vert_list_file = vert_list[0] # xx.txt

with open (vert_list_file, 'r') as f:
    new_vert_list = f.read().split("\n")
    new_vert_list = [l.strip() for l in new_vert_list]
    new_vert_list = [l for l in new_vert_list if len(l) != 0]
new_vert_list = [int(i) for i in new_vert_list]

i_mesh = Mesh(input_file)
l_mesh = Mesh(label_file)

# node-features and edge-index
np.random.seed(42)
x = np.random.normal(size=(i_mesh.vs.shape[0], 16))
x = torch.tensor(x, dtype=torch.float, requires_grad=True)
x_pos = torch.tensor(i_mesh.vs, dtype=torch.float)
y = torch.tensor(l_mesh.vs, dtype=torch.float)

edge_index = torch.tensor(i_mesh.edges.T, dtype=torch.long)
edge_index = torch.cat([edge_index, edge_index[[1,0],:]], dim=1)

data = Data(x=x, y=y, x_pos=x_pos, edge_index=edge_index)
dataset = Dataset(data)
dataset.x = dataset.x.to(device)
dataset.y = dataset.y.to(device)
dataset.edge_index = dataset.edge_index.to(device)
verts_mask = torch.ones(len(dataset.x))
verts_mask[new_vert_list] = 0
verts_mask = verts_mask.to(device)
print(dataset.check_graph(data))

# create model instance
model = Net(FLAGS.skip).to(device)
model.train()

# output experimental conditions
dt_now = datetime.datetime.now()
log_dir = "./logs/completion/" + mesh_name + dt_now.isoformat()
writer = SummaryWriter(log_dir=log_dir)
out_dir = "./datasets/c_output/" + mesh_name + dt_now.isoformat()
os.mkdir(out_dir)
log_file = out_dir + "/condition.json"
condition = {"input":input_file, "label":label_file, "inserted_vs": vert_list_file, "iter": FLAGS.iter, "lap": FLAGS.lap, "skip": FLAGS.skip, "lr": FLAGS.lr}

with open(log_file, mode="w") as f:
    l = json.dumps(condition, indent=2)
    f.write(l)

# learning loop
optimizer = torch.optim.Adam(model.parameters(), lr=FLAGS.lr)

for epoch in range(1, FLAGS.iter+1):
    optimizer.zero_grad()
    out = model(dataset)
    loss1 = Loss.mse_loss(out, dataset.y, verts_mask)
    loss2 = Loss.mesh_laplacian_loss(out, l_mesh.ve, l_mesh.edges)
    loss = loss1 + FLAGS.lap * loss2
    loss.backward()
    optimizer.step()
    writer.add_scalar("total_loss", loss, epoch)
    writer.add_scalar("mse_loss_vert", loss1, epoch)
    writer.add_scalar("laplacian_loss", loss2, epoch)
    if epoch % 10 == 0:
        print('Epoch %d | Loss: %.4f' % (epoch, loss.item()))
    if epoch % 100 == 0:
        o_mesh = ObjMesh(input_file)
        o_mesh.vs = o_mesh.vertices = out.to('cpu').detach().numpy().copy()
        o_mesh.faces = i_mesh.faces
        o_mesh.save(out_dir + '/' + str(epoch) + '_output.obj')

writer.close()