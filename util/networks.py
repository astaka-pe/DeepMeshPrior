import torch
import torch.nn as nn
import numpy as np
from torch_geometric.nn import GCNConv
from torch_scatter import scatter_max

class Net(nn.Module):
    def __init__(self, flags):
        super(Net, self).__init__()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.flags = flags
        h = [16, 32, 64, 128, 256, 256, 512, 512, 256, 256, 128, 64, 32, 32, 3]
        if self.flags:
            # skip net
            self.conv1  = GCNConv(h[0], h[1])
            self.conv2  = GCNConv(h[1], h[2])
            self.conv3  = GCNConv(h[2], h[3])
            self.conv4  = GCNConv(h[3], h[4])
            self.conv5  = GCNConv(h[4], h[5])
            self.conv6  = GCNConv(h[5], h[6])
            self.conv7  = GCNConv(h[6], h[7])
            self.conv8  = GCNConv(h[7]+h[6], h[8])
            self.conv9  = GCNConv(h[8]+h[5], h[9])
            self.conv10 = GCNConv(h[9]+h[4], h[10])
            self.conv11 = GCNConv(h[10]+h[3], h[11])
            self.conv12 = GCNConv(h[11]+h[2], h[12])
            self.conv13 = GCNConv(h[12]+h[1], h[13])
            self.linear1 = nn.Linear(h[13], h[14])
            
            self.bn1 = nn.BatchNorm1d(h[1])
            self.bn2 = nn.BatchNorm1d(h[2])
            self.bn3 = nn.BatchNorm1d(h[3])
            self.bn4 = nn.BatchNorm1d(h[4])
            self.bn5 = nn.BatchNorm1d(h[5])
            self.bn6 = nn.BatchNorm1d(h[6])
            self.bn7 = nn.BatchNorm1d(h[7])
            self.bn8 = nn.BatchNorm1d(h[8])
            self.bn9 = nn.BatchNorm1d(h[9])
            self.bn10 = nn.BatchNorm1d(h[10])
            self.bn11 = nn.BatchNorm1d(h[11])
            self.bn12 = nn.BatchNorm1d(h[12])
            self.bn13 = nn.BatchNorm1d(h[13])
            self.l_relu = nn.LeakyReLU()
        
        else:
            # normal net
            h = [16, 32, 64, 128, 256, 256, 512, 512, 256, 256, 128, 64, 32, 32, 16, 3]
            self.conv1  = GCNConv(h[0], h[1])
            self.conv2  = GCNConv(h[1], h[2])
            self.conv3  = GCNConv(h[2], h[3])
            self.conv4  = GCNConv(h[3], h[4])
            self.conv5  = GCNConv(h[4], h[5])
            self.conv6  = GCNConv(h[5], h[6])
            self.conv7  = GCNConv(h[6], h[7])
            self.conv8  = GCNConv(h[7], h[8])
            self.conv9  = GCNConv(h[8], h[9])
            self.conv10 = GCNConv(h[9], h[10])
            self.conv11 = GCNConv(h[10], h[11])
            self.conv12 = GCNConv(h[11], h[12])
            self.conv13 = GCNConv(h[12], h[13])
            self.linear1 = nn.Linear(h[13], h[14])
            self.linear2 = nn.Linear(h[14], h[15])
            
            self.bn1 = nn.BatchNorm1d(h[1])
            self.bn2 = nn.BatchNorm1d(h[2])
            self.bn3 = nn.BatchNorm1d(h[3])
            self.bn4 = nn.BatchNorm1d(h[4])
            self.bn5 = nn.BatchNorm1d(h[5])
            self.bn6 = nn.BatchNorm1d(h[6])
            self.bn7 = nn.BatchNorm1d(h[7])
            self.bn8 = nn.BatchNorm1d(h[8])
            self.bn9 = nn.BatchNorm1d(h[9])
            self.bn10 = nn.BatchNorm1d(h[10])
            self.bn11 = nn.BatchNorm1d(h[11])
            self.bn12 = nn.BatchNorm1d(h[12])
            self.bn13 = nn.BatchNorm1d(h[13])
            self.l_relu = nn.LeakyReLU()
            

    def forward(self, data):
        x = np.random.normal(0, 0.1, size=(data.x.shape[0], 16))
        x, edge_index, x_pos = data.x.to(self.device), data.edge_index.to(self.device), data.x_pos.to(self.device)
        
        if self.flags:
            # skip net
            dx = self.conv1(x, edge_index)
            dx = self.bn1(dx)
            dx = self.l_relu(dx)
            skip1 = dx

            dx = self.conv2(dx, edge_index)
            dx = self.bn2(dx)
            dx = self.l_relu(dx)
            skip2 = dx

            dx = self.conv3(dx, edge_index)
            dx = self.bn3(dx)
            dx = self.l_relu(dx)
            skip3 = dx

            dx = self.conv4(dx, edge_index)
            dx = self.bn4(dx)
            dx = self.l_relu(dx)
            skip4 = dx

            dx = self.conv5(dx, edge_index)
            dx = self.bn5(dx)
            dx = self.l_relu(dx)
            skip5 = dx
            
            dx = self.conv6(dx, edge_index)
            dx = self.bn6(dx)
            dx = self.l_relu(dx)
            skip6 = dx

            dx = self.conv7(dx, edge_index)
            dx = self.bn7(dx)
            dx = self.l_relu(dx)

            dx = torch.cat([dx, skip6], dim=1)
            dx = self.conv8(dx, edge_index)
            dx = self.bn8(dx)
            dx = self.l_relu(dx)

            dx = torch.cat([dx, skip5], dim=1)
            dx = self.conv9(dx, edge_index)
            dx = self.bn9(dx)
            dx = self.l_relu(dx)

            dx = torch.cat([dx, skip4], dim=1)
            dx = self.conv10(dx, edge_index)
            dx = self.bn10(dx)
            dx = self.l_relu(dx)
            
            dx = torch.cat([dx, skip3], dim=1)
            dx = self.conv11(dx, edge_index)
            dx = self.bn11(dx)
            dx = self.l_relu(dx)

            dx = torch.cat([dx, skip2], dim=1)
            dx = self.conv12(dx, edge_index)
            dx = self.bn12(dx)
            dx = self.l_relu(dx)

            dx = torch.cat([dx, skip1], dim=1)
            dx = self.conv13(dx, edge_index)
            dx = self.bn13(dx)
            dx = self.l_relu(dx)
            dx = self.linear1(dx)
        else:
            # normal net
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

            dx = self.conv11(dx, edge_index)
            dx = self.bn11(dx)
            dx = self.l_relu(dx)
            
            dx = self.conv12(dx, edge_index)
            dx = self.bn12(dx)
            dx = self.l_relu(dx)
            
            dx = self.conv13(dx, edge_index)
            dx = self.bn13(dx)
            dx = self.l_relu(dx)
            
            dx = self.linear1(dx)
            dx = self.l_relu(dx)
            dx = self.linear2(dx)
        
        return x_pos + dx