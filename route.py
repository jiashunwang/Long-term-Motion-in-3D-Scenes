import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

class PointNetfeat(nn.Module):
    def __init__(self, global_feat = True, feature_transform = False):
        super(PointNetfeat, self).__init__()

        self.conv1 = torch.nn.Conv1d(3, 64, 1)
        self.conv2 = torch.nn.Conv1d(64, 128, 1)
        self.conv3 = torch.nn.Conv1d(128, 256, 1)
        self.bn1 = nn.InstanceNorm1d(64)
        self.bn2 = nn.InstanceNorm1d(128)
        self.bn3 = nn.InstanceNorm1d(256)
        self.global_feat = global_feat
        self.feature_transform = feature_transform


    def forward(self, x):
        n_pts = x.size()[2]

        x = F.relu(self.bn1(self.conv1(x)))
        pointfeat = x
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.bn3(self.conv3(x))
        x = torch.max(x, 2, keepdim=True)[0]
        x = x.view(-1, 256)

        return x
        if self.global_feat:
            return x, trans, trans_feat
        else:
            x = x.view(-1, 256, 1).repeat(1, 1, n_pts)
            return torch.cat([x, pointfeat], 1)#, trans, trans_feat




class PointNetDenseCls(nn.Module):
    def __init__(self, k = 13, feature_transform=False):
        super(PointNetDenseCls, self).__init__()
        self.k = k
        
        self.feat = PointNetfeat(global_feat=False, feature_transform=feature_transform)
        self.conv1 = torch.nn.Conv1d(256+64, 256, 1)
        self.conv2 = torch.nn.Conv1d(256, 128, 1)
        self.conv3 = torch.nn.Conv1d(128, 64, 1)
        self.conv4 = torch.nn.Conv1d(64, self.k, 1)
        self.bn1 = nn.BatchNorm1d(256)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(64)

    def forward(self, x):
        batchsize = x.size()[0]
        n_pts = x.size()[2]
        x = self.feat(x)
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = self.conv4(x)
        x = x.transpose(2,1).contiguous()
        x = F.log_softmax(x.view(-1,self.k), dim=-1)
        x = x.view(batchsize, n_pts, self.k)
        return x



class ROUTENET(nn.Module):
    def __init__(self, input_dim=9, hid_dim=64, n_layers=1, dropout=0.4,bidirectional=True,scene_model_ckpt=True,device='cuda'):
        super().__init__()
        self.input_dim = input_dim
        self.hid_dim = hid_dim
        self.n_layers = n_layers
        
        
        self.lstm = nn.LSTM(input_dim, hid_dim, n_layers, dropout=dropout,bidirectional=bidirectional,batch_first=True)
        self.fc_scene = nn.Linear(256,32)
        self.fc = nn.Linear(hid_dim*2*2+32,hid_dim*2*2)
        self.fc2 = nn.Linear(hid_dim*2*2,60*input_dim)

        pointnet = PointNetDenseCls().to(device)#.cuda()
        if scene_model_ckpt is True:
            pointnet.load_state_dict(torch.load('0811_in_256_4.model'))
        removed = list(pointnet.children())[0:1]
        self.pointfeature = nn.Sequential(*removed)

    def forward(self, x,scene_points):
        
        batch_size = x.shape[0]    
        
        outputs, (hidden, cell) = self.lstm(x)
        
        outputs = outputs.reshape(batch_size,-1)
        
        pointfea = self.pointfeature(scene_points)#.detach()
        
        pointfea = self.fc_scene(pointfea)
        outputs  = torch.cat([outputs,pointfea],dim=1)
        
        outputs = self.fc(outputs)
        outputs = self.fc2(outputs)
        
        outputs = outputs.reshape(batch_size,60,self.input_dim)

        return outputs