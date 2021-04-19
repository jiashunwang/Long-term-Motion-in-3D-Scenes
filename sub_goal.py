import numpy as np
import torch
import torchvision

import torch.nn as nn
import torch.nn.functional as F
import torchgeometry as tgm

import random
from torch.nn import init
import functools
from torch.optim import lr_scheduler

import sys, os
import json

from net_layers import BodyGlobalPoseVAE, BodyLocalPoseVAE, ResBlock

class PointNetfeat(nn.Module):
    def __init__(self, global_feat = True, feature_transform = False):
        super(PointNetfeat, self).__init__()
        #self.stn = STN3d()
        self.conv1 = torch.nn.Conv1d(3, 64, 1)
        self.conv2 = torch.nn.Conv1d(64, 128, 1)
        self.conv3 = torch.nn.Conv1d(128, 256, 1)
        self.bn1 = nn.InstanceNorm1d(64)
        self.bn2 = nn.InstanceNorm1d(128)
        self.bn3 = nn.InstanceNorm1d(256)
        self.global_feat = global_feat
        self.feature_transform = feature_transform
        #if self.feature_transform:
        #    self.fstn = STNkd(k=64)

    def forward(self, x):
        n_pts = x.size()[2]
       # trans = self.stn(x)
       # x = x.transpose(2, 1)
       # x = torch.bmm(x, trans)
       # x = x.transpose(2, 1)
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
        #self.feature_transform=feature_transform
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




################################################################################
## Conditional VAE of the human body, 
## Input: 72/75-dim, [T (3d vector), R (3d/6d), shape (10d), pose (32d), 
#         lefthand (12d), righthand (12d)]
## Note that, it requires pre-trained VPoser and latent variable of scene
################################################################################

class Pointnet(nn.Module):
    def __init__(self):
        super().__init__()
        
        self.conv1 = torch.nn.Conv1d(3, 64, 1)
        self.conv2 = torch.nn.Conv1d(64, 128, 1)
        self.conv3 = torch.nn.Conv1d(128, 256, 1)

        self.norm1 = torch.nn.InstanceNorm1d(64)
        self.norm2 = torch.nn.InstanceNorm1d(128)
        self.norm3 = torch.nn.InstanceNorm1d(256)

    def forward(self, x):
        
        x = F.relu(self.norm1(self.conv1(x)))
        
        x = F.relu(self.norm2(self.conv2(x)))
        x = F.relu(self.norm3(self.conv3(x)))
        x,_ = torch.max(x, 2)
        x = x.view(-1, 256)

        return x


class SUBGOAL(nn.Module):
    
    def __init__(self, 
                 latentD=512, 
                 n_dim_body=62,
                 scene_model_ckpt=True,
                 device='cuda'
                 ):
                 
        super(SUBGOAL, self).__init__()

        self.device=device
        self.eps_d = 32
        
        
        pointnet = PointNetDenseCls().to(self.device)
        if scene_model_ckpt is True:
            pointnet.load_state_dict(torch.load('0811_in_256_4.model'))
        removed=list(pointnet.children())[0:1]
        self.pointfeature=nn.Sequential(*removed)
        self.fc=nn.Linear(1024,256)
        self.condition_encoder=nn.Linear(256+9+10,latentD)#256+3+6+6

        self.linear_in = nn.Linear(n_dim_body, latentD)
        self.human_encoder = nn.Sequential(ResBlock(2*latentD),
                                           ResBlock(2*latentD))

        self.mu_enc = nn.Linear(2*latentD, self.eps_d)
        self.logvar_enc = nn.Linear(2*latentD, self.eps_d)

        self.linear_latent = nn.Linear(self.eps_d, latentD)
        self.human_decoder = nn.Sequential(ResBlock(2*latentD),
                                           ResBlock(2*latentD))

        self.linear_out = nn.Linear(2*latentD, n_dim_body)
        


    def _sampler(self, mu, logvar):
        var = logvar.mul(0.5).exp_()
        eps = torch.FloatTensor(var.size()).normal_()
        eps = eps.to(self.device)
        return eps.mul(var).add_(mu)


    def forward(self, x_body, x_s, z_loc, body):
        '''
        x_body: body representation, [batch, 65]
        x_s: point of scene, [batch, 3, N] 
        loc: target_loc, [batch, 3]
        body: body, [batch,10]
        '''
        
        b_ = x_s.shape[0]
        
        z_s = self.pointfeature(x_s) #z_s, [batch,1024]
        
        z_input = torch.cat([z_s,z_loc,body],dim=1)
        
        z_input = self.condition_encoder(z_input) #[batch,latentD]
        

        z_h = self.linear_in(x_body)

        z_ = torch.cat([z_h, z_input], dim=1)
        z_ = self.human_encoder(z_)

        mu = self.mu_enc(z_)
        logvar = self.logvar_enc(z_)

        z_ = self._sampler(mu, logvar)
        z_ = self.linear_latent(z_)
        z_hs = torch.cat([z_, z_input], dim=1)

        z_hs = self.human_decoder(z_hs)

        x_body_rec = self.linear_out(z_hs)


        return x_body_rec, mu, logvar
        
    

    def sample(self, x_body, x_s, loc, body):

        b_ = x_s.shape[0]

        z_s = self.pointfeature(x_s) #z_s, [batch,1024]

        z_loc = loc
        
        z_input=torch.cat([z_s,z_loc,body],dim=1)
        
        
        z_input=self.condition_encoder(z_input) #[batch,latentD]
        

        eps = torch.randn([b_, self.eps_d],dtype=torch.float32).to(self.device)
        
        z_h = self.linear_latent(eps)
        z_hs = torch.cat([z_h, z_input], dim=1)
        z_hs = self.human_decoder(z_hs)
        x_body_gen = self.linear_out(z_hs)

        return x_body_gen















