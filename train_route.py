import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init
import torch.optim as optim
from torch.optim import lr_scheduler
from route import ROUTENET
from route_data import ROUTEDATA
from utils import GeometryTransformer

batch_size = 16

dataset = ROUTEDATA()
dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)

model = ROUTENET(input_dim=9,hid_dim=64)
model = model.cuda()

lrate = 0.001
optimizer = optim.Adam(model.parameters(), lr=lrate)

for epoch in range(20):
    model.train()
    total_loss=0
    
    for j,data in enumerate(dataloader,0):
        optimizer.zero_grad()

        input_list,middle_list,frame_name,scene_name,sdf,scene_points,cam_extrinsic,s_grid_min,s_grid_max = data

        input_list = input_list[:,[0,-1],:]
                
        body = middle_list[:,0:1,6:16].cuda()
        input_list = torch.cat([input_list[:,:,:6],input_list[:,:,16:]],dim=2)
        middle_list = torch.cat([middle_list[:,:,:6],middle_list[:,:,16:]],dim=2)

        scene_points = scene_points.cuda()
        
        input_list = input_list.view(-1,62)
        six_d_input_list = GeometryTransformer.convert_to_6D_rot(input_list)
        six_d_input_list = six_d_input_list.view(-1,2,65)
        x = six_d_input_list.cuda()
        x1 = six_d_input_list[:,:,:9].cuda()
        
        middle_list = middle_list.view(-1,62)
        six_d_middle_list = GeometryTransformer.convert_to_6D_rot(middle_list)
        six_d_middle_list = six_d_middle_list.view(-1,60,65) #60: 2s 30fps

        y = six_d_middle_list[:,:,:9].cuda()
       
        out = model(x1,scene_points.transpose(1,2))

        loss = torch.mean(torch.abs(out-y))
        
        loss.backward()
        optimizer.step()

        total_loss = total_loss+loss

    print('loss:',total_loss/(j+1))
    
    save_path = './saved_model/route_'+str(epoch)+'.model'
    torch.save(model.state_dict(),save_path)
    
    