import sys, os, glob
import json
import argparse
import numpy as np


import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init
import torch.optim as optim
from torch.optim import lr_scheduler

import smplx
from human_body_prior.tools.model_loader import load_vposer


from utils import GeometryTransformer, BodyParamParser
import time

import os
import pickle

import random
import chamfer_pytorch.dist_chamfer as ext
from torch.autograd import Variable



def cal_loss(xhr, xhr_rec, body, cam_extrinsic, scene_points,
            sdf, s_grid_min_batch, s_grid_max_batch, input_list, iteration, vposer, body_mesh_model_input, body_mesh_model_batch, threshold, contact_id_folder, device):
    
    contact_part=['back','gluteus','L_Hand','R_Hand','L_Leg','R_Leg','thighs']
    #xhr_rec=torch.cat([xhr_rec,xhr[:,6:]],dim=1)
    ### reconstruction loss
    loss_rec = 0.1*F.l1_loss(xhr[:,:], xhr_rec[:,:])#+0.5*F.l1_loss(xhr[:,6:48], xhr_rec[:,6:48])
    loss_rec = loss_rec#+0.1*F.l1_loss(xhr[xhr_rec.shape[0]//2,:6], xhr_rec[xhr_rec.shape[0]//2,:6])\
                #+0.2*F.l1_loss(input_list[0], xhr_rec[0])+0.2*F.l1_loss(input_list[1], xhr_rec[-1])
    
    ### vposer loss
    vposer_pose = xhr_rec[:,6:38]
    loss_vposer = torch.mean(vposer_pose**2)

    ### contact loss
    
    body_param_rec = BodyParamParser.body_params_encapsulate_batch_nobody_hand(xhr_rec)
    #print(xhr_rec.shape)
    #print(body_param_rec['body_pose'].shape)
    body_param_rec['body_pose'] = vposer.decode(body_param_rec['body_pose'], 
                                    output_type='aa').view(xhr_rec.shape[0], -1)
    #print(body.shape)
    #print(xhr_rec.shape)
    body_param_rec['betas']=body.repeat(xhr_rec.shape[0],1)
    #print(body_param_rec['body_pose_vp'].shape)
    
    #temp_out = body_mesh_model()

    smplx_output = body_mesh_model_batch(return_verts=True, 
                                        **body_param_rec
                                        )
    
    body_verts_batch = smplx_output.vertices #[b, 10475,3]
    body_joints_batch = smplx_output.joints
    #print('joints',body_joints_batch[:,7:9,:].shape)
    loss_motion=0
    
    left_foot_vid, left_foot_fid = GeometryTransformer.get_contact_id(body_segments_folder=contact_id_folder,
                            contact_body_parts=['L_Leg'])
    right_foot_vid, right_foot_fid = GeometryTransformer.get_contact_id(body_segments_folder=contact_id_folder,
                            contact_body_parts=['R_Leg'])
    gluteus_vid, gluteus_fid = GeometryTransformer.get_contact_id(body_segments_folder=contact_id_folder,
                            contact_body_parts=['gluteus'])
    
    left_foot_verts_batch = body_verts_batch[:, left_foot_vid, :]
    right_foot_verts_batch = body_verts_batch[:, right_foot_vid, :]
    butt_verts_batch = body_verts_batch[:,gluteus_vid,:]
    
    #print(body_verts_batch.shape)
    if iteration<20:
        foot_motion=False
    else:
        foot_motion=False
    
    loss_skating=0
    
    
    #temp_list_=[]
    #temp_list2=[]
    
    #global left_foot_scatter
    #left_foot_scatter=[]
    #global right_foot_scatter
    #right_foot_scatter=[]
    
    if iteration==0:
        global highest_joints
        highest_joints=torch.min(body_joints_batch[:,7:9,1]).detach()
        global temp_list
        temp_list=[]

        #global foot_list
        foot_list=[]
        for iii in range(1,body_verts_batch.shape[0]-1,1):
        
        
            
            
            '''
            butt_verts_avg=torch.tensor([torch.mean(butt_verts_batch[iii][:,:1]),torch.mean(butt_verts_batch[iii][:,1:2]),
                                            torch.mean(butt_verts_batch[iii][:,2:3])])
            left_foot_avg=torch.tensor([torch.mean(left_foot_verts_batch[iii][:,:1]),torch.mean(left_foot_verts_batch[iii][:,1:2]),
                                            torch.mean(left_foot_verts_batch[iii][:,2:3])])
            right_foot_avg=torch.tensor([torch.mean(right_foot_verts_batch[iii][:,:1]),torch.mean(right_foot_verts_batch[iii][:,1:2]),
                                            torch.mean(right_foot_verts_batch[iii][:,2:3])])

            next_left_foot_avg=torch.tensor([torch.mean(left_foot_verts_batch[iii+1][:,:1]),torch.mean(left_foot_verts_batch[iii+1][:,1:2]),
                                            torch.mean(left_foot_verts_batch[iii+1][:,2:3])])

            next_right_foot_avg=torch.tensor([torch.mean(right_foot_verts_batch[iii+1][:,:1]),torch.mean(right_foot_verts_batch[iii+1][:,1:2]),
                                            torch.mean(right_foot_verts_batch[iii+1][:,2:3])])
            '''
            left_foot_joint=body_joints_batch[iii,7,:]
            right_foot_joint=body_joints_batch[iii,8,:]

            left_knee_joint=body_joints_batch[iii,4,:]
            right_knee_joint=body_joints_batch[iii,5,:]

            #left_foot_scatter.append(left_foot_joint.detach().numpy())
            #right_foot_scatter.append(right_foot_joint.detach().numpy())


            next_left_foot_joint=body_joints_batch[iii+1,7,:]
            next_right_foot_joint=body_joints_batch[iii+1,8,:]

            last_left_foot_joint=body_joints_batch[iii-1,7,:]
            last_right_foot_joint=body_joints_batch[iii-1,8,:]

            left_joint_distance=torch.mean((next_left_foot_joint-left_foot_joint)**2)
            right_joint_distance=torch.mean((next_right_foot_joint-right_foot_joint)**2)

            left_joint_longer_distance=torch.mean((next_left_foot_joint-last_left_foot_joint)**2)
            right_joint_longer_distance=torch.mean((next_right_foot_joint-last_right_foot_joint)**2)


            #left_avg_low=left_foot_avg[1:2]
            #right_avg_low=right_foot_avg[1:2]
            #if left_avg_distance>right_avg_distance:
            #    temp_list.append('left')
                #foot_list.append(left_foot_verts_batch[iii])
            #else:
            #    temp_list.append('right')
                #foot_list.append(right_foot_verts_batch[iii])
            #threshold=threshold
            if threshold*right_joint_distance<left_joint_distance and threshold*right_joint_longer_distance<left_joint_longer_distance:
            #if body_joints_batch[iii,5,1]>1.1*body_joints_batch[iii,4,1]:
                temp_list.append('right')
                #foot_list.append(right_foot_verts_batch[iii])
            elif right_joint_distance>threshold*left_joint_distance and right_joint_longer_distance>threshold*left_joint_longer_distance:
            #elif 1.1*body_joints_batch[iii,5,1]<body_joints_batch[iii,4,1]:
                temp_list.append('left')
                #foot_list.append(left_foot_verts_batch[iii])
            else:
                temp_list.append('unknown')
                #foot_list.append(None)
#        #print(temp_list)
        for jj in range(len(temp_list)):
            if jj==0 and temp_list[jj]=='unknown':
                for jjj in range(jj+1,len(temp_list)):
                    if temp_list[jjj]!='unknown':
                        for jjjj in range(jj,jjj):
                            temp_list[jjjj]=temp_list[jjj]
                        break
            if jj==len(temp_list)-1 and temp_list[jj]=='unknown':
                for jjj in range(jj-1,0,-1):
                    if temp_list[jjj]!='unknown':
                        for jjjj in range(jj,jjj,-1):
                            temp_list[jjjj]=temp_list[jjj]
                        break
            if temp_list[jj]!='unknown':
                for jjj in range(jj+1,len(temp_list)):

                    if temp_list[jjj]=='unknown':
                        continue
                    elif temp_list[jjj]==temp_list[jj]:
                        for jjjj in range(jj,jjj+1):
                            temp_list[jjjj]=temp_list[jj]
                        break
                    else:
                        for jjjj in range(jj,jjj+1):
                            #temp_list[jjjj]=temp_list[jj]
                            if jjjj<int((jj+jjj)/2):
                                temp_list[jjjj]=temp_list[jj]
                            else:
                                temp_list[jjjj]=temp_list[jjj]
                        break
        temp_list.insert(0,temp_list[0])
        temp_list.append(temp_list[-1])
        print(temp_list)

        for kkk in range(len(temp_list)):
            if temp_list[kkk] == 'left':
                foot_list.append(left_foot_verts_batch[kkk])
            if temp_list[kkk] =='right':
                foot_list.append(right_foot_verts_batch[kkk])
        #print('foot_list',foot_list.shape)
        global temp_list2
        temp_list2=[]
        global avg_foot_list
        avg_foot_list=[]
        start=0
        for jj in range(len(temp_list)-1):
            if temp_list[jj]==temp_list[jj+1]:

                pass#continue
            if temp_list[jj]!=temp_list[jj+1]:
                temp_list2.append([jj,temp_list[jj]])
                end=jj
                part_avg_foot_list=foot_list[start:end+1]
                part_avg_foot=sum(part_avg_foot_list)/(len(part_avg_foot_list))

                for each in range(len(part_avg_foot_list)):
                    avg_foot_list.append(part_avg_foot.detach())
                start=jj+1
        if start<len(temp_list):
            part_avg_foot_list=foot_list[start:]
            part_avg_foot=sum(part_avg_foot_list)/(len(part_avg_foot_list))
            #print(part_avg_foot.shape)

            for each in range(len(part_avg_foot_list)):
                    avg_foot_list.append(part_avg_foot.detach())
        global midlist
        midlist=[]
        for q in range(len(temp_list2)):
            if q==0:
                temp_highest=torch.min(body_joints_batch[:temp_list2[q][0],7:9,1]).detach()
                midlist.append([int(0.5*temp_list2[q][0]),temp_list2[q][1],temp_highest])
                if len(temp_list2)==1:
                    temp_highest=torch.min(body_joints_batch[temp_list2[q][0]:,7:9,1]).detach()
                    midlist.append([int(0.5*(temp_list2[q][0]+len(avg_foot_list))),temp_list[temp_list2[q][0]+1],temp_highest])
                    
                else:
                    temp_highest=torch.min(body_joints_batch[temp_list2[q][0]:temp_list2[q+1][0],7:9,1]).detach()
                    midlist.append([int(0.5*(temp_list2[q][0]+temp_list2[q+1][0])),temp_list2[q+1][1],temp_highest])
                
            elif q==len(temp_list2)-1:
                temp_highest=torch.min(body_joints_batch[temp_list2[q][0]:,7:9,1]).detach()
                midlist.append([int(0.5*(temp_list2[q][0]+len(avg_foot_list))),temp_list2[q-1][1],temp_highest])
            else:
                temp_highest=torch.min(body_joints_batch[temp_list2[q][0]:temp_list2[q+1][0],7:9,1]).detach()
                midlist.append([int(0.5*(temp_list2[q][0]+temp_list2[q+1][0])),temp_list2[q+1][1],temp_highest])
            
            
    
    #print(temp_list)
    #print(len(temp_list))
    #print(temp_list2)
    #temp_list_.append('unknown')
    #for i in range(1,len(temp_list)-1,1):
    #    if temp_list[i]==temp_list[i-1] and temp_list[i]==temp_list[i+1]:
    #        temp_list_.append(temp_list[i])
    #    else:
    #        temp_list_.append('unknown')
    #print(temp_list_)
    #print(foot_list)
    '''
    if iteration==0:
        global temp_list2
        temp_list2=[]
        global avg_foot_list
        avg_foot_list=[]
        start=0
        for jj in range(len(temp_list)-1):
            if temp_list[jj]==temp_list[jj+1]:

                pass#continue
            if temp_list[jj]!=temp_list[jj+1]:
                temp_list2.append([jj,temp_list[jj]])
                end=jj
                part_avg_foot_list=foot_list[start:end+1]
                part_avg_foot=sum(part_avg_foot_list)/(len(part_avg_foot_list))

                for each in range(len(part_avg_foot_list)):
                    avg_foot_list.append(part_avg_foot)
                start=jj+1
        if start<len(temp_list):
            part_avg_foot_list=foot_list[start:]
            part_avg_foot=sum(part_avg_foot_list)/(len(part_avg_foot_list))
            #print(part_avg_foot.shape)

            for each in range(len(part_avg_foot_list)):
                    avg_foot_list.append(part_avg_foot)
    #print(len(avg_foot_list))
    #print(len(foot_list))
    '''
    
    #print(avg_foot_list[0])
    foot_list=[]
    
    for kkk in range(len(temp_list)):
        if temp_list[kkk] == 'left':
            foot_list.append(left_foot_verts_batch[kkk])
        if temp_list[kkk] =='right':
            foot_list.append(right_foot_verts_batch[kkk])
#    print(loss_skating)
    
    for kk in range(len(foot_list)):
        temp_loss=torch.mean((foot_list[kk]-avg_foot_list[kk])**2)
        loss_skating=loss_skating+temp_loss
#    print(loss_skating)
#    print('temp_list2:',temp_list2)
    #print(avg_foot_list)
    for qq in range(len(temp_list2)):
        each=temp_list2[qq]
        if each[0]<len(right_foot_verts_batch)-1:
            if each[1]=='left':
                #temp_loss_left=torch.mean((avg_foot_list[each[0]]-left_foot_verts_batch[each[0]+1])**2)\
                #           +torch.mean((avg_foot_list[each[0]]-left_foot_verts_batch[each[0]])**2)
                
                temp_loss_=torch.mean((right_foot_verts_batch[each[0]]-avg_foot_list[each[0]+1])**2)
                                        #torch.mean((right_foot_verts_batch[each[0]]-avg_foot_list[each[0]+1])**2)
                
            if each[1]=='right':
                temp_loss_=torch.mean((left_foot_verts_batch[each[0]]-avg_foot_list[each[0]+1])**2)
                #temp_loss_right=torch.mean((avg_foot_list[each[0]]-right_foot_verts_batch[each[0]+1])**2)\
                #                    +torch.mean((avg_foot_list[each[0]]-right_foot_verts_batch[each[0]])**2)
                                    
                
                
            #print(temp_loss_left)
            #print(temp_loss_)
            loss_skating=loss_skating+temp_loss_
    loss_height=0
    for each in midlist:
        if each[0]<len(right_foot_verts_batch)-1:
            if each[1]=='left':
            
                loss_height=loss_height+(body_joints_batch[each[0],8,1]-each[2])**2
            if each[1]=='right':
                
                loss_height=loss_height+(body_joints_batch[each[0],7,1]-each[2])**2
#    print('height:',loss_height)
    loss_skating=loss_skating+0.05*loss_height
    

    #loss_contact
    
    body_param_rec_input = BodyParamParser.body_params_encapsulate_batch_nobody_hand(input_list)
    #print('2:',body_param_rec_input['body_pose_vp'].shape)
    
    body_param_rec_input['body_pose'] = vposer.decode(body_param_rec_input['body_pose'], output_type='aa').view(input_list.shape[0], -1)
    #print('3:',joint_rot_batch.shape)
    body_param_rec_input['betas']=body.repeat(1,1)
    
    #body_param_rec_input['betas']=body.repeat(input_list.shape[0])
    smplx_input = body_mesh_model_input(return_verts=True, 
                                    **body_param_rec_input
                                    )
    body_verts_batch_input = smplx_input.vertices
    #print('4:',body_verts_batch_input.shape)
    #print()
    left_foot_verts_batch_input = body_verts_batch_input[:, left_foot_vid, :]
    right_foot_verts_batch_input = body_verts_batch_input[:, right_foot_vid, :]

    left_start=torch.mean((left_foot_verts_batch[0]-left_foot_verts_batch_input[0])**2)
    right_start=torch.mean((right_foot_verts_batch[0]-right_foot_verts_batch_input[0])**2)

    left_end=torch.mean((left_foot_verts_batch[-1]-left_foot_verts_batch_input[1])**2)
    right_end=torch.mean((right_foot_verts_batch[-1]-right_foot_verts_batch_input[1])**2)

    
    for iii in range(body_verts_batch.shape[0]-1):
        '''
        butt_verts_avg=torch.tensor([torch.mean(butt_verts_batch[iii][:,:1]),torch.mean(butt_verts_batch[iii][:,1:2]),
                                        torch.mean(butt_verts_batch[iii][:,2:3])])
        left_foot_avg=torch.tensor([torch.mean(left_foot_verts_batch[iii][:,:1]),torch.mean(left_foot_verts_batch[iii][:,1:2]),
                                        torch.mean(left_foot_verts_batch[iii][:,2:3])])
        right_foot_avg=torch.tensor([torch.mean(right_foot_verts_batch[iii][:,:1]),torch.mean(right_foot_verts_batch[iii][:,1:2]),
                                        torch.mean(right_foot_verts_batch[iii][:,2:3])])
        '''
        left_foot_loss=torch.mean((left_foot_verts_batch[iii]-left_foot_verts_batch[iii+1])**2)
        right_foot_loss=torch.mean((right_foot_verts_batch[iii]-right_foot_verts_batch[iii+1])**2)
        loss_motion=loss_motion+(left_foot_loss+right_foot_loss)

        loss_motion=loss_motion+torch.mean((body_verts_batch[iii]-body_verts_batch[iii+1])**2)
    
    loss_start_end=torch.mean((body_verts_batch[0]-body_verts_batch_input[0])**2)+torch.mean((body_verts_batch[-1]-body_verts_batch_input[1])**2)
    loss_motion=loss_motion+loss_start_end+min(left_end,right_end)+min(left_start,right_start)
    #/+5*min(left_start,right_start)+5*min(left_end,right_end)
    loss_contact=0
    
    #body_verts_batch__ = GeometryTransformer.verts_transform(body_verts_batch, torch.tensor(cam_extrinsic,dtype=torch.float32))#.cuda())
    body_verts_batch__ = GeometryTransformer.verts_transform(body_verts_batch, cam_extrinsic)
    vid, fid = GeometryTransformer.get_contact_id(body_segments_folder=contact_id_folder,
                            contact_body_parts=contact_part)
    foot_vid, foot_fid = GeometryTransformer.get_contact_id(body_segments_folder=contact_id_folder,
                            contact_body_parts=['L_Leg','R_Leg'])

    body_verts_contact_batch = body_verts_batch[:, vid, :]
    
    dist_chamfer_contact = ext.chamferDist()
    
    
    contact_dist, _ = dist_chamfer_contact(
                                    body_verts_contact_batch.contiguous(), 
                                    scene_points.contiguous()
                                    )
    

    loss_contact = (1 * 
                            torch.mean( torch.sqrt(contact_dist+1e-4)
                            /(torch.sqrt(contact_dist+1e-4)+1.0)  )  )
    
    #loss_sdf
    
    loss_sdf_pene=0
    
    for ttt in range(body_verts_batch__.shape[0]):
        
    
        body_verts_batch_=body_verts_batch__[ttt:ttt+1,:,:]
        
        norm_verts_batch = ((body_verts_batch_ - s_grid_min_batch.unsqueeze(1)) 
                                / (s_grid_max_batch.unsqueeze(1) - s_grid_min_batch.unsqueeze(1)) *2 -1)
        
        n_verts = norm_verts_batch.shape[1]

        body_sdf_batch = F.grid_sample(sdf.unsqueeze(1), 
                        norm_verts_batch[:,:,[2,1,0]].view(-1, n_verts,1,1,3),
                        padding_mode='border')


        # if there are no penetrating vertices then set sdf_penetration_loss = 0
        if body_sdf_batch.lt(0).sum().item() < 1:
            loss_sdf_pene_ = torch.tensor(0.0, dtype=torch.float32,device=device)

        else:
            loss_sdf_pene_ = body_sdf_batch[body_sdf_batch < 0].abs().mean()
        loss_sdf_pene=loss_sdf_pene+loss_sdf_pene_
        #print(loss_sdf_pene)
    
    #loss_sdf_pene=0

    return loss_rec, loss_vposer, loss_contact, loss_sdf_pene, loss_motion, loss_skating

def fitting(xhr_in, body, cam_extrinsic,
            s_verts, s_sdf, s_grid_min, s_grid_max,
            fittingconfig, lossconfig, start_end, vposer, body_mesh_model_batch, threshold=1.6, contact_id_folder='./data/body_segments', device='cuda'):
    
    body_mesh_model_input = smplx.create('./models', 
                               model_type='smplx',
                               gender='neutral', ext='npz',
                               num_pca_comps=12,
                               create_global_orient=True,
                               create_body_pose=True,
                               create_betas=True,
                               create_left_hand_pose=True,
                               create_right_hand_pose=True,
                               create_expression=True,
                               create_jaw_pose=True,
                               create_leye_pose=True,
                               create_reye_pose=True,
                               create_transl=True,
                               batch_size=2
                               )

    body_mesh_model_input=body_mesh_model_input.to(device)

    batch_size = xhr_in.shape[0]

    v = Variable(torch.randn(batch_size,62).to(device)
                , requires_grad=True)
    
    optimizer = optim.Adam([v], lr=fittingconfig['init_lr_h'])
    v.data = xhr_in.clone()

    
    cam_ext_batch = cam_extrinsic.repeat(batch_size, 1,1)
    #max_d_batch = max_d.repeat(batch_size)
    s_verts_batch = s_verts.repeat(batch_size, 1,1)
    s_sdf_batch = s_sdf.repeat(1, 1,1,1)
    s_grid_min_batch = s_grid_min.repeat(1, 1)
    s_grid_max_batch = s_grid_max.repeat(1, 1)
    
    
    
    for ii in range(fittingconfig['num_iter']):

        optimizer.zero_grad()
    
        loss_rec, loss_vposer, loss_contact, loss_collision, loss_motion, loss_skating = cal_loss(xhr_in, v, body, cam_ext_batch, s_verts_batch,
                                                                        s_sdf_batch,s_grid_min_batch, s_grid_max_batch,start_end
                                                                        , ii, vposer, body_mesh_model_input, body_mesh_model_batch, threshold, contact_id_folder, device)
        
        loss_rec=lossconfig['weight_loss_rec']*loss_rec
        loss_contact=lossconfig['weight_contact']*loss_contact
        loss_collision=lossconfig['weight_collision']*loss_collision

        loss_motion=lossconfig['weight_motion']*loss_motion
        loss_skating=lossconfig['weight_skating']*loss_skating
            
        loss = loss_rec + loss_contact + loss_collision+loss_motion+loss_skating
        '''
        if fittingconfig['verbose']:
            print('[INFO][fitting] iter={:d}, l_rec={:f},loss_skating={:f} ,l_contact={:f}, l_collision={:f},l_motion={:f},l_vposer={:f}'.format(
                                    ii, loss_rec, loss_skating,
                                    loss_contact, loss_collision,loss_motion,loss_vposer)) 
        '''
        loss.backward(retain_graph=True)
        optimizer.step()

    ### recover global translation and orientation
    #xh_rec = convert_to_3D_rot(xhr_rec)        
    #xhr_rec=torch.cat([v,xhr_in[:,6:]],dim=1)
    return v