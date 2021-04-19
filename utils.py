import open3d as o3d
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


class ContinousRotReprDecoder(nn.Module):
    '''
    - this class encodes/decodes rotations with the 6D continuous representation
    - Zhou et al., On the continuity of rotation representations in neural networks
    - also used in the VPoser (see smplx)
    '''

    def __init__(self):
        super(ContinousRotReprDecoder, self).__init__()

    def forward(self, module_input):
        reshaped_input = module_input.view(-1, 3, 2)

        b1 = F.normalize(reshaped_input[:, :, 0], dim=1)

        dot_prod = torch.sum(b1 * reshaped_input[:, :, 1], dim=1, keepdim=True)
        b2 = F.normalize(reshaped_input[:, :, 1] - dot_prod * b1, dim=-1)
        b3 = torch.cross(b1, b2, dim=1)

        return torch.stack([b1, b2, b3], dim=-1)


    @staticmethod
    def decode(module_input):
        reshaped_input = module_input.view(-1, 3, 2)

        b1 = F.normalize(reshaped_input[:, :, 0], dim=1)

        dot_prod = torch.sum(b1 * reshaped_input[:, :, 1], dim=1, keepdim=True)
        b2 = F.normalize(reshaped_input[:, :, 1] - dot_prod * b1, dim=-1)
        b3 = torch.cross(b1, b2, dim=1)

        return torch.stack([b1, b2, b3], dim=-1)


    @staticmethod
    def matrot2aa(pose_matrot):
        '''
        :param pose_matrot: Nx1xnum_jointsx9
        :return: Nx1xnum_jointsx3
        '''
        
        homogen_matrot = F.pad(pose_matrot.view(-1, 3, 3), [0,1])
        pose = tgm.rotation_matrix_to_angle_axis(homogen_matrot).view(-1, 3).contiguous()
        return pose

    @staticmethod
    def aa2matrot(pose):
        '''
        :param Nx1xnum_jointsx3
        :return: pose_matrot: Nx1xnum_jointsx9
        '''
        pose_body_matrot = tgm.angle_axis_to_rotation_matrix(pose.reshape(-1, 3))[:, :3, :3].contiguous()
        return pose_body_matrot


class GeometryTransformer():
    
    @staticmethod
    def get_contact_id(body_segments_folder, contact_body_parts=['L_Hand', 'R_Hand']):

        contact_verts_ids = []
        contact_faces_ids = []

        for part in contact_body_parts:
            with open(os.path.join(body_segments_folder, part + '.json'), 'r') as f:
                data = json.load(f)
                contact_verts_ids.append(list(set(data["verts_ind"])))
                contact_faces_ids.append(list(set(data["faces_ind"])))

        contact_verts_ids = np.concatenate(contact_verts_ids)
        contact_faces_ids = np.concatenate(contact_faces_ids)


        return contact_verts_ids, contact_faces_ids

    @staticmethod
    def convert_to_6D_rot(x_batch):
        xt = x_batch[:,:3]
        xr = x_batch[:,3:6]
        xb = x_batch[:, 6:]

        xr_mat = ContinousRotReprDecoder.aa2matrot(xr) # return [:,3,3]
        xr_repr =  xr_mat[:,:,:-1].reshape([-1,6])

        return torch.cat([xt, xr_repr, xb], dim=-1)

    @staticmethod
    def convert_to_3D_rot(x_batch):
        xt = x_batch[:,:3]
        xr = x_batch[:,3:9]
        xb = x_batch[:,9:]

        xr_mat = ContinousRotReprDecoder.decode(xr) # return [:,3,3]
        xr_aa = ContinousRotReprDecoder.matrot2aa(xr_mat) # return [:,3]

        return torch.cat([xt, xr_aa, xb], dim=-1)



    @staticmethod
    def verts_transform(verts_batch, cam_ext_batch):
        verts_batch_homo = F.pad(verts_batch, (0,1), mode='constant', value=1)
        verts_batch_homo_transformed = torch.matmul(verts_batch_homo,
                                                    cam_ext_batch.permute(0,2,1))

        verts_batch_transformed = verts_batch_homo_transformed[:,:,:-1]
        
        return verts_batch_transformed


    @staticmethod
    def recover_global_T(x_batch, cam_intrisic, max_depth):
        xt_batch = x_batch[:,:3]
        xr_batch = x_batch[:,3:]

        fx_batch = cam_intrisic[:,0,0]
        fy_batch = cam_intrisic[:,1,1]
        # fx_batch = 1000
        # fy_batch = 1000
        px_batch = cam_intrisic[:,0,2]
        py_batch = cam_intrisic[:,1,2]
        s_ = 1.0 / torch.max(px_batch, py_batch)
        
        z = (xt_batch[:, 2]+1.0)/2.0 * max_depth

        x = xt_batch[:,0] * z / s_ / fx_batch
        y = xt_batch[:,1] * z / s_ / fy_batch
        
        xt_batch_recoverd = torch.stack([x,y,z],dim=-1)

        return torch.cat([xt_batch_recoverd, xr_batch],dim=-1)


    @staticmethod
    def normalize_global_T(x_batch, cam_intrisic, max_depth):
        '''
        according to the camera intrisics and maximal depth,
        normalize the global translate to [-1, 1] for X, Y and Z.
        input: [transl, rotation, local params]
        '''

        xt_batch = x_batch[:,:3]
        xr_batch = x_batch[:,3:]

        fx_batch = cam_intrisic[:,0,0]
        fy_batch = cam_intrisic[:,1,1]
        px_batch = cam_intrisic[:,0,2]
        py_batch = cam_intrisic[:,1,2]
        s_ = 1.0 / torch.max(px_batch, py_batch)
        x = s_* xt_batch[:,0]*fx_batch / (xt_batch[:,2] + 1e-6)
        y = s_* xt_batch[:,1]*fy_batch / (xt_batch[:,2] + 1e-6)

        z = 2.0*xt_batch[:,2] / max_depth - 1.0

        xt_batch_normalized = torch.stack([x,y,z],dim=-1)


        return torch.cat([xt_batch_normalized, xr_batch],dim=-1)




class BodyParamParser():

    @staticmethod
    def body_params_encapsulate(x_body_rec):
        x_body_rec_np = x_body_rec.detach().cpu().numpy()
        n_batch = x_body_rec_np.shape[0]
        rec_list = []

        for b in range(n_batch):
            body_params_batch_rec={}
            body_params_batch_rec['transl'] = x_body_rec_np[b:b+1,:3]
            body_params_batch_rec['global_orient'] = x_body_rec_np[b:b+1,3:6]
            body_params_batch_rec['betas'] = x_body_rec_np[b:b+1,6:16]
            body_params_batch_rec['body_pose'] = x_body_rec_np[b:b+1,16:48]
            body_params_batch_rec['left_hand_pose'] = x_body_rec_np[b:b+1,48:60]
            body_params_batch_rec['right_hand_pose'] = x_body_rec_np[b:b+1,60:]
            rec_list.append(body_params_batch_rec)

        return rec_list


    @staticmethod
    def body_params_encapsulate_batch(x_body_rec):
        
        body_params_batch_rec={}
        body_params_batch_rec['transl'] = x_body_rec[:,:3]
        body_params_batch_rec['global_orient'] = x_body_rec[:,3:6]
        body_params_batch_rec['betas'] = x_body_rec[:,6:16]
        body_params_batch_rec['body_pose'] = x_body_rec[:,16:48]
        #body_params_batch_rec['left_hand_pose'] = x_body_rec[:,48:60]
        #body_params_batch_rec['right_hand_pose'] = x_body_rec[:,60:]
        return body_params_batch_rec

    def body_params_encapsulate_batch_nobody(x_body_rec):
        
        body_params_batch_rec={}
        body_params_batch_rec['transl'] = x_body_rec[:,:3]
        body_params_batch_rec['global_orient'] = x_body_rec[:,3:6]
        #body_params_batch_rec['betas'] = x_body_rec[:,6:16]
        body_params_batch_rec['body_pose'] = x_body_rec[:,6:38]
        #body_params_batch_rec['left_hand_pose'] = x_body_rec[:,48:60]
        #body_params_batch_rec['right_hand_pose'] = x_body_rec[:,60:]
        return body_params_batch_rec
    
        
    def body_params_encapsulate_batch_hand(x_body_rec):
        
        body_params_batch_rec={}
        body_params_batch_rec['transl'] = x_body_rec[:,:3]
        body_params_batch_rec['global_orient'] = x_body_rec[:,3:6]
        body_params_batch_rec['betas'] = x_body_rec[:,6:16]
        body_params_batch_rec['body_pose'] = x_body_rec[:,16:48]
        body_params_batch_rec['left_hand_pose'] = x_body_rec[:,48:60]
        body_params_batch_rec['right_hand_pose'] = x_body_rec[:,60:]


        return body_params_batch_rec
    
    @staticmethod
    def body_params_encapsulate_batch_nobody_hand(x_body_rec):
        
        body_params_batch_rec={}
        body_params_batch_rec['transl'] = x_body_rec[:,:3]
        body_params_batch_rec['global_orient'] = x_body_rec[:,3:6]
        #body_params_batch_rec['betas'] = x_body_rec[:,6:16]
        body_params_batch_rec['body_pose'] = x_body_rec[:,6:38]
        body_params_batch_rec['left_hand_pose'] = x_body_rec[:,38:50]
        body_params_batch_rec['right_hand_pose'] = x_body_rec[:,50:]


        return body_params_batch_rec

    @staticmethod
    def body_params_encapsulate_latent(x_body_rec, eps=None):

        x_body_rec_np = x_body_rec.detach().cpu().numpy()
        eps_np = eps.detach().cpu().numpy()

        n_batch = x_body_rec_np.shape[0]
        rec_list = []

        for b in range(n_batch):
            body_params_batch_rec={}
            body_params_batch_rec['transl'] = x_body_rec_np[b:b+1,:3]
            body_params_batch_rec['global_orient'] = x_body_rec_np[b:b+1,3:6]
            body_params_batch_rec['betas'] = x_body_rec_np[b:b+1,6:16]
            body_params_batch_rec['body_pose'] = x_body_rec_np[b:b+1,16:48]
            body_params_batch_rec['left_hand_pose'] = x_body_rec_np[b:b+1,48:60]
            body_params_batch_rec['right_hand_pose'] = x_body_rec_np[b:b+1,60:]
            body_params_batch_rec['z'] = eps_np[b:b+1, :]
            rec_list.append(body_params_batch_rec)

        return rec_list

    @staticmethod
    def body_params_parse(body_params_batch):
        '''
        input:  body_params
                    |-- transl: global translation, [1, 3D]
                    |-- global_orient: global rotation, [1, 3D]
                    |-- betas:  body shape, [1, 10D]
                    |-- body_pose:  in Vposer latent space, [1, 32D]
                    |-- left_hand_pose: [1, 12]
                    |-- right_hand_pose: [1, 12]
                    |-- camera_translation: [1, 3D]
                    |-- camera_rotation: [1, 3x3 mat]
                z_s: scene representation [1, 128D]
        '''

        ## parse body_params_batch
        x_body_T = body_params_batch['transl']
        x_body_R = body_params_batch['global_orient']
        x_body_beta = body_params_batch['betas']
        x_body_pose = body_params_batch['body_pose']
        x_body_lh = body_params_batch['left_hand_pose']
        x_body_rh = body_params_batch['right_hand_pose']

        x_body = np.concatenate([x_body_T, x_body_R, 
                                 x_body_beta, x_body_pose,
                                 x_body_lh, x_body_rh], axis=-1)
        x_body_gpu = torch.tensor(x_body, dtype=torch.float32).cuda()

        return x_body_gpu


    @staticmethod
    def body_params_parse_fitting(body_params_batch):
        '''
        input:  body_params
                    |-- transl: global translation, [1, 3D]
                    |-- global_orient: global rotation, [1, 3D]
                    |-- betas:  body shape, [1, 10D]
                    |-- body_pose:  in Vposer latent space, [1, 32D]
                    |-- left_hand_pose: [1, 12]
                    |-- right_hand_pose: [1, 12]
                    |-- camera_translation: [1, 3D]
                    |-- camera_rotation: [1, 3x3 mat]
                z_s: scene representation [1, 128D]
        '''

        ## parse body_params_batch
        x_body_T = body_params_batch['transl']
        x_body_R = body_params_batch['global_orient']
        x_body_beta = body_params_batch['betas']
        x_body_pose = body_params_batch['body_pose']
        x_body_lh = body_params_batch['left_hand_pose']
        x_body_rh = body_params_batch['right_hand_pose']
        cam_ext = torch.tensor(body_params_batch['cam_ext'], dtype=torch.float32).cuda()
        cam_int = torch.tensor(body_params_batch['cam_int'], dtype=torch.float32).cuda()
        
        x_body = np.concatenate([x_body_T, x_body_R, 
                                 x_body_beta, x_body_pose,
                                 x_body_lh, x_body_rh], axis=-1)
        x_body_gpu = torch.tensor(x_body, dtype=torch.float32).cuda()

        return x_body_gpu, cam_ext, cam_int
