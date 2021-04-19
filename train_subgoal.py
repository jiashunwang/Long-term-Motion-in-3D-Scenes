import torch
import torch.optim as optim
import numpy as np
from sub_data import SUBDATA
import time
import torch.nn.functional as F
from human_body_prior.tools.model_loader import load_vposer
from utils import BodyParamParser, ContinousRotReprDecoder, GeometryTransformer
import smplx
import chamfer_pytorch.dist_chamfer as ext
from sub_goal import Pointnet,SUBGOAL

start = time.time()
batch_size = 8
dataset = SUBDATA()
dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)

contact_id_folder = './data/body_segments'
contact_part = ['back','gluteus','L_Hand','R_Hand','L_Leg','R_Leg','thighs']

vposer, _ = load_vposer('./vposer_v1_0', vp_model='snapshot')
vposer = vposer.cuda()

body_mesh_model = smplx.create('./models', 
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
                                            batch_size=batch_size
                                            )

body_mesh_model = body_mesh_model.cuda()
print('finish data loading time:',time.time()-start)

model = SUBGOAL(n_dim_body=65)
model = model.cuda()
lrate = 0.0005
optimizer = optim.Adam(model.parameters(), lr=lrate)

for epoch in range(50):
    
    start_time = time.time()
    
    
    total_loss = 0
    total_collision_loss = 0
    total_contact_loss = 0
    total_kl_loss = 0
    total_rec_loss = 0
    total_rec_orient_loss = 0
    total_rec_transl_loss = 0

    for j,data in enumerate(dataloader,0):
        optimizer.zero_grad()
        
        
        middle_list,_,scene_name,sdf,scene_points,cam_extrinsic,s_grid_min_batch,s_grid_max_batch = data
        
        body = middle_list[:,0,6:16].cuda()
        
        middle_list = torch.cat([middle_list[:,:,:6],middle_list[:,:,16:]],dim=2)
        
        middle = middle_list[:,0,:].cuda()
        scene_points = scene_points.cuda()
        sdf = sdf.cuda()
        s_grid_max_batch = s_grid_max_batch.cuda()
        s_grid_min_batch = s_grid_min_batch.cuda()

        middle = GeometryTransformer.convert_to_6D_rot(middle)

        rec, mu, logsigma2 = model(middle,scene_points.transpose(1,2),middle[:,:9],body)
        
        
        loss_rec_transl = 0.5*(F.l1_loss(rec[:,:3], middle[:,:3]))
        loss_rec_orient = (F.l1_loss(rec[:,3:9], middle[:,3:9]))
        loss_rec = (F.l1_loss(rec[:,9:41], middle[:,9:41]))+loss_rec_transl+loss_rec_orient+0.1*F.l1_loss(rec[:,41:], middle[:,41:])
        
        fca = 1.0
        fca = min(1.0, max(float(epoch) / (10*0.75),0) )
        loss_KL = (fca**2 * 0.1*torch.mean(torch.exp(logsigma2) +mu**2 -1.0 -logsigma2))

        #body mesh

        rec = GeometryTransformer.convert_to_3D_rot(rec)
        body_param_rec = BodyParamParser.body_params_encapsulate_batch_nobody_hand(rec)
        body_param_rec['body_pose'] = vposer.decode(body_param_rec['body_pose'], 
                                    output_type='aa').view(rec.shape[0], -1)
        body_param_rec['betas'] = body
        
        smplx_output = body_mesh_model(return_verts=True, **body_param_rec)

        #body_verts_batch is with scene pointcloud
        #body_verts_batch_ is with scene sdf
        
        body_verts_batch = smplx_output.vertices #[b, 10475,3]
        body_verts_batch_ = GeometryTransformer.verts_transform(body_verts_batch, torch.tensor(cam_extrinsic,dtype=torch.float32).cuda())
        

        #contact loss
        vid, fid = GeometryTransformer.get_contact_id(body_segments_folder=contact_id_folder,
                                contact_body_parts=contact_part)

        body_verts_contact_batch = body_verts_batch[:, vid, :]
        dist_chamfer_contact = ext.chamferDist()
        
        contact_dist, _ = dist_chamfer_contact(
                                        body_verts_contact_batch.contiguous(), 
                                        scene_points.contiguous()
                                        )

        loss_contact = (1 * torch.mean( torch.sqrt(contact_dist+1e-4)
                                /(torch.sqrt(contact_dist+1e-4)+1.0)  )  )
       
        #collision loss
        norm_verts_batch = ((body_verts_batch_ - s_grid_min_batch.unsqueeze(1)) 
                                / (s_grid_max_batch.unsqueeze(1) - s_grid_min_batch.unsqueeze(1)) *2 -1)
        
        n_verts = norm_verts_batch.shape[1]
        
        body_sdf_batch = F.grid_sample(sdf.unsqueeze(1), 
                        norm_verts_batch[:,:,[2,1,0]].view(-1, n_verts,1,1,3),
                        padding_mode='border')

        if body_sdf_batch.lt(0).sum().item() < 1:
            loss_sdf_pene = torch.tensor(0.0, dtype=torch.float32).cuda()
                                         
        else:
            loss_sdf_pene = body_sdf_batch[body_sdf_batch < 0].abs().mean()
        
        loss_KL = loss_KL
        loss_rec = 0.1*loss_rec
        loss_sdf_pene = 0.01*loss_sdf_pene
        loss_contact = 0.01*loss_contact
        
        loss = loss_KL+loss_rec+loss_sdf_pene+loss_contact
    
        loss.backward()
        optimizer.step()

        total_collision_loss = total_collision_loss+loss_sdf_pene
        total_loss = total_loss+loss

        total_contact_loss = total_contact_loss+loss_contact
        total_kl_loss = total_kl_loss+loss_KL
        total_rec_loss = total_rec_loss+loss_rec
        total_rec_orient_loss = total_rec_orient_loss+0.1*loss_rec_orient
        total_rec_transl_loss = total_rec_transl_loss+0.1*loss_rec_transl

    print('##################################')
    print('##################################')
    print('epoch:',epoch)
    end_time=time.time()
    print('time:',end_time-start_time)
    print('total:',total_loss/((j+1)))
    print('collison:',total_collision_loss/(j+1))
    print('contact:',total_contact_loss/((j+1)))
    print('kl:',total_kl_loss/(j+1))
    print('rec_orient:',total_rec_orient_loss/(j+1))
    print('rec_transl:',total_rec_transl_loss/(j+1))
    print('rec:',total_rec_loss/(j+1))

    print('##################################')
    print('##################################')

    if (epoch+1) % 5 == 0:
        save_path = './saved_model/subgoal_'+str(epoch)+'.model'        
        print(save_path)
        torch.save(model.state_dict(),save_path)







