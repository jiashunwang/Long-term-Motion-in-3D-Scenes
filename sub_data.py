import torch.utils.data as data
import torch
import numpy as np
import random
import os
import json
import pickle 

class SUBDATA(data.Dataset):
    def __init__(self):
        self.data=np.load('./0814_input_middle_sub_data_30.npy',allow_pickle=True)
        
    def __getitem__(self, index):
        return self.data[index][0],self.data[index][1],self.data[index][2],self.data[index][3],\
            self.data[index][4],self.data[index][5],self.data[index][6],self.data[index][7]
    def __len__(self):
        return 8#(len(self.data)//8)*8


'''
def create_dataset(data):
        dataset=[]
        for i in range(len(data)):
            body_t = data[i]['transl']
            body_r = data[i]['global_orient']
            body_shape=data[i]['betas']
            body_pose = data[i]['pose_embedding']
            body_lhp = data[i]['left_hand_pose']
            body_rhp = data[i]['right_hand_pose']
            body = np.concatenate([body_t, body_r, body_shape, 
                                body_pose, body_lhp, body_rhp
                                ],
                                axis=-1)
        
            dataset.append(body)
        #train_input=[]
        #train_gt=[]
        #test_input=[]
        #test_gt=[]
        #np.random.seed(2020)
        #sample=np.random.choice(len(data),int(len(data)*0.8),replace=False)
        input_list=[]
        gt_list=[]
    
        for i in range(len(dataset)):
            np.random.seed(i*10)
            random.seed(i*10)
            a=random.random()
            b=random.random()
            if a<0.33:
                seq_len=4*6
            elif a<0.67:
                seq_len=6*6
            else:
                seq_len=8*6
            mid=int(i+0.5*seq_len+10*(b-0.5))
            
            try:
                input_=torch.tensor(np.concatenate([dataset[i],dataset[i+seq_len]]), dtype=torch.float32)
                #input_=torch.tensor(np.concatenate([dataset[i],dataset[i+seq*1],dataset[i+seq*10],dataset[i+seq*11]]), dtype=torch.float32)
                middle=torch.tensor(np.concatenate([dataset[mid-1],dataset[mid],dataset[mid+1],dataset[mid+2]])                                               
                            ,dtype=torch.float32)

                #if i in sample:
                #    train_input.append(input)
                #    train_gt.append(gt)
                #else:
                #    test_input.append(input)
                #    test_gt.append(gt)
                input_list.append(input_)
                gt_list.append(middle)
                
            except:
                pass
        #return train_input,train_gt,test_input,test_gt
        return input_list,gt_list


class DATA(data.Dataset):
    def __init__(self):
        self.sdf_path = scene_sdf_path='../../data/prox/sdf' 
        
        self.data=[]
        total_file_list=sorted(os.listdir('../../data/prox/PROXD/'))

        for each in total_file_list[:10]:
            scene_name=each[:-9]
            with open(os.path.join(scene_sdf_path, scene_name+'.json')) as f:
                sdf_data = json.load(f)
                grid_min = np.array(sdf_data['min'])
                grid_max = np.array(sdf_data['max'])
                grid_dim = sdf_data['dim']
            sdf = np.load(os.path.join(scene_sdf_path, scene_name + '_sdf.npy')).reshape(grid_dim, grid_dim, grid_dim)
            scene_name=each[:-9]
            filelist=os.listdir('../../data/prox/PROXD/'+each+'/results')
            filelist=sorted(filelist)
            temp_data_file=filelist
            data=[]
            for i in range(len(temp_data_file)):
                f=open('../../data/prox/PROXD/'+each+'/results/'+temp_data_file[i]+'/000.pkl','rb')
                temp=pickle.load(f)
                data.append(temp)
            sample_data=[]
            for i in range(0,len(data),2):
                sample_data.append(data[i])
            input_list,middle_list=create_dataset(sample_data)
            dis_list=[]
            full_list=[]
            for i in range(len(input_list)):
                dis=torch.mean((input_list[i][0][:3]-input_list[i][1][:3])**2)
                dis_list.append(dis)
                full_list.append([dis,input_list[i],middle_list[i]])
            sorted_list=sorted(full_list)[300:]
            for i in range(len(sorted_list)):
                self.data.append([sorted_list[i][1],sorted_list[i][2],each,scene_name,sdf])
            
    def __getitem__(self, index):
        return self.data[index][0],self.data[index][1],self.data[index][2],self.data[index][3],self.data[index][4]

    def __len__(self):
        return 1
'''