import torch.utils.data as data
import torch
import numpy as np

class ROUTEDATA(data.Dataset):
    def __init__(self):
        self.data=np.load('./0926_s1_hand_data_down1_seq60_training.npy',allow_pickle=True)
        self.len=(len(self.data)//8)*8 
    def __getitem__(self, index):
        return self.data[index][0],self.data[index][1],self.data[index][2],self.data[index][3],\
                self.data[index][4],self.data[index][5],self.data[index][6],self.data[index][7],self.data[index][8]
    def __len__(self):
        return self.len