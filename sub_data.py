import torch.utils.data as data
import torch
import numpy as np
import random
import os
import json
import pickle 

class SUBDATA(data.Dataset):
    def __init__(self):
        self.data=np.load('./data/sub_data_training.npy',allow_pickle=True)
        
    def __getitem__(self, index):
        return self.data[index][0],self.data[index][1],self.data[index][2],self.data[index][3],\
            self.data[index][4],self.data[index][5],self.data[index][6],self.data[index][7]
    def __len__(self):
        return (len(self.data)//8)*8
