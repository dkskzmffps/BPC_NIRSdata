import torch
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import argparse
import numpy as np

import JH_utile
# import JH_DataAug
import JH_net_param_setting as NetSet
import sys
import math
import os
import time
li = np.array([[10, 20, 30, 40, 50],
               [100, 200, 300, 400, 500]])

print(li.shape)

print(li[0,3])


li = np.array([[[10, 1, 5, 10, 10],[10, 1, 5, 10, 10]],[[10, 1, 5, 10, 10],[10, 1, 5, 10, 10]]])
# min = np.min(li)
# max = np.max(li)
# mean = np.mean(li[:,:,:])
# std = np.std(li[:,:,0])
# a = li.shape[1]

li_size= li.shape


li_tensor = torch.from_numpy(li)

li_permut = li_tensor.permute(2,0,1)



# ====================================================================================================================#
# Define path ========================================================================================================#

dataset_path = '/home/jhpnih/0_DATA/EEG_Data/NIH/EEG_sampled_data_Tensor/'

data_list_sampled = torch.load(dataset_path+'/data_list_dict.pt')

temp_file = dataset_path + data_list_sampled['sampled_data_list_each_train'][0][1][0]
temp_data = torch.load(temp_file)
data_max, data_min = data_list_sampled['max_value'], data_list_sampled['min_value']

pos_data = temp_data - data_min
nor_data = pos_data / (data_max - data_min)
nor_data_max = nor_data.max()
nor_data_min = nor_data.min()
print('check')