#This is the training scripts for random drop model

import numpy as np
import os
import logging
import argparse
from model import structure_transformer
from optimizer import ScheduledOptim
from model_fit import fit
from parse_config import parse_config
from random_drop import random_choice
from retrieve_drop import retrieve_array
from write_drop import write_drop
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader

parser = argparse.ArgumentParser(description = 'Training model for structure prediction')
parser.add_argument('-c',dest = 'config',default = 'config.txt',
                    help = 'the config file for trianing data and working directory.')
print('parsing the config file.')
c = parse_config(parser.parse_args())

#load dataset
spec_train_fp = c['spec_train'] 
spec_val_fp = c['spec_val']

spec_train = np.loadtxt(spec_train_fp,delimiter=",")
spec_val = np.loadtxt(spec_val_fp,delimiter=",")

#split off the target from the spectral information
X_train = spec_train[:,67:]
Y_train = spec_train[:,0:67]
X_val = spec_val[:,67:]
Y_val = spec_val[:,0:67]

#Set up logger
log_name = c['loss_path']
logging.basicConfig(filename = log_name, level = logging.INFO,
                    format = '%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(log_name)

#Assign parameters
num_tokens = 43
vocab_size = 345
dim_model = 256
num_heads = 8
num_feed_forward = 2048
num_spec_enc_layers = 4
num_spec_dec_layers = 4
tgt_maxlen = 67
ms_maxlen = 999
ir_maxlen = 900
nmr_maxlen = 993
epochs = 50
batch_size = 20
warmup_steps = int(len(X_train)/batch_size*50.0*0.04)
mode_lst = [int(element) for element in c['mode_lst']]
best_chk_path = c['best_chk_path']
end_chk_path = c['end_chk_path']
device = torch.device('cuda')
val_min = 1
patience = 30
current_patience = 0


#Apply random drop and save the dropped information 
ms_drop_list, ir_drop_list, nmr_drop_list, X_train = random_choice(X_train,0.7)
ms_val_drop, ir_val_drop, nmr_val_drop, X_val = random_choice(X_val,0.7)
write_drop(ms_drop_list,'ms_train_drop_0.7.txt')
write_drop(ir_drop_list,'ir_train_drop_0.7.txt')
write_drop(nmr_drop_list,'nmr_train_drop_0.7.txt')
write_drop(ms_val_drop,'ms_val_drop_0.7.txt')
write_drop(ir_val_drop,'ir_val_drop_0.7.txt')
write_drop(nmr_val_drop,'nmr_val_drop_0.7.txt')

'''
#Retrieve random drop for continue training
X_train = retrieve_array('ms_train_drop_0.7.txt','ir_train_drop_0.7.txt','nmr_train_drop_0.7.txt',X_train)
X_val = retrieve_array('ms_val_drop_0.7.txt','ir_val_drop_0.7.txt','nmr_val_drop_0.7.txt',X_val)
'''

#Prepare input
tensor_x_train = torch.Tensor(X_train)
tensor_y_train = torch.Tensor(Y_train)
train_dataset = TensorDataset(tensor_x_train,tensor_y_train)
train_dataloader = DataLoader(train_dataset, batch_size = batch_size, shuffle = True)

tensor_x_val = torch.Tensor(X_val)
tensor_y_val = torch.Tensor(Y_val)
val_dataset = TensorDataset(tensor_x_val,tensor_y_val)
val_dataloader = DataLoader(val_dataset, batch_size = batch_size, shuffle = True)

model = structure_transformer(vocab_size,
                              num_tokens,
                              dim_model,
                              num_heads,
                              num_feed_forward,
                              num_spec_enc_layers,
                              num_spec_dec_layers,
                              mode_lst,
                              device
                              )
model = model.to(device)

'''
PATH = END_CHECKPOINT_PATH
checkpoint = torch.load(PATH)
model.load_state_dict(checkpoint['model_state_dict'])
best_path = BEST_CHECKPOINT_PATH
best_checkpoint = torch.load(best_path)
val_min = best_checkpoint['val_min']
'''

optimizer = torch.optim.Adam(model.parameters(), lr = 0.0001, betas = (0.9, 0.98), eps = 1e-9)
sched = ScheduledOptim(optimizer, lr_mul = 1, d_model = dim_model, n_warmup_steps = warmup_steps, n_current_steps = checkpoint['final_steps'])

fit(model, sched, mode_lst, train_dataloader, val_dataloader, num_tokens, epochs, val_min, patience, current_patience, best_chk_path, device, logger)
torch.save({
            'model_state_dict': model.state_dict(),
            'final_steps': sched.get_final_steps(),
            }, end_chk_path)
