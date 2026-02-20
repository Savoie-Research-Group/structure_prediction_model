import argparse
import torch
import pickle
import numpy as np
import concurrent.futures
import logging
from calc_decisive import bs_accuracy
from parse_config_smiles import parse_config
import sys
sys.path.append('/depot/bsavoie/data/Tianfan/structure_paper/train_script')
from model import structure_transformer

parser = argparse.ArgumentParser(description = 'Evaluate deduction model performance')
parser.add_argument('-c',dest = 'config',default = 'config_eval.txt',
                    help = 'the config file for training data and working directory.')
print('parsing the config file.')
c = parse_config(parser.parse_args())

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
mode_lst = [int(element) for element in c['mode_lst']]
best_chk_path = c['best_chk_path']
txt_path = c['txt_path']
device = torch.device('cpu')
idx_to_ch_fh = open(c['idx_to_ch_dict_path'], "rb")
idx_to_ch_dict = pickle.load(idx_to_ch_fh)
ch_to_idx_fh = open(c['ch_to_idx_dict_path'], "rb")
ch_to_idx_dict = pickle.load(ch_to_idx_fh)
target_start_token_idx = ch_to_idx_dict["<"]
target_end_token_idx = ch_to_idx_dict["$"]
test_output_path = c['test_output_path']

#Load model checkpoint
model = structure_transformer(vocab_size,
                              num_tokens,
                              dim_model,
                              num_heads,
                              num_feed_forward,
                              num_spec_enc_layers,
                              num_spec_dec_layers,
                              [1,1,1],
                              device
                              )
model = model.to(device)
checkpoint = torch.load(best_chk_path,map_location=torch.device('cpu'))
model.load_state_dict(checkpoint['model_state_dict'])

#Load test set data
tgt_test_fp = c['tgt_test']
tgt_test = np.loadtxt(tgt_test_fp,delimiter=",")
num_example = (int(tgt_test.shape[0] / 100) + 1) * 100
mol_test = tgt_test[:,0:67]
spec_test = tgt_test[:,67:]
X_test = torch.Tensor(spec_test)
Y_test = torch.Tensor(mol_test)

#Calculate overall accuracy
model.eval()

log_name = c['log_path']
logging.basicConfig(filename = log_name, level = logging.INFO,
                    format = '%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(log_name)

correct_num_top1 = []
correct_num_top10 = []
pred_10 = []

'''
with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
    futures = []
    for i in range(0,num_example,1):
        batch = (X_test[i:i+1,:],Y_test[i:i+1,:])
        future = executor.submit(bs_accuracy,batch,model,tgt_maxlen,idx_to_ch_dict,target_start_token_idx,target_end_token_idx)
        futures.append(future)

    for future in concurrent.futures.as_completed(futures):
        result1, result2, batch_pred = future.result()
        correct_num_top1.append(result1)
        correct_num_top10.append(result2)
        pred_10 += batch_pred
        length = len(correct_num_top1)
        logger.info(f'complete {length}')
'''
for i in range(0,num_example,100):
    batch = (X_test[i:i+100,:],Y_test[i:i+100,:])
    count_top1,batch_pred,batch_decisive = bs_accuracy(batch,model,tgt_maxlen,idx_to_ch_dict,target_start_token_idx,target_end_token_idx,mode_lst)
    correct_num_top1.append(count_top1)
    logger.info(f'start_number {i}')
    with open(txt_path,'a') as f:
        for ele in batch_pred:
            f.write(ele)
            f.write('\n')
    with open(test_output_path,'a') as f:
        for ele in batch_decisive:
            f.write(ele)
            f.write('\n')
print(sum(correct_num_top1))



    
