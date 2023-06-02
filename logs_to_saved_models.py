import pandas as pd
import os
from os import listdir
from os.path import isfile, join
import re
from collections import OrderedDict
import json

import torch
from torch import tensor
import fun_Data_timeseries_basket
import fun_Data__bootstrap_wrapper
import fun_Data__MCsim_wrapper
import class_Neural_Network
import fun_train_NN_algorithm_defaults
import fun_RUN__wrapper
import class_NN_Pytorch
import torch

#-----------------------------------------------------------------------------------------------
# NEURAL NETWORK (NN) SETUP
#-----------------------------------------------------------------------------------------------


# Withdrawal NN: NN_withdraw 30.0
#---------------------------# Nr of hidden layers of NN
                   # NN will have total layers 1 (input) + N_L (hidden) + 1 (output) = N_L + 2 layers in total
                   # layer_id list: [0, 1,...,N_L, N_L+1]


NN_withdraw_orig = class_Neural_Network.Neural_Network(n_nodes_input = 2,
                                         n_nodes_output = 1,
                                         n_layers_hidden = 2)

print("Withdrawal NN:")
NN_withdraw_orig.print_layers_info()  #Check what to update

#Update layers info

for l in range(1, 2+1):
    NN_withdraw_orig.update_layer_info(layer_id = l , n_nodes = 2 + 8 , activation = "logistic_sigmoid", add_bias=True)
    
NN_withdraw_orig.update_layer_info(layer_id = 2+1, activation = "none", add_bias= False) #output layer

NN_withdraw_orig.print_layers_info() #Check if structure is correct
# ---------------------------------------------------------------------


# Allocation NN: NN_allocate 
#---------------------------# Nr of hidden layers of NN
                   # NN will have total layers 1 (input) + N_L (hidden) + 1 (output) = N_L + 2 layers in total
                   # layer_id list: [0, 1,...,N_L, N_L+1]


NN_allocate_orig = class_Neural_Network.Neural_Network(n_nodes_input = 2,
                                         n_nodes_output = 2,
                                         n_layers_hidden = 2)

print("Allocation NN:")
NN_allocate_orig.print_layers_info()  #Check what to update

#Update layers info
for l in range(1, 2+1):
    NN_allocate_orig.update_layer_info(layer_id = l , n_nodes = 2 + 8 , activation = "logistic_sigmoid", add_bias=True)

NN_allocate_orig.update_layer_info(layer_id = 2+1, activation = "softmax", add_bias= False)


NN_allocate_orig.print_layers_info() #Check if structure is correct
#---------------------------------------------------------------------
pytorch_flag = True
device = 'cuda' if torch.cuda.is_available() else 'cpu'

#put original NNs in list:
    
NN_orig_list = [NN_withdraw_orig, NN_allocate_orig]
    
# copy NN structures into pytorch NN
NN_withdraw = class_NN_Pytorch.pytorch_NN(NN_withdraw_orig)
NN_withdraw.to(device)
# NN_withdraw.import_weights(NN_withdraw_orig, params)

NN_allocate = class_NN_Pytorch.pytorch_NN(NN_allocate_orig)
NN_allocate.to(device)
# NN_allocate.import_weights(NN_allocate_orig, params)

NN_list = torch.nn.ModuleList([NN_withdraw, NN_allocate])


# os.chdir("/home/marcchen/Documents/testing_pyt_decum/researchcode/feb13_log_output_yyfeb3_big_PAPERMODEL")


model_folder = "/home/marcchen/Documents/testing_pyt_decum/researchcode/bootstrap_trained_models_block12month/models_rerun2019"

with open("/home/marcchen/Documents/testing_pyt_decum/researchcode/log_output/jun2_kappainf_bootstrap_train_12month.txt", 'r') as f:
    text = f.read()

lines_text = text.splitlines()

for i,line in enumerate(lines_text):
    if line == "saved model: ":
        model_str = "".join(lines_text[i+1:i+80])
        
        idx1 = model_str.index("OrderedDict([(")
        idx2 = model_str.index("))])")
        idx3 =model_str.index("Optimal xi:  [")
        
        opt_xi = re.split(r'\[|\]', model_str[idx3:idx3+30])[1]
        model_str = model_str[idx1:idx2+4]
        
    
        kappa = " ".join(lines_text[i+80:i+110]).split("Tracing param: ",1)[1].split(" ")[0]
        
        NN_list.load_state_dict(eval(model_str))
    
        torch.save(NN_list.state_dict(),f"{model_folder}/NN_opt_mc_decum_25-05-23_21:36_kappa_{kappa}")

        
        optimal_xi = {"xi":opt_xi}
        
        with open(f'{model_folder}/xi_opt_mc_decum_25-05-23_21:36_kappa_{kappa}.json', 'w') as outfile:
            json.dump(optimal_xi, outfile)
        
