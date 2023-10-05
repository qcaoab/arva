
import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from collections import OrderedDict


# def sigmoid_w_constrain(nn_output,phi_2):
#     torch.sigmoid(phi_2*torch.maximum(0,nn_output))
#     return
    


class pytorch_NN(nn.Module):
    # def __init__(self, original_NN):

    #     super(pytorch_NN, self).__init__()
    #     self.flatten = nn.Flatten()

    #     #activation function helper        
    #     def activation_function(name):
    #         if name == "logistic_sigmoid":
    #             return nn.Sigmoid()
    #         elif name == "ReLU":
    #             return nn.ReLU()
    #         elif name == "softmax":
    #             return nn.Softmax(dim=1)
    #         elif name == "none":
    #             return nn.Identity()
    #         else:
    #             raise("activation function not included yet")

        
    #     # loop to get original NN structure and convert to pytorch layer 
    #     for l in np.arange(0, original_NN.n_layers_total, 1):
    #         orig_dict = {"obj.layers[layer_id]" : "obj.layers[" + str(l) + "]",
    #                       "layer_id" : original_NN.layers[l].layer_id,
    #                       "description": original_NN.layers[l].description,
    #                       "n_nodes" : original_NN.layers[l].n_nodes,
    #                       "activation":  original_NN.layers[l].activation,
    #                       "x_l(weights)":  [original_NN.layers[l].x_l_shape],
    #                       "add_bias" : original_NN.layers[l].add_bias,
    #                       "b_l(biases)" : original_NN.layers[l].b_l_length}

    #         if l == 0 :
    #             nn_orig_df = pd.DataFrame.from_dict(orig_dict)

    #         else:
    #             nn_orig_df = pd.concat([nn_orig_df,pd.DataFrame.from_dict(orig_dict)])

    #     nn_orig_df.reset_index()
        
        
    #     #create ordered dict for pytorch layers
    #     pytorch_layers = OrderedDict()
        
    #     #iterate to create each pytorch layer, skipping input layer
    #     for l in np.arange(1, original_NN.n_layers_total, 1):

    #         #create layer name
    #         name = nn_orig_df["description"].iloc[l] + "_" + str(l) 
                        
    #         #create linear layers
    #         pytorch_layers[name] = nn.Linear(nn_orig_df["x_l(weights)"].iloc[l][0], 
    #                                             nn_orig_df["x_l(weights)"].iloc[l][1],
    #                                             bias = nn_orig_df["add_bias"].iloc[l])
        
    #         #activation function 
    #         pytorch_layers[name + "_activation"] = activation_function(nn_orig_df["activation"].iloc[l])
            
    #     self.model = nn.Sequential(pytorch_layers)
        
            
    #     # print structure
    #     pd.set_option("display.max_rows", None, "display.max_columns", None)
    #     print(nn_orig_df)
    #     print("Pytorch NN pbject created from original NN class. Change\
    #         original NN object to change structure.")
    
    def __init__(self, nn_options):

        super(pytorch_NN, self).__init__()
        self.flatten = nn.Flatten()

        #activation function helper        
        def activation_function(name):
            if name == "logistic_sigmoid":
                return nn.Sigmoid()
            elif name == "ReLU":
                return nn.ReLU()
            elif name == "softmax":
                return nn.Softmax(dim=1)
            elif name == "none":
                return nn.Identity()
            else:
                raise("activation function not included yet")

        #unpack NN options
        nn_purpose = nn_options["nn_purpose"]
        N_layers_hid = nn_options["N_layers_hid"] 
        N_nodes_input = nn_options["N_nodes_input"] 
        N_nodes_hid = nn_options["N_nodes_hid"] 
        N_nodes_out = nn_options["N_nodes_out"]
        hidden_activation = nn_options["hidden_activation"]        
        output_activation = nn_options["output_activation"]                    
        biases = nn_options["biases"] 
        
        # create list of number of nodes in each layer:
        nodes_list = [N_nodes_hid]*N_layers_hid
        nodes_list.insert(0, N_nodes_input)
        nodes_list.append(N_nodes_out) 
        
        #create ordered dict for pytorch layers
        pytorch_layers = OrderedDict()
        
        #iterate to create each hidden layer (input layer is automatically created in pytorch) 
        for l in range(len(nodes_list)-1):

            #create layer name
            name = "hidden_layer_" + str(l) 
                        
            #create linear layers
            pytorch_layers[name] = nn.Linear(nodes_list[l], 
                                                nodes_list[l+1],
                                                bias = biases)
        
            # add activation function 
            if l == len(nodes_list)-2: #output layer
                pytorch_layers[name + "_activation"] = activation_function(output_activation)
            else: #hidden layers
                pytorch_layers[name + "_activation"] = activation_function(hidden_activation)
            
        self.model = nn.Sequential(pytorch_layers)       
    
    
    def forward(self, input_tensor):
        return self.model(input_tensor)    
        
        