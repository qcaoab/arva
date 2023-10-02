
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
    def __init__(self, original_NN):

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

        
        # loop to get original NN structure and convert to pytorch layer 
        for l in np.arange(0, original_NN.n_layers_total, 1):
            orig_dict = {"obj.layers[layer_id]" : "obj.layers[" + str(l) + "]",
                          "layer_id" : original_NN.layers[l].layer_id,
                          "description": original_NN.layers[l].description,
                          "n_nodes" : original_NN.layers[l].n_nodes,
                          "activation":  original_NN.layers[l].activation,
                          "x_l(weights)":  [original_NN.layers[l].x_l_shape],
                          "add_bias" : original_NN.layers[l].add_bias,
                          "b_l(biases)" : original_NN.layers[l].b_l_length}

            if l == 0 :
                nn_orig_df = pd.DataFrame.from_dict(orig_dict)

            else:
                nn_orig_df = pd.concat([nn_orig_df,pd.DataFrame.from_dict(orig_dict)])

        nn_orig_df.reset_index()
        
        
        #create ordered dict for pytorch layers
        pytorch_layers = OrderedDict()
        
        #iterate to create each pytorch layer, skipping input layer
        for l in np.arange(1, original_NN.n_layers_total, 1):

            #create layer name
            name = nn_orig_df["description"].iloc[l] + "_" + str(l) 
                        
            #create linear layers
            pytorch_layers[name] = nn.Linear(nn_orig_df["x_l(weights)"].iloc[l][0], 
                                                nn_orig_df["x_l(weights)"].iloc[l][1],
                                                bias = nn_orig_df["add_bias"].iloc[l])
        
            #activation function 
            pytorch_layers[name + "_activation"] = activation_function(nn_orig_df["activation"].iloc[l])
            
        self.model = nn.Sequential(pytorch_layers)
        
            
        # print structure
        pd.set_option("display.max_rows", None, "display.max_columns", None)
        print(nn_orig_df)
        print("Pytorch NN pbject created from original NN class. Change\
            original NN object to change structure.")
    
    
    # def __init__(self, nn_options):

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

        
    #     N_layers_h = nn_options["N_layers"] 
    #     N_input = nn_options["N_input"] 
    #     N_nodes = nn_options["N_nodes"] 
    #     hidden_activation = nn_options["hidden_activation"]        
    #     output_activation = nn_options["output_activation"]                    
    #     biases = nn_options["biases"] 
        
    #     # create list of number of nodes in each layer:
    #     nodes_list = [N_nodes]*N_layers_h
    #     nodes_list.insert(0, N_input)
    #     nodes_list.insert(-1, ) 
        
    #     #create ordered dict for pytorch layers
    #     pytorch_layers = OrderedDict()
        
    #     #iterate to create each hidden layer (input layer is automatically created in pytorch) 
    #     for l in range(len(nodes_list)):

    #         #create layer name
    #         name = "hidden_layer_" + str(l) 
                        
    #         #create linear layers
    #         pytorch_layers[name] = nn.Linear(nodes_list[l], 
    #                                             nodes_list[l+1],
    #                                             bias = biases)
        
    #         #activation function 
    #         pytorch_layers[name + "_activation"] = activation_function(nn_orig_df["activation"].iloc[l])
            
    #     self.model = nn.Sequential(pytorch_layers)
        
            
    #     # print structure
    #     pd.set_option("display.max_rows", None, "display.max_columns", None)
    #     print(nn_orig_df)
    #     print("Pytorch NN pbject created from original NN class. Change\
    #         original NN object to change structure.")
    
       

    def forward(self, input_tensor):
        return self.model(input_tensor)
    
    
    def import_weights(self, original_NN, params):
        
        original_NN.unpack_NN_parameters()
        
        pyt_state_dict = self.state_dict()
        
        # if original_NN has bias, raise exception
        for i in range(1,len(original_NN.layers)):
            if original_NN.layers[i].b_l is not None:
                raise Exception("Support for NN biases not implemented yet. -mc")
            
        for i, layer in enumerate(pyt_state_dict.keys()):
            pyt_state_dict[layer] = torch.tensor(original_NN.layers[i+1].x_l.T, 
                                                device = params["device"])
        
        self.load_state_dict(pyt_state_dict)
    
    def export_weights(self, original_NN):
        
        pyt_state_dict = self.state_dict()
        
        pyt_state_key_list = list(pyt_state_dict.keys())
        pyt_state_weights = filter(lambda k: '.weight' in k, pyt_state_key_list)
        pyt_state_biases = filter(lambda k: '.bias' in k, pyt_state_key_list)
        
        
        for i, layer in enumerate(pyt_state_weights):
            original_NN.layers[i+1].x_l = pyt_state_dict[layer].detach().cpu().numpy().T
        
        for i, layer in enumerate(pyt_state_biases):
            original_NN.layers[i+1].b_l = pyt_state_dict[layer].detach().cpu().numpy().T
           
        #copy from layers to theta vector
        original_NN.stack_NN_parameters()
        
        return original_NN
    
    
        
        
        