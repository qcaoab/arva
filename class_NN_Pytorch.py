
import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from collections import OrderedDict

class pytorch_NN(nn.Module):
    def __init__(self, original_NN):

        super(pytorch_NN, self).__init__()
        self.flatten = nn.Flatten()

        # initialize from parameters/structure of original NN object
        n_nodes_input_orig = original_NN.n_nodes_input  #nr of input nodes = size of feature vector
        n_nodes_output_orig = original_NN.n_nodes_output    #nr of output nodes
        n_layers_hidden_orig = original_NN.L   # nr of hidden layers
        n_layers_total_orig = original_NN.n_layers_total
        
        #activation function helper
        
        def activation_function(name):
            if name == "logistic_sigmoid":
                return nn.Sigmoid()
            elif name == "ReLU":
                return nn.ReLU()
            elif name == "softmax":
                return nn.Softmax(dim=1)
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
        
        
        activation = {}
        def get_activation(name):
            def hook(model, input, output):
                activation[name] = output.detach()
            return hook
            
        # print structure
        pd.set_option("display.max_rows", None, "display.max_columns", None)
        print(nn_orig_df)
        print("Pytorch NN pbject created from original NN class. Change\
            original NN object to change structure.")
       

    def forward(self, input_tensor):
        return self.model(input_tensor.float())
    
    
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
        
        for i, layer in enumerate(pyt_state_dict.keys()):
            original_NN.layers[i+1].x_l = pyt_state_dict[layer].detach().cpu().numpy().T
        
        #copy from layers to theta vector
        original_NN.stack_NN_parameters()
        
        return original_NN
    
    
        
        
        