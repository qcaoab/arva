import pickle
import torch
import os
import json
import time
import numpy as np
import pandas as pd
import copy

def save_model(model_object, params, other):
    
    #OBJECTIVE: Gather the the trained model and all metadata necessary to replicate it and save it into a serialized dictionary object. 
    #---------------------------------------------------------------------------------------------
    #INPUTS: model_object = the Pytorch NN object
    #        params = same params dictioary used to organize all model parameters
    #        other = any other needed parameters to save, such as optimal "xi", in case of mean-CVAR objective #                function  
    #---------------------------------------------------------------------------------------------
    #OUTPUTS: none, but automatically outputs saved dictionary object including model and all metadata in the saved_model/ directory specified in driver file.
    #---------------------------------------------------------------------------------------------
    
    # unpack 'other' according to objective function:
    if params['obj_fun'] == "mean_cvar_single_level":
        opt_xi = other
                     
    # Gather standardizing parameters that were used on the training features
    standardization_values = {"benchmark_W_mean_train":params["benchmark_W_mean_train"].tolist(),
                            "benchmark_W_std_train":params["benchmark_W_std_train"].tolist(),
                            "benchmark_W_mean_train_post_withdraw":params["benchmark_W_mean_train_post_withdraw"].tolist(),
                            "benchmark_W_std_train_post_withdraw":params["benchmark_W_std_train_post_withdraw"].tolist()}

    # Delete large data from 'params': since it would be cumbersome to save the entire dataset. The data can be recreated using the other information kept in params. 
    to_delete = ['Y_train', 'Y_order_train','Y', 'Y_order','W','Feature_phi_paths_withdrawal', 'Feature_phi_paths_allocation', 'NN_asset_prop_paths', 'W_T']
    
    params_to_save = copy.deepcopy(params)
    
    for key in to_delete:
        del params_to_save[key]
    
    # create dictionary object
    saved_model_dict = {}    
    saved_model_dict["NN_object"] = model_object
    saved_model_dict["params_saved"] = params_to_save
    saved_model_dict["opt_xi"] = opt_xi
    saved_model_dict["standardization_values"] = standardization_values
    saved_model_dict["kappa"] = params["obj_fun_rho"]
    
    # dump into pickle object
    filehandler_dump = open(params["saved_model_dir"] + "kappa_" + str(params["obj_fun_rho"]) + ".pkl", "wb")
    pickle.dump(saved_model_dict, filehandler_dump)
    filehandler_dump.close()

    return  

def load_model(model_pickle_filepath, params_active):
    
    #unpickle
    with open(model_pickle_filepath,'rb') as f:
        model_dict = pickle.load(f)
    
    #unpack
    NN_object = model_dict["NN_object"]
    standardization_values_dict = model_dict["standardization_values"]
    params_saved = model_dict["params_saved"]
    
    # put standardization values in active 'params' dictionary
    for (k,v) in standardization_values_dict.items():
        params_active[k] = np.array(v)
    
    
        
    # Set flag so that standardization values are not overwritten
    params_active["sideloaded_standardization"] = True
    
    # overwrite initial xi value
    params_active["xi_0"] = model_dict["opt_xi"]
    
    
    # TODO: implement functionality to warn user if 'params_saved' and 'params_active' are inconsistent.

    
    return NN_object, params_active