import numpy as np
import fun_Objective_functions
import fun_invest_NN_strategy
import torch


def eval_obj_NN_strategy_pyt(NN_list, params, xi):
    
    #Objective: Selected the chosen objective function from fun_Objective_functions.py
    #           Calculates objective function value F_val.
    #NOTE: through the use of the function fun_invest_NN_strategy.invest_NN_strategy_pyt below,
    #       this code also "invests" the NN control (pytorch) as investment strategy and updates "params"
    #       withdrawal NN functionality is also added.

    
    #OUTPUTS:
    # return F_val, params
    # params dictionary, with added fields:
    #      ALL fields added by the FUNCTION:  fun_invest_NN_strategy.invest_NN_strategy
    #     params["F_val"] = F_val     #Obj func value
   
    #INPUTS:
    # NN_pyt = object of pytorch NN class with structure as setup in main code
    # params = dictionary with investment parameters as set up in main code
    
    # ---------------------Invest according to given NN_object with params NN_theta------------------------------

    
    if params["nn_withdraw"]: #if decumulation problem
        
        #Calculate the  wealth paths, terminal wealth: essentially a forward pass of NN for each time step.
        params, W_T_vector, qsum_T_vector = fun_invest_NN_strategy.withdraw_invest_NN_strategy(NN_list, params)

        #Select objective function from fun_Objective_functions.py:
        if params["obj_fun"] == "mean_cvar_single_level":
            
            # xi is initialized as tensor in params["xi"] in driver code
            
            fun = fun_Objective_functions.objective_mean_cvar_decumulation(params, qsum_T_vector, W_T_vector, xi)
    
    else: # just allocation problem: NO DECUMULATION
        
        params, W_T_vector = fun_invest_NN_strategy.invest_NN_strategy_pyt(NN_list, params)

        #Select objective function from fun_Objective_functions.py:
        if params["obj_fun"] == "mean_cvar_single_level":
            
            # xi is initialized as tensor in params["xi"] in driver code
            
            fun = fun_Objective_functions.objective_mean_cvar_pytorch(params, W_T_vector, xi)
        
    
    # options for additional objective functions should be added here.
        
    return fun, params

