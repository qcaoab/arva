import numpy as np
import fun_Objective_functions
import fun_invest_NN_strategy
import torch


def eval_obj_NN_strategy_pyt(NN_pyt, params):
    
    #Objective: Calculates the pytorch (mean cvar only atm) objective function value F_val.
    #NOTE: through the use of the function fun_invest_NN_strategy.invest_NN_strategy_pyt below,
    #       this code also "invests" the NN control (pytorch) as investment strategy and updates "params"

    
    #OUTPUTS:
    # return params, F_val
    # params dictionary, with added fields:
    #      ALL fields added by the FUNCTION:  fun_invest_NN_strategy.invest_NN_strategy
    #     params["F_val"] = F_val     #Obj func value
    #                                  if params["obj_fun"] = "mean_cvar" this is the LAGRANGIAN


    #INPUTS:
    # NN_pyt = object of pytorch NN class with structure as setup in main code
    # params = dictionary with investment parameters as set up in main code
    
    # ---------------------Invest according to given NN_object with params NN_theta------------------------------

    #Calculate the  wealth paths, terminal wealth
    
    params, Q_prev, Q , g = fun_invest_NN_strategy.invest_NN_strategy_pyt(NN_pyt, params)
    # modify the global variable
    
    #Unpack F_theta

    #if params["obj_fun"] == "mean_cvar_single_level":
        
        # xi currently initialized as tensor in driver code
        #W_T_vector = g
        
        #fun = fun_Objective_functions.objective_mean_cvar_pytorch(params, g, xi)
    
        #for output
        #params["F_val"] = fun.detach().to("cpu").numpy()     #Obj func value
        #params["xi"] = xi
    
    
    if params['obj_fun'] == 'arva':
        
        fun = fun_Objective_functions.fun_objective_arva_pytorch(params, Q_prev, Q)
        params["F_val"] = fun.detach().to("cpu").numpy()     #Obj func value
        
        #return fun.detach().to("cpu").numpy()
    
    return fun, params

