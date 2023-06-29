#Contains objective functions for portfolio optimization

from logging import raiseExceptions
import numpy as np
import torch


#MC added pytorch version of objective mean cvar
def objective_mean_cvar_pytorch(params, W_T_vector, xi):
    
    #Assign and evaluate objective function and its gradient w.r.t. terminal wealth
    # adapted for pytorch tensor format

    # OUTPUT: objective function value and its gradient with respect to terminal wealth
    #   return fun -- gradient is calculated automatically by pytorch 
    
    #   Note: fun.shape = grad_fun.shape = (terminal wealth vector).shape
    #only mean cvar implemented for pytorch so far
    
    # W_T_vector = params["W"][:, -1] #Last column of params["W"] is TERMINAL WEALTH

    # Pytorch objective function shortcut (mean cvar single)
    #Get info in params specific for mean-cvar
    rho = params["obj_fun_rho"]
    alpha = params["obj_fun_alpha"]
    # xi is passed from fun_train_SGD_algos to allow optimizer access.

    # assuming no lambda smoothing 
    # also W_T and xi already in tensor         
    
    
    xi_squared = torch.square(xi)
    # ind_W_T_below_xi_squared = (W_T_vector <= xi_squared) * 1   # 1 if W_T <=  xi_squared,
                                                            # 0 if W_T >  xi_squared

    zeros = torch.zeros(W_T_vector.size(), device = params["device"])
    
    bracket = torch.minimum(W_T_vector - xi_squared, zeros)
    
    # diff = W_T_vector - xi_squared
    # bracket = torch.multiply(ind_W_T_below_xi_squared, diff)   #same as: minimum(W_T_vector - xi_squared, 0)

    #obj Function
    fun = -(rho * W_T_vector) - xi_squared - (1/alpha)*bracket
    fun = torch.mean(fun)


    #return only fun
    return fun


def fun_objective_arva_pytorch(params, Q):
    Q_prev = Q.insert(0, 0)[0:-1]
    fun = np.sum(Q) - params["lam"] *np.sum(np.power(np.min(Q - Q_prev, 0),2))
    fun = torch.mean(fun)
    
    return fun
    
    


