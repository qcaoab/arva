#Contains objective functions for portfolio optimization

from logging import raiseExceptions
import numpy as np
import torch

def quad_smooth_min(x, lambda_quad, device):
    
    #smoothing helper function to replace max(x,0)
    
    lambda_v = torch.ones(x.size()[0], device=device)*lambda_quad
    quad_smooth = (1/(4*lambda_quad)) * torch.square(x) - 0.5*x + 0.25*lambda_quad
    
    return torch.where(torch.gt(x, lambda_v), 0, torch.where(torch.lt(x,-lambda_v), x, quad_smooth))

def objective_mean_cvar_decumulation(params, qsum_T_vector, W_T_vector, xi):
    #pytorch implementation for objective function: mean cvar with decumulation
       
    rho = params["obj_fun_rho"]
    alpha = params["obj_fun_alpha"]

    if params["smooth_cvar_func"]:
        
        bracket = xi + (1/alpha) * quad_smooth_min(W_T_vector - xi, params["lambda_quad"], params["device"])    
        
    else:

        bracket = xi + (1/alpha) * torch.minimum(W_T_vector - xi, torch.zeros(W_T_vector.size(), device = params["device"]))
    
    if not params["kappa_inf"]:
        fun = -qsum_T_vector - rho*bracket #formulate as minimization
                
        fun = fun - params["obj_fun_epsilon"]*W_T_vector  #stabilization
        fun = torch.mean(fun)
    else:
        fun = -bracket #formulate as minimization
                
        fun = fun - params["obj_fun_epsilon"]*W_T_vector  #stabilization
        fun = torch.mean(fun)

    #return only fun
    return fun



#MC added pytorch version of objective mean cvar NO DECUMULATION
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
    rho = params["obj_fun_rho"] #this is usually referred to as "\kappa" in Forsyth and Li's papers.
    alpha = params["obj_fun_alpha"]
    # xi is passed from fun_train_SGD_algos to allow optimizer access.

    # assuming no lambda smoothing 
    # also W_T and xi already in tensor         
    
    
    xi_squared = torch.square(xi)
    ind_W_T_below_xi_squared = (W_T_vector <= xi_squared) * 1   # 1 if W_T <=  xi_squared,
                                                            # 0 if W_T >  xi_squared

    diff = W_T_vector - xi_squared
    bracket = torch.multiply(ind_W_T_below_xi_squared, diff)   #same as: minimum(W_T_vector - xi_squared, 0)

    #obj Function
    fun = -rho * W_T_vector - xi_squared - (1/alpha)*bracket
    fun = torch.mean(fun)

    #return only fun
    return fun


