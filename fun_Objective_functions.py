#Contains objective functions for portfolio optimization

from logging import raiseExceptions
import numpy as np
import torch

def objective_mean_cvar_decumulation(params, qsum_T_vector, W_T_vector, xi):
    #pytorch implementation for objective function: mean cvar with decumulation
       
    rho = params["obj_fun_rho"]
    alpha = params["obj_fun_alpha"]

    # xi is passed from fun_train_SGD_algos to allow optimizer access.

    # assuming no lambda smoothing 
    # also W_T, q_sum, and xi already in tensor         
    
    # xi_cubed = torch.pow(xi,3)
    # ind_W_T_below_xi_squared = (W_T_vector <= xi_squared) * 1   # 1 if W_T <=  xi_squared,
    #                                                         # 0 if W_T >  xi_squared

    # diff = W_T_vector - xi_squared
    # bracket = torch.multiply(ind_W_T_below_xi_squared, diff)   #same as: minimum(W_T_vector - xi_squared, 0)
    
    # xi_10 = torch.mul(xi,10) # multiply by 50 for 

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
    ind_W_T_below_xi_squared = (W_T_vector <= xi_squared) * 1   # 1 if W_T <=  xi_squared,
                                                            # 0 if W_T >  xi_squared

    diff = W_T_vector - xi_squared
    bracket = torch.multiply(ind_W_T_below_xi_squared, diff)   #same as: minimum(W_T_vector - xi_squared, 0)

    #obj Function
    fun = -rho * W_T_vector - xi_squared - (1/alpha)*bracket
    fun = torch.mean(fun)

    #return only fun
    return fun


def fun_objective(params, standardize=True):
    #Assign and evaluate objective function and its gradient w.r.t. terminal wealth

    # -> EXCEPTION: "te_stochastic" evaluates f_TE at every (N_d, N_rb+1),
    #               with aggregation (rebal time summation) done in fun_eval_objfun_NN_strategy

    # OUTPUT: objective function value and its gradient with respect to terminal wealth
    #   return fun, grad_fun

    # if objective is mean-cvar:
    #   return fun, grad_fun_w, grad_fun_xi, grad2_fun_xi2

    #   Note: fun.shape = grad_fun.shape = (terminal wealth vector).shape


    #INPUTS:
    # standardize = True: means objective is standardized e.g. by dividing it by (W_target)^2

    # params dictionary as setup in the main code
    # Code ALWAYS makes use of:
    #   params["obj_fun"] = STRING specifying the objective function
    #   params["W"][:, -1] = terminal wealth vector

    # Code CAN make use of other stuff in params, for example:
    #   params["W_target"] = target terminal wealth value used in quadratic target error objectives
    #   params["eps"] = small parameter "lambda" in my notes, e.g. 1e-06, used in one sided quadratic

    # also see parameters required for mean-cvar below


    obj_fun = params["obj_fun"]
    W_T_vector = params["W"][:, -1] #Last column of params["W"] is TERMINAL WEALTH
     

    if obj_fun == "ads_stochastic":  #Asymmetric distribution shaping (ADS) obj with STOCHASTIC benchmark

        #Get info in params specific for ads depending
        W_T_benchmark = params["benchmark_W_T_vector"]

        if "obj_fun_lambda_smooth" in params:
            lambda_smooth = params["obj_fun_lambda_smooth"]
        else:   #use un-smoothed version of objective
            lambda_smooth = 0.0

        ads_beta = params["obj_fun_ads_beta"]
        ads_T = params["T"]

        # Get function and gradient
        fun, grad_fun = ads_stochastic(W_T_vector, W_T_benchmark, ads_beta, ads_T, lambda_smooth, standardize)

        return fun, grad_fun

    elif obj_fun == "qd_stochastic": # Quadratic deviation from elevated target as in Forsyth (2021)
        #Get info in params specific for qd
        W_T_benchmark = params["benchmark_W_T_vector"]
        qd_beta = params["obj_fun_qd_beta"] #beta value >= 0
        qd_T = params["T"]

        # Get function and gradient
        fun, grad_fun = qd_stochastic(W_T_vector, W_T_benchmark, qd_beta,qd_T, standardize)

        return fun, grad_fun

    elif obj_fun == "ir_stochastic": #Dynamic Information ratio (IR) with STOCHASTIC benchmark

        #Get info in params specific for IR
        W_T_benchmark = params["benchmark_W_T_vector"]

        ir_s =  params["obj_fun_ir_s"]  #For benchmark outperformance calc
        ir_T  = params["T"]
        ir_gamma = params["obj_fun_ir_gamma"]

        # Get function and gradient
        fun, grad_fun = ir_stochastic(W_T_vector, W_T_benchmark, ir_s, ir_T, ir_gamma, standardize)

        return fun, grad_fun

    elif obj_fun == "te_stochastic": # TRACKING ERROR as in Forsyth (2021)
        #EXCEPTION compared to other objectives
        # -> Does NOT calculate a single objective function value for each path
        # -> Instead, assesses f_TE and grad_f_TE at every (N_d, N_rb+1)
        # -> Summation over rebalancing events will be done in fun_eval_objfun_NN_strategy

        #Benchmark W paths: contains paths of outcomes of W(t_n+) using constant prop strategy
        W_paths_benchmark = params["benchmark_W_paths"] #shape (N_d, N_rb+1) since it contains paths including terminal wealth

        # W paths: contains paths of outcomes of W(t_n+) using NN strategy
        W_paths = params["W"]   #shape (N_d, N_rb+1) since it contains paths including terminal wealth

        # Construct time grid, *INCLUDING* terminal time
        t_n_grid = np.arange(0, params["T"]+params["delta_t"], params["delta_t"])

        if params["obj_fun_te_beta_hat_CONSTANT_TrueFalse"] is False:
            # beta_hat = exp(beta*t_n) is a function of time:
            te_beta_hat = np.exp(params["obj_fun_te_beta"]*t_n_grid)
        else:  # if params["obj_fun_te_beta_hat_CONSTANT_TrueFalse"] is True
            # beta_hat just a constant value
            te_beta_hat = params["obj_fun_te_beta_hat"]*np.ones(shape = t_n_grid.shape)

        # Get function and gradient *SUM COMPONENTS* evaluated at each (N_d, N_rb+1)
        fun, grad_fun = te_stochastic(W_paths, W_paths_benchmark, te_beta_hat, standardize)

        return fun, grad_fun


    elif obj_fun == "ads":   #Asymmetric distribution shaping (ADS) objective function with CONSTANT benchmark

        #Get info in params specific for ads
        W_target = params["obj_fun_W_target"]

        if "obj_fun_lambda_smooth" in params:
            lambda_smooth = params["obj_fun_lambda_smooth"]
        else:   #use un-smoothed version of objective
            lambda_smooth = 0.0

        # Get function and gradient
        fun, grad_fun = ads(W_T_vector, W_target, lambda_smooth, standardize)
        return fun, grad_fun

    elif obj_fun == "huber_loss":   #Huber_loss objective function
        #Get info in params specific for huber_loss
        W_target = params["obj_fun_W_target"]
        huber_delta = params["obj_fun_huber_delta"]

        # Get function and gradient
        fun, grad_fun = huber_loss(W_T_vector, W_target, huber_delta, standardize)
        return fun, grad_fun

    elif obj_fun == "mean_cvar":  #Mean-CVAR using the BILEVEL formulation

        #Get info in params specific for mean-cvar
        rho = params["obj_fun_rho"]
        alpha = params["obj_fun_alpha"]
        xi = params["xi"]

        if "obj_fun_lambda_smooth" in params:
            lambda_smooth = params["obj_fun_lambda_smooth"]
        else:   #use un-smoothed version of objective
            lambda_smooth = 0.0

        #Get objective function and gradients
        fun, grad_fun_w, grad_fun_xi, grad2_fun_xi2 = mean_cvar(W_T_vector, rho, alpha, xi, lambda_smooth)

        return fun, grad_fun_w, grad_fun_xi, grad2_fun_xi2


    elif obj_fun == "mean_cvar_single_level":   #Mean-CVAR using the SINGLE LEVEL formulation

        #Get info in params specific for mean-cvar
        rho = params["obj_fun_rho"]
        alpha = params["obj_fun_alpha"]
        xi = params["xi"]

        if "obj_fun_lambda_smooth" in params:
            lambda_smooth = params["obj_fun_lambda_smooth"]
        else:   #use un-smoothed version of objective
            lambda_smooth = 0.0

        #Get objective function and gradients
        fun, grad_fun_w, grad_fun_xi = mean_cvar_single_level(W_T_vector, rho, alpha, xi, lambda_smooth)

        return fun, grad_fun_w, grad_fun_xi


    elif obj_fun == "meancvarLIKE_constant_wstar":  #NOT TRUE MEAN-CVAR!!

        #Get info in params specific for mean-cvar
        rho = params["obj_fun_rho"]
        alpha = params["obj_fun_alpha"]
        wstar = params["obj_fun_constant_wstar"]

        if "obj_fun_lambda_smooth" in params:
            lambda_smooth = params["obj_fun_lambda_smooth"]
        else:   #use un-smoothed version of objective
            lambda_smooth = 0.0

        # Get function and gradient
        fun, grad_fun = meancvarLIKE_constant_wstar(W_T_vector, rho, alpha, wstar, lambda_smooth)

        return fun, grad_fun


    elif obj_fun == "one_sided_quadratic_target_error":

        W_target = params["obj_fun_W_target"]
        eps = params["obj_fun_eps"]

        #Get function and gradient
        fun, grad_fun = one_sided_quadratic_target_error(W_T_vector, W_target, eps, standardize)
        return fun, grad_fun


    elif obj_fun == "quad_target_error":

        W_target = params["obj_fun_W_target"]

        # Get function and gradient
        fun, grad_fun = quad_target_error(W_T_vector, W_target, standardize)
        return fun, grad_fun


    elif obj_fun == "kelly_criterion":

        # Get function and gradient
        fun, grad_fun = kelly_criterion(W_T_vector)
        return fun, grad_fun

    else:
        raise ValueError("PVS error in 'fun_Objectives': specified objective function not coded.")

#End of function: fun_objective
    

def mean_cvar_single_level(W_T_vector, rho, alpha, xi, lambda_smooth):
   
    # Mean-CVAR objective for SINGLE level optimization problem

    # OUTPUT: objective function value and its gradients (if vectors, same size as W_T_vector)
    #     return fun, grad_fun_w, grad_fun_xi


    #   grad_fun_w = gradient of f w.r.t. wealth
    #   grad_fun_xi = gradient of f w.r.t. xi

    # INPUT:
    #   xi:     xi**2 = xi_squared is the candidate value for value-at-risk at level alpha
    #   W_T_vector = terminal wealth (can be a vector)
    #   rho = scalarization param multiplying MEAN
    #   alpha is level of CVAR, e.g. 0.01 or 0.05
    #   lambda_smooth is the smoothing factor for the piecewise quadratic approximation to the objective
    #   if lambda_smooth = 0, no smoothing is used


    xi_squared = xi**2

    if np.abs(lambda_smooth) <= 1e-9:   # if lambda_smooth is zero

        ind_W_T_below_xi_squared = (W_T_vector <= xi_squared) * 1   # 1 if W_T <=  xi_squared,
                                                                # 0 if W_T >  xi_squared

        diff = W_T_vector - xi_squared

        bracket = np.multiply(ind_W_T_below_xi_squared, diff)   #same as: np.minimum(W_T_vector - xi_squared, 0)

        #Function
        fun = -rho * W_T_vector - xi_squared - (1/alpha)*bracket
        
        # print("k*EW: ", np.mean(-rho * W_T_vector), "ES: ", np.mean(- xi_squared - (1/alpha)*bracket))
        

        #Gradients
        grad_fun_w = -(rho + (1/alpha)) * ind_W_T_below_xi_squared  \
                    -rho * (1 -ind_W_T_below_xi_squared )

        grad_fun_xi = -2 * xi *(1 - (1/alpha)) * ind_W_T_below_xi_squared  \
                    - 2 * xi * (1 -ind_W_T_below_xi_squared )


    else:  # if lambda_smooth is NOT zero
        
        
        #Get indicators of the range of W_T
        ind_W_T_below = (W_T_vector < (xi_squared - lambda_smooth)) * 1
        ind_W_T_above = (W_T_vector > (xi_squared + lambda_smooth)) * 1
        ind_W_T_middle = 1 - (ind_W_T_below + ind_W_T_above)

        diff = W_T_vector - xi_squared

        bracket = (1/lambda_smooth) * diff - 1

        #phi = Smoothed "max" function (using cont. diff., piecewise quadratic)

        phi_below = -1*diff
        phi_middle = 0.25 * np.multiply(diff, bracket - 1) + 0.25*lambda_smooth
        phi_above = 0

        phi = np.multiply( phi_below, ind_W_T_below) \
                + np.multiply( phi_middle , ind_W_T_middle )    \
                + np.multiply( phi_above , ind_W_T_above )


        #Function
        fun = -rho * W_T_vector - xi_squared + (1/alpha)*phi

        # print("k*EW: ", np.mean(-rho * W_T_vector), "ES: ", np.mean(- xi_squared + (1/alpha)*phi))
        

        # Gradients
        grad_fun_w_below = -rho - (1/alpha)
        grad_fun_w_middle = -rho + (1 / (2 * alpha)) * bracket
        grad_fun_w_above =  -rho

        grad_fun_w = np.multiply( grad_fun_w_below, ind_W_T_below) \
                    + np.multiply( grad_fun_w_middle , ind_W_T_middle )    \
                    + np.multiply( grad_fun_w_above , ind_W_T_above )

        #------------
        grad_fun_xi_below = -2 * xi *(1 - (1/alpha))
        grad_fun_xi_middle = -2 * xi * (1 + (1/ (2*alpha))*bracket )
        grad_fun_xi_above = -2 * xi

        grad_fun_xi = np.multiply( grad_fun_xi_below, ind_W_T_below)    \
                    + np.multiply( grad_fun_xi_middle , ind_W_T_middle )    \
                    + np.multiply( grad_fun_xi_above , ind_W_T_above )


    return fun, grad_fun_w, grad_fun_xi

def mean_cvar(W_T_vector, rho, alpha, xi, lambda_smooth, d2option = "unsmoothed"):
    # Mean-CVAR objective for BILEVEL optimization problem

    # OUTPUT: objective function value and its gradients (if vectors, same size as W_T_vector)
    #     return fun, grad_fun_w, grad_fun_xi, grad2_fun_xi2


    #   grad_fun_w = gradient of f w.r.t. wealth
    #   grad_fun_xi = gradient of f w.r.t. xi
    #   grad2_fun_xi2 = second-order derivative of f w.r.t. xi

    # INPUT:
    #   xi:     xi**2 = xi_squared is the candidate value for value-at-risk at level alpha
    #   W_T_vector = terminal wealth (can be a vector)
    #   rho = scalarization param multiplying MEAN
    #   alpha is level of CVAR, e.g. 0.01 or 0.05
    #   lambda_smooth is the smoothing factor for the piecewise quadratic approximation to the objective
    #   if lambda_smooth = 0, no smoothing is used

    #   d2option = "unsmoothed" only applies when lambda_smooth is NOT 0
    #   - calculates the piecewise grad2_fun_xi2 as if NO smoothing is being applied


    xi_squared = xi**2

    if np.abs(lambda_smooth) <= 1e-9:   # if lambda_smooth is zero

        ind_W_T_below_xi_squared = (W_T_vector <= xi_squared) * 1   # 1 if W_T <=  xi_squared,
                                                                # 0 if W_T >  xi_squared

        diff = W_T_vector - xi_squared

        bracket = np.multiply(ind_W_T_below_xi_squared, diff)   #same as: np.minimum(W_T_vector - xi_squared, 0)

        #Function
        fun = -rho * W_T_vector - xi_squared - (1/alpha)*bracket

        #Gradients
        grad_fun_w = -(rho + (1/alpha)) * ind_W_T_below_xi_squared  \
                    -rho * (1 -ind_W_T_below_xi_squared )

        grad_fun_xi = -2 * xi *(1 - (1/alpha)) * ind_W_T_below_xi_squared  \
                    - 2 * xi * (1 -ind_W_T_below_xi_squared )

        grad2_fun_xi2 = -2 *(1 - (1/alpha)) * ind_W_T_below_xi_squared  \
                    - 2 * (1 -ind_W_T_below_xi_squared )

    else:  # if lambda_smooth is NOT zero

        #Get indicators of the range of W_T
        ind_W_T_below = (W_T_vector < (xi_squared - lambda_smooth)) * 1
        ind_W_T_above = (W_T_vector > (xi_squared + lambda_smooth)) * 1
        ind_W_T_middle = 1 - (ind_W_T_below + ind_W_T_above)

        diff = W_T_vector - xi_squared

        bracket = (1/lambda_smooth) * diff - 1

        #phi = Smoothed "max" function (using cont. diff., piecewise quadratic)

        phi_below = -1*diff
        phi_middle = 0.25 * np.multiply(diff, bracket - 1) + 0.25*lambda_smooth
        phi_above = 0

        phi = np.multiply( phi_below, ind_W_T_below) \
                + np.multiply( phi_middle , ind_W_T_middle )    \
                + np.multiply( phi_above , ind_W_T_above )


        #Function
        fun = -rho * W_T_vector - xi_squared + (1/alpha)*phi

        # Gradients
        grad_fun_w_below = -rho - (1/alpha)
        grad_fun_w_middle = -rho + (1 / (2 * alpha)) * bracket
        grad_fun_w_above =  -rho

        grad_fun_w = np.multiply( grad_fun_w_below, ind_W_T_below) \
                    + np.multiply( grad_fun_w_middle , ind_W_T_middle )    \
                    + np.multiply( grad_fun_w_above , ind_W_T_above )

        #------------
        grad_fun_xi_below = -2 * xi *(1 - (1/alpha))
        grad_fun_xi_middle = -2 * xi * (1 + (1/ (2*alpha))*bracket )
        grad_fun_xi_above = -2 * xi

        grad_fun_xi = np.multiply( grad_fun_xi_below, ind_W_T_below)    \
                    + np.multiply( grad_fun_xi_middle , ind_W_T_middle )    \
                    + np.multiply( grad_fun_xi_above , ind_W_T_above )


        # ------------
        if d2option == "unsmoothed":
            #Calculate second derivative as if we did no smoothing

            ind_W_T_below_xi_squared = (W_T_vector <= xi_squared) * 1  # 1 if W_T <=  xi_squared,
            # 0 if W_T >  xi_squared
            grad2_fun_xi2 = -2 * (1 - (1 / alpha)) * ind_W_T_below_xi_squared \
                            - 2 * (1 - ind_W_T_below_xi_squared)

        else:   #d2option is NOT 'unsmoothed'
            grad2_fun_xi2_below = -2 *(1 - (1/alpha))
            grad2_fun_xi2_middle = -2 -(1/alpha)*bracket + (2*xi_squared)/(lambda_smooth*alpha)
            grad2_fun_xi2_above = -2

            grad2_fun_xi2 = np.multiply( grad2_fun_xi2_below, ind_W_T_below)    \
                            + np.multiply( grad2_fun_xi2_middle , ind_W_T_middle )    \
                            + np.multiply( grad2_fun_xi2_above , ind_W_T_above )


    return fun, grad_fun_w, grad_fun_xi, grad2_fun_xi2


def meancvarLIKE_constant_wstar(W_T_vector, rho, alpha, wstar, lambda_smooth):
    # Mean-CVAR-LIKE objective, but with CONSTANT "candidate VAR" given by wstar
    # - we do NOT solve for wstar here, so not true mean-CVAR!

    # OUTPUT: objective function value and its gradient with respect to wealth only (if vectors, same size as W_T_vector)
    #     return fun, grad_fun
    #   grad_fun = gradient of f w.r.t. wealth, there are NO additional variables, ONLY wealth

    # INPUT:
    #   wstar: CONSTANT, in usual mean-CVAR this would be candidate value for value-at-risk at level alpha
    #   W_T_vector = terminal wealth (can be a vector)
    #   rho = scalarization param multiplying MEAN
    #   alpha is level of CVAR, e.g. 0.01 or 0.05
    #   lambda_smooth is the smoothing factor for the piecewise quadratic approximation to the objective
    #   if lambda_smooth = 0, no smoothing is used


    if np.abs(lambda_smooth) <= 1e-9:   # if lambda_smooth is zero

        ind_W_T_below_wstar = (W_T_vector <= wstar) * 1   # 1 if W_T <=  wstar,
                                                                # 0 if W_T >  wstar

        diff = W_T_vector - wstar

        bracket = np.multiply(ind_W_T_below_wstar, diff)   #same as: np.minimum(W_T_vector - wstar, 0)

        #Function
        fun = -rho * W_T_vector - wstar - (1/alpha)*bracket

        #Gradients
        grad_fun = -(rho + (1/alpha)) * ind_W_T_below_wstar  \
                    -rho * (1 -ind_W_T_below_wstar )


    else:  # if lambda_smooth is NOT zero

        #Get indicators of the range of W_T
        ind_W_T_below = (W_T_vector < (wstar - lambda_smooth)) * 1
        ind_W_T_above = (W_T_vector > (wstar + lambda_smooth)) * 1
        ind_W_T_middle = 1 - (ind_W_T_below + ind_W_T_above)

        diff = W_T_vector - wstar

        bracket = (1/lambda_smooth) * diff - 1

        #phi = Smoothed "max" function (using cont. diff., piecewise quadratic)

        phi_below = -1*diff
        phi_middle = 0.25 * np.multiply(diff, bracket - 1) + 0.25*lambda_smooth
        phi_above = 0

        phi = np.multiply( phi_below, ind_W_T_below) \
                + np.multiply( phi_middle , ind_W_T_middle )    \
                + np.multiply( phi_above , ind_W_T_above )


        #Function
        fun = -rho * W_T_vector - wstar + (1/alpha)*phi

        # Gradients
        grad_fun_below = -rho - (1/alpha)
        grad_fun_middle = -rho + (1 / (2 * alpha)) * bracket
        grad_fun_above =  -rho

        grad_fun = np.multiply( grad_fun_below, ind_W_T_below) \
                    + np.multiply( grad_fun_middle , ind_W_T_middle )    \
                    + np.multiply( grad_fun_above , ind_W_T_above )


    return fun, grad_fun


def kelly_criterion(W_T_vector):
    #Kelly criterion: maximizing the expected log of terminal wealth (Sato (2019))
    # - we minimize expected value of -log(W(T))
    # - NO standardization here (yet)

    # OUTPUT: objective function value and its gradient (if vectors, same size as W_T_vector)
    #   return fun, grad_fun

    # Input:
    #   W_T_vector = terminal wealth (can be a vector)

    #Function
    fun = - np.log(W_T_vector)

    #Derivative
    grad_fun = - np.divide(1,W_T_vector)

    return fun, grad_fun

def one_sided_quadratic_target_error(W_T_vector, W_target, eps, standardize=True):
    # One-sided Quadratic target error objective function

    # OUTPUT: objective function value and its gradient (if vectors, same size as W_T_vector)
    #   return fun, grad_fun

    # Input:
    #   W_T_vector = terminal wealth (can be a vector)
    #   W_target = scalar
    #   eps = "lambda" in my notes, e.g. 1e-06
    #   standardize = True: means we divided objective by (W_target)^2


    ind_W_T_below_target = (W_T_vector <= W_target)*1   # 1 if W_T <=  W_target,
                                                      # 0 if W_T >  W_target

    diff = W_T_vector - W_target

    bracket = np.multiply(ind_W_T_below_target, diff)   #same as: np.minimum(W_T_vector - W_target, 0)

    #Function
    fun = 0.5 * np.power(bracket, 2) + eps*W_T_vector

    #Derivative
    grad_fun = np.multiply(diff + eps, ind_W_T_below_target) + \
               eps * (1 - ind_W_T_below_target)


    if standardize == True:  # standardize
        fun = (1 / (W_target ** 2)) * fun
        grad_fun = (1 / (W_target ** 2)) * grad_fun

    return fun, grad_fun



def quad_target_error(W_T_vector, W_target, standardize = True):
    #Quadratic target error objective function

    #OUTPUT: objective function value and its gradient (if vectors, same size as W_T_vector)
    #   return fun, grad_fun

    #Input:
    #   W_T_vector = terminal wealth (can be a vector)
    #   W_target = scalar
    #   standardize = True: means we divided objective by (W_target)^2

    #Function
    fun = 0.5* np.power(W_T_vector - W_target, 2)

    #Derivative
    grad_fun = W_T_vector - W_target

    if standardize == True: #standardize
        fun = (1/(W_target**2)) * fun
        grad_fun = (1 / (W_target ** 2)) * grad_fun


    return fun, grad_fun



def huber_loss(W_T_vector, W_target, huber_delta, standardize = True):
    #Huber loss function as objective

    #OUTPUT: objective function value and its gradient (if vectors, same size as W_T_vector)
    #   return fun, grad_fun

    #Input:
    #   W_T_vector = terminal wealth (can be a vector)
    #   W_target = scalar
    #   huber_delta is the +- slope far from the target (see notes)
    #   standardize = True: means we divided objective by (W_target)^2

    #Bounds
    LB = W_target - huber_delta
    UB = W_target + huber_delta

    #Get key difference
    diff = W_T_vector - W_target


    #indicators
    ind_W_T_below_LB = (W_T_vector < LB) * 1.0  # 1 if W_T < W_target - huber_delta
                                                # 0 otherwise

    ind_W_T_above_UB = (W_T_vector > UB) * 1.0  # 1 if W_T > W_target + huber_delta
                                                # 0 otherwise

    ind_W_T_middle = 1 - ind_W_T_below_LB - ind_W_T_above_UB


    #Function
    fun_below_LB = -huber_delta*diff - 0.5*(huber_delta**2)
    fun_above_UB =  huber_delta*diff - 0.5*(huber_delta**2)
    fun_middle = 0.5*np.power(diff, 2)

    fun = np.multiply(ind_W_T_below_LB, fun_below_LB) \
            + np.multiply(ind_W_T_above_UB, fun_above_UB) \
            + np.multiply(ind_W_T_middle, fun_middle)

    #Derivative
    grad_fun = np.multiply(ind_W_T_below_LB, -huber_delta) \
            + np.multiply(ind_W_T_above_UB, huber_delta) \
            + np.multiply(ind_W_T_middle, diff)

    if standardize == True: #standardize
        fun = (1/(W_target**2)) * fun
        grad_fun = (1 / (W_target ** 2)) * grad_fun


    return fun, grad_fun



def ads(W_T_vector, W_target, lambda_smooth, standardize = True):
    #Asymmetric distribution shaping (ADS) objective function
    # - Assuming *CONSTANT* W_target, from Ni, Li, Forsyth and Carroll (2020)

    #OUTPUT: objective function value and its gradient (if vectors, same size as W_T_vector)
    #   return fun, grad_fun

    #Input:
    #   W_T_vector = terminal wealth (can be a vector)
    #   W_target = scalar
    #   lambda_smooth is the smoothing factor for continuously differentiable approximation
    #   standardize = True: means we divided objective by (W_target)^2


    #Bounds
    LB = W_target - lambda_smooth
    UB = W_target + lambda_smooth

    #Get key difference
    diff = W_T_vector - W_target

    #indicators
    ind_W_T_below_LB = (W_T_vector < LB) * 1.0  # 1 if W_T < W_target - lambda_smooth
                                                # 0 otherwise

    ind_W_T_above_UB = (W_T_vector > UB) * 1.0  # 1 if W_T > W_target + lambda_smooth
                                                # 0 otherwise

    ind_W_T_middle = 1 - ind_W_T_below_LB - ind_W_T_above_UB

    #Function
    fun_below_LB = np.power(diff + lambda_smooth, 2)
    fun_above_UB = diff
    fun_middle = (1 / (4*lambda_smooth) )*np.power(diff, 2) + 0.5*diff + lambda_smooth/4

    fun = np.multiply(ind_W_T_below_LB, fun_below_LB) \
            + np.multiply(ind_W_T_above_UB, fun_above_UB) \
            + np.multiply(ind_W_T_middle, fun_middle)

    #Derivative
    grad_fun = np.multiply(ind_W_T_below_LB, 2*(diff + lambda_smooth) ) \
          + ind_W_T_above_UB  \
          + np.multiply(ind_W_T_middle, (1 / (2*lambda_smooth) )*diff + 0.5 )


    if standardize is True: #standardize
        fun = (1/(W_target**2)) * fun
        grad_fun = (1 / (W_target ** 2)) * grad_fun




    return fun, grad_fun


def ads_stochastic(W_T_vector, W_T_benchmark, ads_beta, ads_T, lambda_smooth, standardize = True):
    #Asymmetric distribution shaping (ADS) objective function
    # - Assuming *STOCHASTIC* W_target, from Ni, Li, Forsyth and Carroll (2020)

    #OUTPUT: objective function value and its gradient (if vectors, same size as W_T_vector)
    #   return fun, grad_fun

    #Input:
    #   W_T_vector = terminal wealth (can be a vector) of adaptive NN strategy
    #   W_T_benchmark = terminal wealth vector of benchmark, same shape as W_T_vector
    #   ads_beta and ads_T: beta is annual pre-determined out-performance spread:
    #           Elevated target: exp{ads_beta*ads_T}*W_T_benchmark
    #   lambda_smooth is the smoothing factor for continuously differentiable approximation
    #  ONLY used for GRADIENT: standardize = True: divide gradient by

    #Calculate (elevated) target vector
    W_target = np.exp(ads_beta*ads_T) * W_T_benchmark


    #Bounds
    LB = W_target - lambda_smooth
    UB = W_target + lambda_smooth

    #Get key difference
    diff = W_T_vector - W_target

    #indicators
    ind_W_T_below_LB = (W_T_vector < LB) * 1.0  # 1 if W_T < W_target - lambda_smooth
                                                # 0 otherwise

    ind_W_T_above_UB = (W_T_vector > UB) * 1.0  # 1 if W_T > W_target + lambda_smooth
                                                # 0 otherwise

    ind_W_T_middle = 1 - ind_W_T_below_LB - ind_W_T_above_UB

    #Function
    fun_below_LB = np.power(diff + lambda_smooth, 2)
    fun_above_UB = diff
    fun_middle = (1 / (4*lambda_smooth) )*np.power(diff, 2) + 0.5*diff + lambda_smooth/4

    fun = np.multiply(ind_W_T_below_LB, fun_below_LB) \
            + np.multiply(ind_W_T_above_UB, fun_above_UB) \
            + np.multiply(ind_W_T_middle, fun_middle)

    #Derivative
    grad_fun = np.multiply(ind_W_T_below_LB, 2*(diff + lambda_smooth) ) \
          + ind_W_T_above_UB  \
          + np.multiply(ind_W_T_middle, (1 / (2*lambda_smooth) )*diff + 0.5 )


    if standardize is True: #Standardize ONLY the GRADIENT, not the function value
        scale = np.mean(W_target)
        grad_fun = (1 / scale) * grad_fun   #Same scaling as used in the Matlab code


    return fun, grad_fun    #end: ads_stochastic


def qd_stochastic(W_T_vector, W_T_benchmark, qd_beta, qd_T, standardize = True):
    # Quadratic deviation from elevated target as in Forsyth (2021)
    # - Assuming *STOCHASTIC* benchmark
    # - Basically quadratic target minimization:  inf { ( W - qd_beta_hat*W_benchmark)^2  }

    #OUTPUT: objective function value and its gradient (if vectors, same size as W_T_vector)
    #   return fun, grad_fun

    #Input:
    #   W_T_vector = terminal wealth (can be a vector) of adaptive NN strategy
    #   W_T_benchmark = terminal wealth vector of benchmark, same shape as W_T_vector
    #   qd_beta: multiplier of benchmark is exp(qd_beta*T)
    #  NO standardization of objective

    #Calculate (elevated) target vector
    W_target = np.exp(qd_beta*qd_T) * W_T_benchmark


    #Get key difference
    diff = W_T_vector - W_target


    #Function
    fun = 0.5* np.power(diff, 2)

    #Derivative
    grad_fun = diff.copy()


    return fun, grad_fun    #end: qd_stochastic


def ir_stochastic(W_T_vector, W_T_benchmark, ir_s, ir_T, ir_gamma, standardize = True):
    #Dynamic Information Ratio (IR) objective function from Goetzmann et al (2002)
    # - Assuming *STOCHASTIC* benchmark
    # - Basically quadratic target minimization:  inf { ( W - W_benchmark - ir_gamma )^2  }

    #OUTPUT: objective function value and its gradient (if vectors, same size as W_T_vector)
    #   return fun, grad_fun

    #Input:
    #   W_T_vector = terminal wealth (can be a vector) of adaptive NN strategy
    #   W_T_benchmark = terminal wealth vector of benchmark, same shape as W_T_vector
    #   ir_s and ir_T: s is annual pre-determined out-performance spread:
    #           Elevated target: exp{ir_s*ir_T}*W_T_benchmark
    #  NO standardization of objective


    #Calculate (elevated) target vector
    W_target = np.exp(ir_s*ir_T) * W_T_benchmark


    #Get key difference
    diff = W_T_vector - W_target


    #Function
    fun = 0.5* np.power(diff - ir_gamma, 2)

    #Derivative
    grad_fun = diff - ir_gamma


    return fun, grad_fun    #end: ir_stochastic



def te_stochastic(W_paths, W_paths_benchmark, te_beta_hat, standardize = True):
    # TRACKING ERROR sum components compared to elevated benchmark as in Forsyth (2021)
    # - Assuming *STOCHASTIC* benchmark

    # EXCEPTION compared to other objectives
    # -> Does NOT calculate a single objective function value for each path
    # -> Instead, assesses f_TE and grad_f_TE at every (N_d, N_rb+1)
    # -> Summation over rebalancing events will be done in fun_eval_objfun_NN_strategy


    # - At each (N_d, N_rb+1), calculate { 0.5*( W - te_beta_hat*W_benchmark)^2 }  and its derivative

    #OUTPUT: objective function value and its gradient (if vectors, same size as W_T_vector)
    #   return fun, grad_fun

    #Input:
    # W paths: contains paths of outcomes of W(t_n+) using NN strategy
    #           shape (N_d, N_rb+1) since it contains paths including terminal wealth
    # W_paths_benchmark: Benchmark W paths of outcomes of W(t_n+) using constant prop strategy
    #           shape (N_d, N_rb+1) since it contains paths including terminal wealth
    # te_beta_hat = multiplier of benchmark at each point in time
    #               can be a function of time, but pre-computed above
    #               shape (N_rb+1, )    [same multiplier time-series for each of the N_d paths]
    #  NO standardization of objective

    # Get value of N_rb + 1
    length_t_n_grid = te_beta_hat.shape[0]

    #Check that dimensions work out
    if (W_paths.shape[1] != length_t_n_grid) or (W_paths_benchmark.shape[1] != length_t_n_grid):
        raise ValueError("PVS error in te_stochastic: dimensions need to align!")

    # Calculate (elevated) target vector at each rebalancing event
    W_target_paths = np.zeros(shape=W_paths.shape)    #initialize

    for n_index in np.arange(0, length_t_n_grid, 1):
        W_target_paths[:, n_index] = te_beta_hat[n_index] * W_paths_benchmark[:, n_index]


    #Get key difference
    diff = W_paths - W_target_paths


    # TRACKING ERROR sum components (np.power is done elementwise)
    fun = 0.5* np.power(diff, 2)

    # TRACKING ERROR Derivative sum components
    grad_fun = diff.copy()


    return fun, grad_fun    #end: te_stochastic
