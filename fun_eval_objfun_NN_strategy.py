import numpy as np
import fun_Objective_functions
import fun_invest_NN_strategy
import torch


def eval_obj_NN_strategy_pyt(NN_list, params, xi):
    
    #Objective: Calculates the pytorch (mean cvar only atm) objective function value F_val.
    #NOTE: through the use of the function fun_invest_NN_strategy.invest_NN_strategy_pyt below,
    #       this code also "invests" the NN control (pytorch) as investment strategy and updates "params"
    #       withdrawal NN functionality also added

    
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
    params, g, qsum_T_vector = fun_invest_NN_strategy.withdraw_invest_NN_strategy(NN_list, params)

    #Unpack F_theta

    if params["obj_fun"] == "mean_cvar_single_level":
        
        # xi currently initialized as tensor in driver code
        W_T_vector = g
        
        fun = fun_Objective_functions.objective_mean_cvar_decumulation(params, qsum_T_vector, W_T_vector, xi)
    
        #for output
        params["F_val"] = fun.detach().to("cpu").numpy()     #Obj func value
        params["xi"] = xi
        
    return fun, params

def eval_obj_NN_strategy(F_theta, NN_object, params, output_Gradient = False,
                         LRP_for_NN_TrueFalse = False, PRP_TrueFalse = False):
    #Objective: Calculates the specified objective function value F_val,
    # and its derivative F_grad_theta w.r.t NN_object parameters (weights, biases)
    # if NN_object is the control/investment strategy and NN_object.theta = NN_theta

    #NOTE: through the use of the function fun_invest_NN_strategy.invest_NN_strategy below,
    #       this code also "invests" the NN control as investment strategy and updates "params"

    #OUTPUTS:
    # return params, F_val, F_theta, F_grad_theta
    # params dictionary, with added fields:
    #      ALL fields added by the FUNCTION:  fun_invest_NN_strategy.invest_NN_strategy
    #     params["F_val"] = F_val     #Obj func value
    #                                  if params["obj_fun"] = "mean_cvar" this is the LAGRANGIAN

    #     params["F_theta"] = F_theta #Parameter vector theta of objective function and NN at which F_val is obtained
    #                if params["obj_fun"] = "mean_cvar" this is the *VECTOR* [NN_theta, xi, gamma]
    #                if params["obj_fun"] = "mean_cvar_single_level" this is the *VECTOR* [NN_theta, xi]
    #     params["NN_theta"]" = parameter vector of NN only

    # output_Gradient == True
    #     params["F_grad_theta"] = F_grad_theta #Gradient of F with respect to theta evaluated at F_theta

    #INPUTS:
    # F_theta = [NN_theta, extra_theta]
    #           NN_theta = parameter vector (weights and biases) for NN_object
    #           extra_theta = ADDITIONAL parameters for objective function used by e.g. mean-cvar
    #                   if params["obj_fun"] = "mean_cvar":
    #                           extra_theta = [xi, gamma] where
    #                                           xi: (xi**2) is candidate VAR
    #                                           gamma = Lagrange multiplier in CVAR objective for bilevel formulation
    #                  if params["obj_fun"] = "mean_cvar_single_level"
    #                           extra_theta = [xi]
    # NN_object = object of class_Neural_Network with structure as setup in main code
    # params = dictionary with investment parameters as set up in main code
    # LRP_for_NN_TrueFalse = TRUE if we want to do layerwise-relevance propagation to explain contribution of features
    #                       we want to use LRP_for_NN_TrueFalse = False during TRAINING
    # PRP_TrueFalse = TRUE if we want to do the PRP algorithm for explaining contribution of features




    # ---------------------Invest according to given NN_object with params NN_theta------------------------------


    #Unpack F_theta
    if params["obj_fun"] == "mean_cvar":
        NN_theta = F_theta[0:-2]    #NN_theta except last 2 entries
        xi = F_theta[-2]        # Second-last entry is xi, where (xi**2) is candidate VAR
        gamma = F_theta[-1]     # Lagrange multiplier

        #Make sure parameter dictionary is updated so that e.g. fun_Objective_functions can work correctly
        params["xi"] = xi
        params["gamma"] = gamma


    elif params["obj_fun"] == "mean_cvar_single_level":
        NN_theta = F_theta[0:-1]    #NN_theta except last entry
        xi = F_theta[-1]        # Last entry is xi, where (xi**2) is candidate VAR

        #Make sure parameter dictionary is updated so that e.g. fun_Objective_functions can work correctly
        params["xi"] = xi

    else:
        NN_theta = F_theta  #No additional variables


    #Calculate the  wealth paths, terminal wealth, and gradient of terminal wealth w.r.t.
    # NN_theta (= NN_object.theta) parameters if NN_object with parameters NN_theta is the control/investment strategy
    params = fun_invest_NN_strategy.invest_NN_strategy(NN_theta= NN_theta,
                                                       NN_object= NN_object,
                                                       params= params,
                                                       output_Gradient= output_Gradient,
                                                       LRP_for_NN_TrueFalse = LRP_for_NN_TrueFalse,
                                                       PRP_TrueFalse = PRP_TrueFalse)


    if output_Gradient is True:
        # Gradient of terminal wealth with respect to NN parameter vector theta
        grad_g_theta = params[ "W_T_grad_g_theta"]


    #-------------------------------------------------------------------------
    # Objective function eval at each terminal wealth outcome:

    #   (f_val, grad_f) is (obj func val, gradient of obj func w.r.t terminal wealth)
    #           evaluated at at each of the terminal wealth values in params["W"][:, -1],
    #           i.e. the last column of params["W"]
    #   f_val.shape = grad_f.shape = (N_d,), one entry for each of the training data sample paths



    #Function value is always the first returned value, so use np notation
    f_val  = fun_Objective_functions.fun_objective(params=params, standardize=True)[0]
    #   Note on dimensions:
    #   f_val and grad_f.shape = (N_d,),
    #   EXCEPT for obj_fun "te_stochastic", where f_val and grad_f.shape=(N_d, N_rb +1)




    #If mean_cvar, we need "grad_fun_xi" to calculate the Lagrangian (i.e. the objectve),
    #   even if we don't want the other gradients
    #   Not required if using just the "mean_cvar_single_level" formulation
    if params["obj_fun"] == "mean_cvar":
        _, _, grad_fun_xi, _ = \
            fun_Objective_functions.fun_objective(params=params)


    #Only if we need the gradients
    if output_Gradient is True:

        # note that grad_f = grad_f_w (i.e. gradient of f w.r.t. terminal wealth value w)

        if  "mean_cvar" in params["obj_fun"]: #both SINGLE and BILEVEL formulation
            if params["obj_fun"] == "mean_cvar":    #BILEVEL
                f_val, grad_f, grad_fun_xi, grad2_fun_xi2 = \
                    fun_Objective_functions.fun_objective(params=params)

            elif params["obj_fun"] == "mean_cvar_single_level": #SINGLE level
                f_val, grad_f, grad_fun_xi = \
                    fun_Objective_functions.fun_objective(params=params)

        else:   #Other objective functions

            f_val, grad_f = fun_Objective_functions.fun_objective(params=params, standardize=True)

            #   Note on dimensions:
            #   f_val and grad_f.shape = (N_d,),
            #   EXCEPT for obj_fun "te_stochastic", where f_val and grad_f.shape=(N_d, N_rb +1)


    # -------------------------------------------------------------------------
    # L2 weight regularization
    lambda_reg = params["lambda_reg"]  # relative contribution of norm penalty term
    x_length = NN_object.x_length  # Get total number of weights
    x_weights = np.zeros(NN_object.theta_length) #Initialize vector to total NN_theta length
    x_weights[0:x_length] = NN_theta[0:x_length]    #Update the first entries with weights, leave biases zero

    omega = 0.5* np.sum(np.power(x_weights,2))  #L2 weight regularization term
    omega_grad = x_weights

    # -------------------------------------------------------------------------
    # Get CONFIDENCE PENALTY terms, if applicable
    if params["ConfPenalty_TrueFalse"] is True:
        ConfPenalty_lambda =pytorch_mean_cvar_single_level
        if output_Gradient is True:
            grad_H_T = params["ConfPenalty_grad_H_T"]


    #-------------------------------------------------------------------------
    #Approximation to objective function of portf optim problem (expectation)
    #   in mean-cvar case this is the LAGRANGIAN!

    F_val = calc_core_F_val(f_val, params)

    # Add weight regularization
    F_val = F_val + lambda_reg * omega  # Obj func value + Weight regularization term

    # Add confidence penalty if applicable
    if params["ConfPenalty_TrueFalse"] is True:
        F_val = F_val - ConfPenalty_lambda * np.mean(H_T)

    # Do objective function-specific adjustments

    if params["obj_fun"] == "mean_cvar":  # BILEVEL formulation of cvar
        #LAGRANGIAN
        F_val = F_val - gamma * np.mean(grad_fun_xi)

    # -------------------------------------------------------------------------
    # Construct F_theta, the parameter vector where F_val is obtained

    # Initialize: Parameter vector of NN at which F_val is obtained
    F_theta = NN_object.theta

    # Do objective function-specific adjustments
    if params["obj_fun"] == "mean_cvar":  # BILEVEL formulation
        F_theta = np.concatenate([F_theta, [xi, gamma]])  # Parameter vector = [NN_theta, xi, gamma]

    elif params["obj_fun"] == "mean_cvar_single_level":  # SINGLE level formulation
        F_theta = np.concatenate([F_theta, [xi]])  # Parameter vector = [NN_theta, xi]


    #-------------------------------------------------------------------------
    # GRADIENT approximation to objective function grad w.r.t. NN parameters
    #           (and possibly additional parameters, e.g. for mean-cvar)
    F_grad_theta = None #Initialize for output, overwritten below if needed

    if output_Gradient is True:
        #NN part first
        F_grad_theta_NN = calc_core_F_grad_theta_NN(grad_f, grad_g_theta, params)

        #Add weight regularization
        F_grad_theta_NN = F_grad_theta_NN + lambda_reg * omega_grad  # Add weight regularization term

        # Add confidence penalty gradient if applicable
        if params["ConfPenalty_TrueFalse"] is True:
            F_grad_theta_NN = F_grad_theta_NN - ConfPenalty_lambda * np.mean(grad_H_T, axis=0)

        #Set result for output
        F_grad_theta = F_grad_theta_NN.copy()


        #Do objective function-specific adjustments
        if params["obj_fun"] == "mean_cvar":  # BILEVEL formulation
            #Add xi and Lagrange multiplier (gamma) parts of gradient of F
            F_grad_xi = np.mean(grad_fun_xi - gamma*grad2_fun_xi2)
            F_grad_gamma = np.mean(grad_fun_xi)

            #Gradient of F (LAGRANGIAN) with respect to theta evaluated at F_theta
            F_grad_theta = np.concatenate([F_grad_theta,[F_grad_xi,F_grad_gamma]])

        elif params["obj_fun"] == "mean_cvar_single_level":  # SINGLE level formulation

            #Add xi part of gradient of F
            F_grad_xi = np.mean(grad_fun_xi)

            #Combine for final result
            F_grad_theta = np.concatenate([F_grad_theta,[F_grad_xi]])


    #-------------------------------------------------------------------------
    #Append for output
    params["F_val"] = F_val     #Obj func value
    params["F_theta"] = F_theta #Parameter vector theta of objective function and NN at which F_val is obtained
    params["NN_theta"] = NN_object.theta    #Parameter vector theta of NN only
    params["F_grad_theta"] = F_grad_theta #Gradient of F with respect to theta evaluated at F_theta


    return params, F_val, F_theta, F_grad_theta

# ------------------------------------------------------------------
#Following functions are used to streamline the code

def calc_core_F_val(f_val, params):
    # OBJECTIVE: Calculate core obj function value, without weight reg. or conf penalty
    # Note on dimensions:
    #   f_val and grad_f.shape = (N_d,),
    #   EXCEPT for obj_fun "te_stochastic", where f_val and grad_f.shape=(N_d, N_rb +1)

    if params["obj_fun"] == "te_stochastic":
        sum_f_val = np.zeros(params["N_d"])    #initialize

        for n_index in np.arange(0, params["N_rb"] + 1, 1):     #INCLUDE terminal time T
            sum_f_val = sum_f_val + f_val[:,n_index]    #Cumulate sum over rebalancing time periods

        core_F_val = np.mean(sum_f_val) # (1/N_d)*sum(sum_f_val)

    else: #ALL other objectives
        core_F_val = np.mean(f_val) # (1/N_d)*sum(f_val)

    return core_F_val


def calc_core_F_grad_theta_NN(grad_f, grad_g_theta, params):
    # OBJECTIVE: Calculate core obj function gradient wrt NN params, without weight reg. or conf penalty
    # Note on dimensions:
    #   f_val and grad_f.shape = (N_d,),
    #   EXCEPT for obj_fun "te_stochastic", where f_val and grad_f.shape=(N_d, N_rb +1)

    if params["obj_fun"] == "te_stochastic":
        N_d = params["N_d"]  # Nr of training data return sample paths

        # Get the gradients at each t_n:
        grad_g_TEcomponents = params["grad_g_TEcomponents"].copy()
        # grad_g_TEcomponents: shape [N_d, N_theta, N_rb+1], gradient for each t_n(-1) including time T

        # Double check:
        if np.array_equal(grad_g_TEcomponents[:, :, -1], grad_g_theta) is False:
            raise ValueError("PVS error in eval_objfun_NN: TE gradient components gone wrong.")


        for n_index in np.arange(0, params["N_rb"] + 1, 1):     #INCLUDE terminal time T
            grad_f_t_n = grad_f[:, n_index].copy() #gradient of f_TE for THIS t_n
            grad_g_t_nmin1 = grad_g_TEcomponents[:,:, n_index].copy()   #grad_g for t_n_min1

            # Add dimension for matrix multiplication
            grad_f_t_n = np.expand_dims(grad_f_t_n, axis=1)  # changes shape from (N_d,) to (N_d,1)

            #Multiply for this t_n
            term_t_n = (1 / N_d) * np.matmul(np.transpose(grad_f_t_n), grad_g_t_nmin1)
            term_t_n = np.squeeze(term_t_n)   # remove extra dimension

            #Accumulate
            if n_index == 0:
                #create
                cumsum_grad = term_t_n.copy()

            else: #ACCUMULATE over rebalancing events
                cumsum_grad = cumsum_grad + term_t_n    #Cumulate sum over rebalancing time periods

        core_F_grad_theta_NN = cumsum_grad.copy()


    else:  # ALL other objectives
        # Add dimension for matrix multiplication
        grad_f = np.expand_dims(grad_f, axis=1)  # changes shape from (N_d,) to (N_d,1)

        # Gradient of F with respect to theta evaluated at F_theta
        N_d = params["N_d"]  # Nr of training data return sample paths
        core_F_grad_theta_NN = (1 / N_d) * np.matmul(np.transpose(grad_f), grad_g_theta)
        core_F_grad_theta_NN = np.squeeze(core_F_grad_theta_NN)  # remove extra dimension

    return core_F_grad_theta_NN



# ------------------------------------------------------------------
#Following functions is used for scipy algorithms

def fun_val(F_theta, NN_object, params):
    # Returns ONLY the objective function value at NN_theta
    _, F_val, _, _ = eval_obj_NN_strategy(F_theta, NN_object, params, output_Gradient=False)

    return F_val

def fun_gradient(F_theta, NN_object, params):
    # Returns ONLY the gradient at NN_theta
    _, _, _, F_grad_theta = eval_obj_NN_strategy(F_theta, NN_object, params, output_Gradient=True)

    return F_grad_theta

