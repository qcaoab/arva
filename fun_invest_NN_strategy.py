import numpy as np
import pandas as pd
import fun_W_T_stats
import copy
import fun_utilities    #using smooth approx to abs value function
import torch 
from constraint_activations import w_custom_activation
from constraint_activations import asset_constraint_activation
from fun_construct_Feature_vector import construct_Feature_vector

def withdraw_invest_NN_strategy(NN_list, params):
    #OBJECTIVE: Using pytorch, Calculate the withdrawals, wealth paths, terminal wealth,
    # where NN_list  control/investment strategy in pytorch object, with theta parameters
    
    #OUTPUT in tensor format
    # returns dictionary 'params' as per the main code, with following ADDED keys + values:
    # params["W"]: contains paths of outcomes of W(t_N+) using investment strategy given by NN
    #              W.shape = (N_d, N_rb+1) since it contains paths, not returns+
    # params["W_paths_mean"]: contains MEAN of W(t_n+) across sample paths using NN strategy
    #                W.shape = (1, N_rb+1) since it is the mean of paths, not returns
    # params["W_paths_std"]: contains STANDARD DEVIATION of W(t_n+) across sample paths using NN strategy
    #                W.shape = (1, N_rb+1) since it is the mean of paths, not returns
    # params["NN_object"] = NN_object used to obtain the results, object of pytorch class NN, which gives control
    # params["W_T_stats_dict"] = W_T_stats_dict: summary W_T stats as a dictionary
    
    #params["q_matrix"] = paths of all withdrawals
    #params["qsum_T_vector"] = cumulative sum of all withdrawals
    
    
    # params["Feature_phi_paths"] = np.zeros([N_d, N_rb, N_phi])  #Paths for the (possibly standardized) feature values

    # params["NN_asset_prop_paths"].shape = [N_d, N_rb+1, N_a]  Paths for the proportions or wealth in each asset for given dataset
    #   params["NN_asset_prop_paths"][j, n, i]:
    #       for i <= N_a-1 (i is *index*): Proportion of wealth t_n^+ invested in asset i
    #       at rebal time t_n along sample path j

    #INPUTS:
    #   params = parameter dictionary 
    #   NN_object = object of class pytorch NN (which gives control) with structure as setup in main code
    
    # params = copy.deepcopy(params)
    
    # Append for output
    params["strategy_description"] = "invest_NN_strategy"

    #Define local copies for ease of reference
    N_rb = params["N_rb"]       # Nr of equally-spaced rebalancing events in [0,T]
    N_a = params["N_a"]         # Nr of assets = nr of output nodes
    N_phi = params["N_phi"]     # Nr of features, i.e. the number of input nodes
    N_d = params["N_d"]         # Nr of  data return sample paths
    W0 = params["W0"]           # Initial wealth W0
    q = params["q"]             # Cash injection schedule (a priori specified)
    T = params["T"]             #Terminal time
    delta_t = params["delta_t"] #time interval between rebalancing events
    # N_theta = NN_object.theta_length  #Nr of parameters (weights and biases) in NN

    #Do quick checks (adapted for pyt)
    
    #check return data right shapeparams["W"]
    #check number of output nodes 
    
    if params["dynamic_total_factorprop"] == True:
        node_adj = 1
    else:
        node_adj = 0
        
    if NN_list[1].model[-2].out_features != N_a + node_adj or NN_list[1].model[0].in_features != N_phi:
        raise ValueError("PVS error in 'fun_NN_terminal_wealth': NN not right shape.")

    if sum(params["q"].shape) != N_rb:
        raise ValueError("PVS error in 'fun_NN_terminal_wealth': cash injection schedule q "
                         "not right shape.")


    #Append fields to params, and initialize
    params["W"] = np.zeros([N_d, N_rb+1]) #   W contains PATHS, so W.shape = (N_d, N_rb+1)
    params["W_paths_mean"] = np.zeros([1, N_rb + 1])    #mean of W paths at each rebalancing time
    params["W_paths_std"] = np.zeros([1, N_rb + 1])     #stdev of W paths at each rebalancing time
    
    params["W_allocation"] = np.zeros([N_d, N_rb+1])

    params["Feature_phi_paths_withdrawal"] = np.zeros([N_d, N_rb+1, N_phi])  #Paths for the (possibly standardized) feature values
    params["Feature_phi_paths_allocation"] = np.zeros([N_d, N_rb, N_phi])  #Paths for the (possibly standardized) feature values
    params["NN_asset_prop_paths"] = np.zeros([N_d, N_rb, N_a+1])  #Paths for the proportions or wealth in each asset for given dataset

    # params["q_matrix"] = np.zeros([N_d, N_rb+1]) #withdrawals for all paths at each Rb step
    qsum_T_vector = torch.zeros([N_d], device = params["device"])     #cumsum of withdrawals for each path
    
    q_min = torch.tensor(params["q_min"], device= params["device"])
    # q_range = torch.tensor(params["q_max"] - params["q_min"], device= params["device"])
    q_max = torch.tensor(params["q_max"], device= params["device"])
    # ---------------------INITIALIZE values for timestepping loop -------------------
    g = W0 * torch.ones(N_d, requires_grad=True, device=params["device"])  # Initialize g for g_prev (initial wealth) below for first rebalancing time
                        # MC note: this is the initialization of the wealth at t=0 that will be used to run 
                        # the NN strategy. The timestepping loop over rb times will find a new 
                        # portfolio wealth  each step of loop, calculated applying the current NN strategy.
                        # The same NN is used at each timestep, which includes a forward pass and backprop
                        # step for each Rb time. 
                          
    
    #-------------------------------------------------------------------------------
    #   TIMESTEPPING
    #-------------------------------------------------------------------------------
                                #extra iteration for withdrawal only
    for n in np.arange(1,N_rb+1 +1,1): #loop over rebalancing events n = 1,...,N_rb, *N_rb +1** 
        
        n_index = n - 1 #index of rebalancing event in the data

        # ---------------------Assign previous values from loop -------------------
        # g = g(t_n) = Wealth at time (t_n_plus_1)^-
        #   g(t_N_rb) = terminal wealth, i.e. wealth at time (t_N_rb + delta_t) = T
        # g_prev = g(t_n_min_1) = Wealth at time (t_n)^-

        #Assign values from previous loop
        g_prev = g.clone() #g_prev = g(t_n_min_1) = wealth (t_n)^-
        
                
        # --------------------------- WEALTH (t_n^+), before withdrawal ---------------------------
        # g_prev will construct_Feature_vector
        #cash injection
        # g_prev = g_prev + q[n_index] #g_prev now contains W(t_n^+)

        params["W"][:,n_index] = g_prev.detach().cpu().numpy() #Update W to contain W(t_n^+)
        
        params["W_paths_mean"][0,n_index] = torch.mean(g_prev).detach().cpu().numpy()

        if torch.std(g_prev) > 0.0:
            params["W_paths_std"][0, n_index] = torch.std(g_prev, unbiased=True) #ddof=1 for (N_d -1) in denominator (bessels correction)
        else:
            params["W_paths_std"][0, n_index] = torch.std(g_prev)

        #--------------------------- CONSTRUCT FEATURE VECTOR and standardize, for withdrawal ---------------------------

        phi_1 = construct_Feature_vector(params = params,  # params dictionary as per MAIN code
                                 n = n,  # n is rebalancing event number n = 1,...,N_rb, used to calculate time-to-go
                                 wealth_n = g_prev,  # Wealth vector W(t_n^+), *after* contribution at t_n
                                                    # but *before* rebalancing at time t_n for (t_n, t_n+1)
                                 feature_calc_option= None,  # "None" matches my code.  Set calc_option = "matlab" to match matlab code
                                 withdraw= params["withdrawal_standardize"])

        for feature_index in np.arange(0,N_phi,1):  #loop over feature index
            params["Feature_phi_paths_withdrawal"][:,n_index,feature_index] =  phi_1[:,feature_index].detach().cpu().numpy()
            #    phi[j,i] = index i=0,...,(N_phi - 1) value of (standardized) feature i along sample path j

        # ---------------------------WITHDRAW---------------------------------------
        
        #get withdrawal NN output as value in [0,1], to multiply with the range.
        nn_out = torch.squeeze(NN_list[0].forward(phi_1))
        
        q_n = w_custom_activation(nn_out, g_prev, params)
        
                
        #save withdrawals
        # params["q_matrix"][:,n_index]  = q_n.detach().cpu().numpy()
        qsum_T_vector += q_n

        g_prev = g_prev - q_n
        
        #check if last withdrawal, break out of loop if so:
        if n == N_rb + 1:
            g =g_prev
            break
        
        
        # --------------------------- RETURNS FOR  (t_n^+, t_n+1^-) ---------------------------
        #Construct matrix from training data using the subset of returns for (t_n^+, t_n+1^-)
        #   params["Y"][j, n_index, i] = Return, along sample path j, over time period (t_n, t_n+1),
        #                           for asset i
        # Y_t_n = params["Y"][:, n_index, :]
        Y_t_n = torch.tensor(params["Y"][:, n_index, :], device=params["device"]) 
        # Y_t_n[j,i] = return for asset i, along sample path j, over time period (t_n, t_n+1)
        
        #wondering about effect of negative wealth on allocation NN

        #------------------------negative portfolios only incur borrowing cost (bond, not stock yield)----
        
        neg_indices = g_prev < 0 
        
        #to do: change this for multi asset problem
        if neg_indices.any():
            for i in range(Y_t_n.size()[1]):
                Y_t_n[neg_indices][:,i] = Y_t_n[neg_indices][:,params["b10_idx"]]  
        
        
        #--------------------------- CONSTRUCT FEATURE VECTOR and standardize, for allocation ---------------------------
        params["W_allocation"][:,n_index] = g_prev.detach().cpu().numpy() #Update W to contain W(t_n^+)
        
        phi_2 = construct_Feature_vector(params = params,  # params dictionary as per MAIN code
                                 n = n,  # n is rebalancing event number n = 1,...,N_rb, used to calculate time-to-go
                                 wealth_n = g_prev,  # Wealth vector W(t_n^+), *after* contribution at t_n
                                                    # but *before* rebalancing at time t_n for (t_n, t_n+1)
                                 feature_calc_option= None,  # "None" matches my code.  Set calc_option = "matlab" to match matlab code
                                 withdraw=False) #use 

        for feature_index in np.arange(0,N_phi,1):  #loop over feature index
            params["Feature_phi_paths_allocation"][:,n_index,feature_index] =  phi_2[:,feature_index].detach().cpu().numpy()
            #    phi[j,i] = index i=0,...,(N_phi - 1) value of (standardized) feature i along sample path j


        # ---------------------------ALLOCATION CONTROL  -----------------------------------------------
        #Get proportions to invest in each asset at time t_n^+
        #   a_t_n[j,i] = proportion to invest in asset i along sample path j
        
        if params["factor_constraint"]:
            a_t_n_output = asset_constraint_activation(torch.squeeze(NN_list[1].forward(phi_2)), params)
        else:
            a_t_n_output = torch.squeeze(NN_list[1].forward(phi_2))
        
        
            
        # --------------------------- PROPORTIONS INVESTED in EACH ASSET ALONG EACH PATH-------------------

        # params["NN_asset_prop_paths"].shape = [N_d, N_rb+1, N_a]  Paths for the proportions in each asset for given dataset
        #   params["NN_asset_prop_paths"][j, n, i]:
        #       for i <= N_a-1 (i is *index*): Proportion of wealth t_n^+ invested in asset i
        #       at rebal time t_n along sample path j
        # --- Note: params["q"] gives the *withdrawal* paths, so no need to add this to NN_asset_prop_paths

        for asset_index in np.arange(0,N_a,1):  #loop over asset index
            params["NN_asset_prop_paths"][:,n_index,asset_index] =  a_t_n_output[:,asset_index].detach().cpu().numpy()
            #    a_t_n_output[j,i] = index i=0,...,(N_a - 1) proportion to invest in asset i along sample path j
        params["NN_asset_prop_paths"][:,n_index,N_a] =  q_n.detach().cpu().numpy()

        # --------------------------- WEALTH (t_n+1^-) ---------------------------

        h_components = torch.multiply(a_t_n_output, Y_t_n)
        #  h_components[j,i] = a_t_n[j,i] * Y_t_n[j,i], where
        #       a_t_n[j,i] = proportion to invest in asset i over (t_n^+, t_n+1^-), along sample path j
        #       Y_t_n[j,i] = return for asset i, over time period (t_n, t_n+1), along sample path j

        h = torch.sum(h_components, axis=1)  # axis = 1 sums the COLUMS of h_components



        #Calculate wealth at (t_n+1^-)
        #       g = g(t_n) = Wealth at time (t_n+1)^-
        #       g_prev includes cash injection at time t_n
        g = torch.multiply(g_prev, h)

        #end: TIMESTEPPING
            
   #-------------------------------------------------------------------------------
    #Update terminal wealth
    g_np = g.detach().to('cpu').numpy()
    params["W"][:, N_rb] = g_np.copy()

    #Mean and std of paths
    params["W_paths_mean"][0, N_rb] = np.mean(g_np)

    if np.std(g_np) > 0.0: #to avoid errors with ddof
        params["W_paths_std"][0, N_rb] = np.std(g_np, ddof=1)  # ddof=1 for (N_d -1) in denominator
    else:
        params["W_paths_std"][0, N_rb] = np.std(g_np)


    # TERMINAL WEALTH: possible modification for possibly cash withdrawal at T^-
    W_T = g_np.copy()  #terminal wealth
    params["W_T"] = W_T.copy()

              
    return params, g, qsum_T_vector

def invest_NN_strategy_pyt(NN_pyt, params):
    
    #OBJECTIVE: Using pytorch, Calculate the  wealth paths, terminal wealth,
    # where NN_object is the control/investment strategy in pytorch object, with theta parameters
    
    #OUTPUT in tensor format
    # returns dictionary 'params' as per the main code, with following ADDED keys + values:
    # params["W"]: contains paths of outcomes of W(t_N+) using investment strategy given by NN
    #              W.shape = (N_d, N_rb+1) since it contains paths, not returns+
    # params["W_paths_mean"]: contains MEAN of W(t_n+) across sample paths using NN strategy
    #                W.shape = (1, N_rb+1) since it is the mean of paths, not returns
    # params["W_paths_std"]: contains STANDARD DEVIATION of W(t_n+) across sample paths using NN strategy
    #                W.shape = (1, N_rb+1) since it is the mean of paths, not returns
    # params["NN_object"] = NN_object used to obtain the results, object of pytorch class NN, which gives control
    # params["W_T_stats_dict"] = W_T_stats_dict: summary W_T stats as a dictionary
    
    
    # params["Feature_phi_paths"] = np.zeros([N_d, N_rb, N_phi])  #Paths for the (possibly standardized) feature values

    # params["NN_asset_prop_paths"].shape = [N_d, N_rb+1, N_a]  Paths for the proportions or wealth in each asset for given dataset
    #   params["NN_asset_prop_paths"][j, n, i]:
    #       for i <= N_a-1 (i is *index*): Proportion of wealth t_n^+ invested in asset i
    #       at rebal time t_n along sample path j

    #INPUTS:
    #   params = parameter dictionary 
    #   NN_object = object of class pytorch NN (which gives control) with structure as setup in main code
    
    params = copy.deepcopy(params)
    
    # Append for output
    params["strategy_description"] = "invest_NN_strategy"

    #Define local copies for ease of reference
    N_rb = params["N_rb"]       # Nr of equally-spaced rebalancing events in [0,T]
    N_a = params["N_a"]         # Nr of assets = nr of output nodes
    N_phi = params["N_phi"]     # Nr of features, i.e. the number of input nodes
    N_d = params["N_d"]         # Nr of  data return sample paths
    W0 = params["W0"]           # Initial wealth W0
    q = params["q"]             # Cash injection schedule (a priori specified)
    T = params["T"]             #Terminal time
    delta_t = params["delta_t"] #time interval between rebalancing events
    # N_theta = NN_object.theta_length  #Nr of parameters (weights and biases) in NN

    #Do quick checks (adapted for pyt)
    
    #check return data right shapeparams["W"]
    #check number of output nodes 
    if NN_pyt.model[-2].out_features != N_a or NN_pyt.model[0].in_features != N_phi:
        raise ValueError("PVS error in 'fun_NN_terminal_wealth': NN not right shape.")

    if sum(params["q"].shape) != N_rb:
        raise ValueError("PVS error in 'fun_NN_terminal_wealth': cash injection schedule q "
                         "not right shape.")


    #Append fields to params, and initialize
    params["W"] = np.zeros([N_d, N_rb+1]) #   W contains PATHS, so W.shape = (N_d, N_rb+1)
    params["W_paths_mean"] = np.zeros([1, N_rb + 1])    #mean of W paths at each rebalancing time
    params["W_paths_std"] = np.zeros([1, N_rb + 1])     #stdev of W paths at each rebalancing time

    params["Feature_phi_paths"] = np.zeros([N_d, N_rb, N_phi])  #Paths for the (possibly standardized) feature values
    params["NN_asset_prop_paths"] = np.zeros([N_d, N_rb, N_a])  #Paths for the proportions or wealth in each asset for given dataset

    
        # ---------------------INITIALIZE values for timestepping loop -------------------
    g = W0 * torch.ones(N_d, requires_grad=True, device=params["device"])  # Initialize g for g_prev (initial wealth) below for first rebalancing time
                        # MC note: this is the initialization of the wealth at t=0 that will be used to run 
                        # the NN strategy. The timestepping loop over rb times will find a new 
                        # portfolio wealth  each step of loop, calculated applying the current NN strategy.
                        # The same NN is used at each timestep, which includes a forward pass and backprop
                        # step for each Rb time. 
    
    
    #-------------------------------------------------------------------------------
    #   TIMESTEPPING
    #-------------------------------------------------------------------------------

    for n in np.arange(1,N_rb+1,1): #loop over rebalancing events n = 1,...,N_rb
        
        n_index = n - 1 #index of rebalancing event in the data

        # ---------------------Assign previous values from loop -------------------
        # g = g(t_n) = Wealth at time (t_n_plus_1)^-
        #   g(t_N_rb) = terminal wealth, i.e. wealth at time (t_N_rb + delta_t) = T
        # g_prev = g(t_n_min_1) = Wealth at time (t_n)^-

        #Assign values from previous loop
        g_prev = g.clone() #g_prev = g(t_n_min_1) = wealth (t_n)^-

        
        # --------------------------- RETURNS FOR  (t_n^+, t_n+1^-) ---------------------------
        #Construct matrix from training data using the subset of returns for (t_n^+, t_n+1^-)
        #   params["Y"][j, n_index, i] = Return, along sample path j, over time period (t_n, t_n+1),
        #                           for asset i
        Y_t_n = params["Y"][:, n_index, :]
        # Y_t_n[j,i] = return for asset i, along sample path j, over time period (t_n, t_n+1)



        # --------------------------- WEALTH (t_n^+) ---------------------------
        # g_prev will construct_Feature_vector
        #cash injection
        g_prev = g_prev + q[n_index] #g_prev now contains W(t_n^+)

        
        params["W"][:,n_index] = g_prev.detach().cpu().numpy() #Update W to contain W(t_n^+)
        params["W_paths_mean"][0,n_index] = torch.mean(g_prev)

        if torch.std(g_prev) > 0.0:
            params["W_paths_std"][0, n_index] = torch.std(g_prev, unbiased=True) #ddof=1 for (N_d -1) in denominator (bessels correction)
        else:
            params["W_paths_std"][0, n_index] = torch.std(g_prev)



        #--------------------------- CONSTRUCT FEATURE VECTOR and standardize ---------------------------


        phi = construct_Feature_vector(params = params,  # params dictionary as per MAIN code
                                 n = n,  # n is rebalancing event number n = 1,...,N_rb, used to calculate time-to-go
                                 wealth_n = g_prev,  # Wealth vector W(t_n^+), *after* contribution at t_n
                                                    # but *before* rebalancing at time t_n for (t_n, t_n+1)
                                 feature_calc_option= None  # "None" matches my code.  Set calc_option = "matlab" to match matlab code
                                 )

        for feature_index in np.arange(0,N_phi,1):  #loop over feature index
            params["Feature_phi_paths"][:,n_index,feature_index] =  phi[:,feature_index].detach().cpu().numpy()
            #    phi[j,i] = index i=0,...,(N_phi - 1) value of (standardized) feature i along sample path j


        # --------------------------- CONTROL  ---------------------------
        #Get proportions to invest in each asset at time t_n^+
        #   a_t_n[j,i] = proportion to invest in asset i along sample path j

        a_t_n_output = NN_pyt.forward(phi)
        
        
            # # output_Gradient == False: No need to keep track of the outputs
            # a_t_n_output, _, _ = NN_object.forward_propagation(phi=phi)


        # --------------------------- PROPORTIONS INVESTED in EACH ASSET ALONG EACH PATH-------------------

        # params["NN_asset_prop_paths"].shape = [N_d, N_rb+1, N_a]  Paths for the proportions in each asset for given dataset
        #   params["NN_asset_prop_paths"][j, n, i]:
        #       for i <= N_a-1 (i is *index*): Proportion of wealth t_n^+ invested in asset i
        #       at rebal time t_n along sample path j
        # --- Note: params["q"] gives the *withdrawal* paths, so no need to add this to NN_asset_prop_paths

        for asset_index in np.arange(0,N_a,1):  #loop over asset index
            params["NN_asset_prop_paths"][:,n_index,asset_index] =  a_t_n_output[:,asset_index].detach().cpu().numpy()
            #    a_t_n_output[j,i] = index i=0,...,(N_a - 1) proportion to invest in asset i along sample path j




        # --------------------------- WEALTH (t_n+1^-) ---------------------------

        h_components = torch.multiply(a_t_n_output, Y_t_n)
        #  h_components[j,i] = a_t_n[j,i] * Y_t_n[j,i], where
        #       a_t_n[j,i] = proportion to invest in asset i over (t_n^+, t_n+1^-), along sample path j
        #       Y_t_n[j,i] = return for asset i, over time period (t_n, t_n+1), along sample path j

        h = torch.sum(h_components, axis=1)  # axis = 1 sums the COLUMS of h_components



        #Calculate wealth at (t_n+1^-)
        #       g = g(t_n) = Wealth at time (t_n+1)^-
        #       g_prev includes cash injection at time t_n
        g = torch.multiply(g_prev, h)

            
    #end: TIMESTEPPING

    # -------------------------------------------------

    return params, g

def invest_NN_strategy(NN_theta, NN_object, params, output_Gradient = True,
                       LRP_for_NN_TrueFalse = False, PRP_TrueFalse = False):
    #OBJECTIVE: Calculate the  wealth paths, terminal wealth,
    # and gradient of terminal wealth w.r.t. NN_object parameters (weights and biases, i.e. parameter vector theta)
    # where NN_object is the control/investment strategy, and
    # NN_object.theta = NN_theta parameters is implemented


    #OUTPUT
    # returns dictionary 'params' as per the main code, with following ADDED keys + values:
    # params["W"]: contains paths of outcomes of W(t_N+) using investment strategy given by NN
    #              W.shape = (N_d, N_rb+1) since it contains paths, not returns+
    # params["W_paths_mean"]: contains MEAN of W(t_n+) across sample paths using NN strategy
    #                W.shape = (1, N_rb+1) since it is the mean of paths, not returns
    # params["W_paths_std"]: contains STANDARD DEVIATION of W(t_n+) across sample paths using NN strategy
    #                W.shape = (1, N_rb+1) since it is the mean of paths, not returns
    # params["W_T_grad_g_theta"] = (grad_g in notes) gradient of terminal wealth wrt parameters (theta vector) of NN
    #   - only contains values when output_Gradient == True, otherwise None
    # params["NN_object"] = NN_object used to obtain the results, object of class_Neural_Network (which gives control)
    # params["W_T_stats_dict"] = W_T_stats_dict: summary W_T stats as a dictionary

    # if params["TransCosts_TrueFalse"] is True, also adds:
    #   params["TransCosts_cum"] = vector of cum trans costs over [0,T], for each path
    #   params["TransCosts_cum_with_interest"] = vector of cum trans costs over [0,T} for each path, WITH interest
    #   params["TransCosts_cum_mean"] = np.mean(TransCosts_cum)
    #   params["TransCosts_cum_with_interest_mean"] = np.mean(TransCosts_cum_with_interest)

    # params["Feature_phi_paths"] = np.zeros([N_d, N_rb, N_phi])  #Paths for the (possibly standardized) feature values

    # params["NN_asset_prop_paths"].shape = [N_d, N_rb+1, N_a]  Paths for the proportions or wealth in each asset for given dataset
    #   params["NN_asset_prop_paths"][j, n, i]:
    #       for i <= N_a-1 (i is *index*): Proportion of wealth t_n^+ invested in asset i
    #       at rebal time t_n along sample path j

    #       IF LRP_for_NN_TrueFalse == True, also outputs
    # params["LRPscores"] is matrix of relevance scores, where params["LRPscores"][j, n, i] = relevance score,
    #                           along sample path j, at rebalancing time n, for feature i
    # params["PRPscores"] is matrix of relevance scores, where params["PRPscores"][j, n, i] = relevance score,
    #                           along sample path j, at rebalancing time n, for feature i



    #INPUTS:
    #   params = parameter dictionary as in _MAIN_QuadTarget_.py
    #   NN_object = object of class_Neural_Network (which gives control) with structure as setup in main code
    #   NN_theta = parameter vector (weights and biases) for NN_object
    #   LRP_for_NN_TrueFalse = TRUE if we want to do layerwise-relevance propagation to explain contribution of features
    #                           we want to use LRP_for_NN_TrueFalse = False during TRAINING
    #   PRP_TrueFalse = TRUE if we want to do PRP to explain contribution of features
    #                           we want to use PRP_for_NN_TrueFalse = False during TRAINING


    params = copy.deepcopy(params) #Create a copy
    
    # Append for output
    params["strategy_description"] = "invest_NN_strategy"

    # Assign weights matrices and bias vectors in NN_object using given value of NN_theta
    NN_object.theta = NN_theta
    NN_object.unpack_NN_parameters()

    #Define local copies for ease of reference
    N_rb = params["N_rb"]       # Nr of equally-spaced rebalancing events in [0,T]
    N_a = params["N_a"]         # Nr of assets = nr of output nodes
    N_phi = params["N_phi"]     # Nr of features, i.e. the number of input nodes
    N_d = params["N_d"]         # Nr of  data return sample paths
    W0 = params["W0"]           # Initial wealth W0
    q = params["q"]             # Cash injection schedule (a priori specified)
    T = params["T"]             #Terminal time
    delta_t = params["delta_t"] #time interval between rebalancing events
    N_theta = NN_object.theta_length  #Nr of parameters (weights and biases) in NN

    #Transaction cost parameters ---------------------------
    if params["TransCosts_TrueFalse"] is True:
        TransCosts_r_b = params["TransCosts_r_b"]
        TransCosts_propcost = params["TransCosts_propcost"]
        TransCosts_lambda = params["TransCosts_lambda"]

    # Confidence penalty parameters ---------------------------
    if params["ConfPenalty_TrueFalse"] is True:
        ConfPenalty_lambda = params["ConfPenalty_lambda"]   # weight (>0) on confidence penalty term; if == 0, then NO confidence penalty is applied
        ConfPenalty_n_H = params[ "ConfPenalty_n_H"]   # integer in {1,...,N_a}, where N_a is number of assets in params["asset_basket_id"]
        # only large (confident/undiversified) investments in assets {ConfPenalty_n_H,...,N_a}
        # will be penalized, *NOT* the other assets.
        # Generates runtime error if ConfPenalty_n_H > N_a


    #Do quick checks
    if params["Y"].shape != (N_d, N_rb, N_a):
        raise ValueError("PVS error in 'fun_NN_terminal_wealth': training data Y not right shape.")

    if NN_object.n_nodes_output != N_a or NN_object.n_nodes_input != N_phi:
        raise ValueError("PVS error in 'fun_NN_terminal_wealth': NN not right shape.")

    if sum(params["q"].shape) != N_rb:
        raise ValueError("PVS error in 'fun_NN_terminal_wealth': cash injection schedule q "
                         "not right shape.")


    #Append fields to params, and initialize
    params["W"] = np.zeros([N_d, N_rb+1]) #   W contains PATHS, so W.shape = (N_d, N_rb+1)
    params["W_paths_mean"] = np.zeros([1, N_rb + 1])    #mean of W paths at each rebalancing time
    params["W_paths_std"] = np.zeros([1, N_rb + 1])     #stdev of W paths at each rebalancing time

    params["Feature_phi_paths"] = np.zeros([N_d, N_rb, N_phi])  #Paths for the (possibly standardized) feature values
    params["NN_asset_prop_paths"] = np.zeros([N_d, N_rb, N_a])  #Paths for the proportions or wealth in each asset for given dataset

    params["W_T_grad_g_theta"] = None #only assigned values if output_Gradient == True



    # -------------------------------------------------------------------------------
    #Initialize relevance scores, if needed
    if LRP_for_NN_TrueFalse is True:
        params["LRPscores"] = np.zeros([N_d, N_rb, N_phi])    #params["LRPscores"][j, n, i] = relevance score,
                                                        #  along sample path j, at rebalancing time n, for feature i

    if PRP_TrueFalse is True:
        params["PRPscores"] = np.zeros([N_d, N_rb, N_phi])    #params["PRPscores"][j, n, i] = relevance score,
                                                        #  along sample path j, at rebalancing time n, for feature i


    # ---------------------INITIALIZE values for timestepping loop -------------------
    g = W0 * np.ones(N_d)  # Initialize g for g_prev below for first rebalancing time
    if output_Gradient is True:
        grad_g = np.zeros([N_d, N_theta])

        if params["obj_fun"] == "te_stochastic": #Need to keep track of the gradients at each t_n as well
            grad_g_TEcomponents = np.zeros([N_d, N_theta, N_rb+1])  #shape [N_d, N_theta, N_rb+1], gradient for each t_n including time T


    if params["ConfPenalty_TrueFalse"] is True:
        H_t_n = np.zeros(N_d)   #Vector of entropy of NN outputs along each sample path *up to and incl* time t_n

        #Initialize gradients
        if output_Gradient is True:
            grad_H_t_n = np.zeros([N_d, N_theta])


    if params["TransCosts_TrueFalse"] is True:
        # Matrix of amounts in each asset at time t_{n+1}^-
        Amounts_t_nplus1_min = np.zeros([N_d, N_a])
        Amounts_t_nplus1_min[:, 0] = W0  # Initial wealth assumed to be in the CASH account

        # Initialize Transaction costs due from PREVIOUS rebalancing event
        C_t_n = np.zeros(N_d)  # Initialize to make loop work out

        # Initialize cumulative transaction costs
        TransCosts_cum = np.zeros(N_d)  # Cumulative transaction costs initialize
        TransCosts_cum_with_interest = np.zeros(N_d)  # Cumulative transaction costs INCLUDING interest initialize


        #Initialize gradients
        if output_Gradient is True:
            grad_C_t_n = np.zeros([N_d, N_theta])
            grad_Amounts_t_nplus1_min = {}

            for asset_index in np.arange(0, N_a, 1):
                grad_Amounts_t_nplus1_min.update({asset_index: np.zeros([N_d, N_theta])})
    #END: if params["TransCosts_TrueFalse"] is True


    #-------------------------------------------------------------------------------
    #   TIMESTEPPING
    #-------------------------------------------------------------------------------

    for n in np.arange(1,N_rb+1,1): #loop over rebalancing events n = 1,...,N_rb
        n_index = n - 1 #index of rebalancing event in the data

        # ---------------------Assign previous values from loop -------------------
        # g = g(t_n) = Wealth at time (t_n_plus_1)^-
        #   g(t_N_rb) = terminal wealth, i.e. wealth at time (t_N_rb + delta_t) = T
        # g_prev = g(t_n_min_1) = Wealth at time (t_n)^-

        #Assign values from previous loop
        g_prev = g.copy() #g_prev = g(t_n_min_1) = wealth (t_n)^-

        if params["TransCosts_TrueFalse"] is True:
            Amounts_t_n_min = Amounts_t_nplus1_min.copy()  # Matrix of amounts in each asset at time t_n^-
            C_prev = C_t_n.copy()


        if params["ConfPenalty_TrueFalse"] is True:
            H_prev = H_t_n.copy()


        if output_Gradient is True:
            #Assign values from previous loop
            grad_g_prev = grad_g.copy()


            if params["obj_fun"] == "te_stochastic":   # If we need to keep track of the gradient at each t_n as well
                grad_g_TEcomponents[:,:, n_index] = grad_g.copy()


            if params["TransCosts_TrueFalse"] is True:
                grad_C_prev = grad_C_t_n.copy()
                grad_Amounts_t_n_min = copy.deepcopy(grad_Amounts_t_nplus1_min)  # Gradient of Amount at t_{n+1}^- in each asset


            if params["ConfPenalty_TrueFalse"] is True:
                grad_H_prev = grad_H_t_n.copy()




        # --------------------------- RETURNS FOR  (t_n^+, t_n+1^-) ---------------------------
        #Construct matrix from training data using the subset of returns for (t_n^+, t_n+1^-)
        #   params["Y"][j, n_index, i] = Return, along sample path j, over time period (t_n, t_n+1),
        #                           for asset i
        Y_t_n = params["Y"][:, n_index, :]
        # Y_t_n[j,i] = return for asset i, along sample path j, over time period (t_n, t_n+1)



        # --------------------------- WEALTH (t_n^+) ---------------------------
        # g_prev will contain W(t_n^+)

        # Update wealth at t_n^- with cash injection, pay transaction costs if needed

        if params["TransCosts_TrueFalse"] is True:
            g_prev = g_prev + q[n_index] - np.exp(TransCosts_r_b*delta_t)*C_prev  # g_prev now contains W(t_n^+)

        else:   #No transaction costs
            g_prev = g_prev + q[n_index] #g_prev now contains W(t_n^+)


        params["W"][:,n_index] = g_prev #Update W to contain W(t_n^+)
        params["W_paths_mean"][0,n_index] = np.mean(g_prev)

        if np.std(g_prev) > 0.0:
            params["W_paths_std"][0, n_index] = np.std(g_prev, ddof = 1) #ddof=1 for (N_d -1) in denominator
        else:
            params["W_paths_std"][0, n_index] = np.std(g_prev)



        #--------------------------- CONSTRUCT FEATURE VECTOR and standardize ---------------------------


        phi = construct_Feature_vector(params = params,  # params dictionary as per MAIN code
                                 n = n,  # n is rebalancing event number n = 1,...,N_rb, used to calculate time-to-go
                                 wealth_n = g_prev,  # Wealth vector W(t_n^+), *after* contribution at t_n
                                                    # but *before* rebalancing at time t_n for (t_n, t_n+1)
                                 feature_calc_option= None  # "None" matches my code.  Set calc_option = "matlab" to match matlab code
                                 )

        for feature_index in np.arange(0,N_phi,1):  #loop over feature index
            params["Feature_phi_paths"][:,n_index,feature_index] =  phi[:,feature_index]
            #    phi[j,i] = index i=0,...,(N_phi - 1) value of (standardized) feature i along sample path j


        # --------------------------- CONTROL  ---------------------------
        #Get proportions to invest in each asset at time t_n^+
        #   a_t_n[j,i] = proportion to invest in asset i along sample path j

        if output_Gradient is True:
            #Also output, from forward propagation:
            #   a_layers_all = {layer_id : a}, where a is output for layer_id
            #   z_output = matrix of weighted inputs (for each sample path) into each node of the output layer

            a_t_n_output, a_layers_all, z_layers_all = NN_object.forward_propagation(phi=phi)

        else:   
            # output_Gradient == False: No need to keep track of the outputs
            a_t_n_output, _, _ = NN_object.forward_propagation(phi=phi)


        # --------------------------- PROPORTIONS INVESTED in EACH ASSET ALONG EACH PATH-------------------

        # params["NN_asset_prop_paths"].shape = [N_d, N_rb+1, N_a]  Paths for the proportions in each asset for given dataset
        #   params["NN_asset_prop_paths"][j, n, i]:
        #       for i <= N_a-1 (i is *index*): Proportion of wealth t_n^+ invested in asset i
        #       at rebal time t_n along sample path j
        # --- Note: params["q"] gives the *withdrawal* paths, so no need to add this to NN_asset_prop_paths

        for asset_index in np.arange(0,N_a,1):  #loop over asset index
            params["NN_asset_prop_paths"][:,n_index,asset_index] =  a_t_n_output[:,asset_index]
            #    a_t_n_output[j,i] = index i=0,...,(N_a - 1) proportion to invest in asset i along sample path j



        # --------------------------- Do RELEVANCE PROPAGATION if required  ---------------------------
        #LRP = layerwise relevance propagation
        #PRP = Portfolio relevance propagation
        if (LRP_for_NN_TrueFalse is True) or (PRP_TrueFalse is True):

            #Also output, from forward propagation:
            #   a_layers_all = {layer_id : a}, where a is output for layer_id
            #   z_output = matrix of weighted inputs (for each sample path) into each node of the output layer

            a_t_n_output, a_layers_all, z_layers_all = NN_object.forward_propagation(phi=phi)


            # --------------- LRP ----------------
            if LRP_for_NN_TrueFalse is True:

                LRP_epsilon = params["LRP_for_NN_epsilon"]

                # Get relevance scores
                #    LRP_score_t_n[j,i] is the LRP relevance score, along sample path j for feature i
                LRP_score_t_n = NN_object.LRPscores_back_propagation(phi, a_layers_all, z_layers_all, LRP_epsilon)

                #Loop through features and populate  params["LRPscores"]
                #   params["LRPscores"][j, n, i] = relevance score, along sample path j, at rebalancing time n, for feature i
                for phi_index in np.arange(0,N_phi,1):
                    params["LRPscores"][:, n_index, phi_index] = LRP_score_t_n[:,phi_index].copy()


            # --------------- PRP ----------------
            if PRP_TrueFalse is True:

                PRP_eps_1 = params["PRP_eps_1"]     # >=0, for numerical stability + absorption of relevance
                PRP_eps_2 = params["PRP_eps_2"]     # >=0, for numerical stability + absorption of relevance

                # Get relevance scores
                #    PRP_score_t_n[j,i] is the PRP relevance score, along sample path j for feature i
                PRP_score_t_n = NN_object.PRPscores_back_propagation(phi, a_layers_all, PRP_eps_1, PRP_eps_2)

                #Loop through features and populate  params["PRPscores"] for output
                #   params["PRPscores"][j, n, i] = relevance score, along sample path j, at rebalancing time n, for feature i
                for phi_index in np.arange(0,N_phi,1):
                    params["PRPscores"][:, n_index, phi_index] = PRP_score_t_n[:,phi_index].copy()


        # --------------------------- CONFIDENCE PENALTY (if required) -----------
        if params["ConfPenalty_TrueFalse"] is True:
            log_a_t_n_output = np.log(a_t_n_output, out=np.zeros_like(a_t_n_output), where=(a_t_n_output != 0.))

            #hH is the entropy added for this timestep
            hH_components = np.multiply(a_t_n_output, log_a_t_n_output) #elementwise
            hH_components[:, 0:ConfPenalty_n_H-1] = 0.    #Sum only over cols [ConfPenalty_n_H ,N_a]
                                                        # so set cols with INDICES [0,,..,ConfPenalty_n_H-2 ] equal to zero

            hH = - np.sum(hH_components, axis=1)  # axis = 1 sums the COLUMS of hH_components
            #Note the MINUS in hH!

            #Update cumulate entropy up to this point
            H_t_n = H_prev + hH

            #Do confidence penalty gradient here, easier to keep things together
            if output_Gradient is True:
                minone_minus_log_a_t_n_output = - 1. - log_a_t_n_output

                grad_hH_a = np.zeros([N_d,N_a])  # gradient of h_H w.r.t. a (output of NN), input for backprop
                grad_hH_a[:, ConfPenalty_n_H-1:N_a] = minone_minus_log_a_t_n_output[:, ConfPenalty_n_H-1:N_a].copy()

                # -- BACKPROPAGATION to get grad_h_H (gradient of h_H w.r.t. parameters of NN
                grad_hH, _ = NN_object.back_propagation(grad_hH_a, a_layers_all, z_layers_all)

                #Update gradient of entropy
                grad_H_t_n = grad_H_prev + grad_hH



        # --------------------------- WEALTH (t_n+1^-) ---------------------------

        h_components = np.multiply(a_t_n_output, Y_t_n)
        #  h_components[j,i] = a_t_n[j,i] * Y_t_n[j,i], where
        #       a_t_n[j,i] = proportion to invest in asset i over (t_n^+, t_n+1^-), along sample path j
        #       Y_t_n[j,i] = return for asset i, over time period (t_n, t_n+1), along sample path j

        h = np.sum(h_components, axis=1)  # axis = 1 sums the COLUMS of h_components



        #Calculate wealth at (t_n+1^-)
        #       g = g(t_n) = Wealth at time (t_n+1)^-
        #       g_prev includes cash injection at time t_n
        g = np.multiply(g_prev, h)



        # --------------------------- TRANSACTION COSTS (t_n) ---------------------------

        if params["TransCosts_TrueFalse"] is True:

            #INITIALIZE:
            Amounts_t_n_plus = np.zeros([N_d, N_a])  # Amounts in each asset after rebal
            delta_Amounts_t_n = np.zeros([N_d, N_a])  # Changes in Amounts over rebal
            abs_delta_Amounts = np.zeros([N_d, N_a])  # Abs value of change in Amounts over rebal
            C_t_n = np.zeros(N_d)  # Transaction costs from THIS rebal event
            Amounts_t_nplus1_min =  np.zeros([N_d, N_a]) #amounts in each asset immediately before NEXT rebalancing event

            for asset_index in np.arange(0, N_a, 1):  # loop over asset index

                # Transaction cost calcs for this timestep
                Amounts_t_n_plus[:, asset_index] = np.multiply(g_prev, a_t_n_output[:,asset_index])  # Recall g_prev contains W(t_n^+)
                delta_Amounts_t_n[:, asset_index] = Amounts_t_n_plus[:, asset_index] - Amounts_t_n_min[:, asset_index]
                abs_delta_Amounts[:, asset_index], _ =  fun_utilities.abs_value_smooth_quad( x = delta_Amounts_t_n[:, asset_index],
                                                                                            lambda_param=TransCosts_lambda,
                                                                                            output_Gradient=False)

                # Calculate total transaction costs for THIS rebal event
                if asset_index > 0:  # Cash will always be asset_index == 0, and changes in cash incurs zero transaction costs
                    C_t_n = C_t_n + TransCosts_propcost * abs_delta_Amounts[:, asset_index]


                #Update amounts in each asset immediately before NEXT rebalancing event
                #   Amounts_t_nplus1_min = Matrix of amounts in each asset at time t_{n+1}^-
                Amounts_t_nplus1_min[:, asset_index] = np.multiply(Amounts_t_n_plus[:, asset_index], Y_t_n[:, asset_index])

            #END: Loop over assets

            #Update cumulative transaction costs: this is NOT lagged, can do this here
            TransCosts_cum = TransCosts_cum + C_t_n
            TransCosts_cum_with_interest = TransCosts_cum_with_interest + np.exp(TransCosts_r_b*delta_t)*C_t_n

        # --------------------------- GRADIENT CALCULATIONS ---------------------------
        if output_Gradient is True:

            #Notes:
            # grad_h_theta, grad_h_phi is calculated in NN_object.backpropagation
            # grad_g is calculated via recursion

            grad_h_a = Y_t_n.copy() #gradient of h w.r.t. a (output of NN), input for backprop

            #-- BACKPROPAGATION to get grad_h_theta, grad_h_phi (gradient of h w.r.t. 
            # parameters of NN and std wealth (phi)
            grad_h_theta, grad_h_phi = NN_object.back_propagation(grad_h_a, a_layers_all, z_layers_all)
            # MC note: add output from grad_h_phi above?

            #Update the derivative of wealth with respect to parameters of NN
            if n == 1:  #First rebalancing event, create grad_g
                grad_g = (W0 + q[n_index]) * grad_h_theta    #same, whether we have TCs or not

            else: #Other rebalancing events, build up grad_g, column by column
                for k in np.arange(0,N_theta,1):    #loop over number of parameters of NN

                    if params["TransCosts_TrueFalse"] is True:  #Need to update gradient with transaction cost gradient
                        # grad_g_prev contains gradient of W(t_n^+)
                        grad_g_prev[:, k] = grad_g_prev[:, k] - np.exp(TransCosts_r_b * delta_t) * grad_C_prev[:,k]

                    #Whether transaction costs or not (grad_g_prev is now gradient of W(t_n^+))
                    grad_g[:, k] = np.multiply(grad_g_prev[:, k], h) \
                                   + np.multiply(g_prev, grad_h_theta[:, k]) \
                                    + np.multiply(g_prev, np.multiply(grad_g_prev[:,k], grad_h_phi[:,1])) \
                                        * 1/params["benchmark_W_std_train"][:,n_index]



            #TRANSACTION COST gradients
            if params["TransCosts_TrueFalse"] is True:

                #INITIALIZE: these are all for a SPECIFIC n (rebal event), so need to be initialized only once!
                grad_C_t_n = np.zeros([N_d, N_theta])
                grad_Amounts_t_nplus1_min = {}  # Dictionary with grad_A_t_nplus1_min for each asset
                grad_psi = np.zeros([N_d, N_a])  # elementwise derivative of abs value function

                for asset_index in np.arange(0, N_a, 1):  # loop over asset index

                    #INITIALIZE: these are all for a SPECIFIC ASSET, so need to re-initialize each time!
                    grad_A_t_n_plus = np.zeros([N_d, N_theta])
                    grad_A_t_nplus1_min = np.zeros([N_d, N_theta])
                    grad_A_t_n_min = np.zeros([N_d, N_theta])
                    grad_delta_A_t_n = np.zeros([N_d, N_theta])

                    #Get elementwise derivatives of abs. value of change in amounts in each asset
                    _ , grad_psi[:, asset_index] = fun_utilities.abs_value_smooth_quad(
                                                                                x=delta_Amounts_t_n[:, asset_index],
                                                                                lambda_param=TransCosts_lambda,
                                                                                output_Gradient=True)

                    #Gradients of projection functions for this asset
                    grad_pi_a = np.zeros([N_d, N_a])
                    grad_pi_a[:, asset_index] = 1.

                    # -- BACKPROPAGATION to get grad_pi for this asset
                    grad_pi, _ = NN_object.back_propagation(grad_pi_a, a_layers_all, z_layers_all)


                    for k in np.arange(0,N_theta,1):    #loop over number of parameters of NN

                        #Gradient of Amount at t_n^+ in each asset;
                        #           Note that grad_g_prev is the gradient of W(t_n^+)
                        grad_A_t_n_plus[:, k] = np.multiply(a_t_n_output[:,asset_index], grad_g_prev[:, k]) + \
                                         np.multiply(g_prev, grad_pi[:,k])

                        grad_A_t_n_min[:,k] = grad_Amounts_t_n_min[asset_index][:,k]

                        grad_delta_A_t_n[:, k] = grad_A_t_n_plus[:, k] - grad_A_t_n_min[:,k]

                        # Gradient of Amount at t_{n+1}^- in each asset
                        grad_A_t_nplus1_min[:,k] = np.multiply(Y_t_n[:, asset_index], grad_A_t_n_plus[:, k])


                        if asset_index > 0:  # Cash will always be asset asset_index == 0, and changes in cash incurs zero transaction costs
                            grad_C_t_n[:,k] = grad_C_t_n[:,k] + \
                                              TransCosts_propcost * \
                                              np.multiply(grad_psi[:, asset_index], grad_delta_A_t_n[:, k])

                    #END: Loop over parameters of NN [but still WITHIN asset loop]

                    #Update gradient dictionary for this asset
                    grad_Amounts_t_nplus1_min.update({asset_index: grad_A_t_nplus1_min})

                #END: Loop over asset_index

            #END: if params["TransCosts_TrueFalse"] is True

    #end: TIMESTEPPING

    # -------------------------------------------------------------------------------
    # CONFIDENCE PENALTY terms append:
    if params["ConfPenalty_TrueFalse"] is True:
        params["ConfPenalty_H_T"] = H_t_n.copy()

        if output_Gradient is True:
            params["ConfPenalty_grad_H_T"] = grad_H_t_n.copy()


    # -------------------------------------------------------------------------------
    # TERMINAL WEALTH: needs adjustments for final Transaction cost payment
    if params["TransCosts_TrueFalse"] is True:
        g = g -  np.exp(TransCosts_r_b*delta_t)*C_t_n   #Need to subtract transaction costs from last rebal event

        #Update cumulative values
        params["TransCosts_cum"] = TransCosts_cum.copy()
        params["TransCosts_cum_with_interest"] = TransCosts_cum_with_interest.copy()
        params["TransCosts_cum_mean"] = np.mean(TransCosts_cum)
        params["TransCosts_cum_with_interest_mean"] = np.mean(TransCosts_cum_with_interest)

        if output_Gradient is True: #Update gradient of terminal wealth
            grad_g = grad_g -  np.exp(TransCosts_r_b*delta_t)* grad_C_t_n


    #-------------------------------------------------------------------------------
    #Update terminal wealth, g(t_N_rb) = W(T^-) with possible adjustment for TCs
    params["W"][:, N_rb] = g.copy()

    #Mean and std of paths
    params["W_paths_mean"][0, N_rb] = np.mean(g)

    if np.std(g) > 0.0: #to avoid errors with ddof
        params["W_paths_std"][0, N_rb] = np.std(g, ddof=1)  # ddof=1 for (N_d -1) in denominator
    else:
        params["W_paths_std"][0, N_rb] = np.std(g)


    # params["W_T_mean"] =  params["W_paths_mean"][0,-1]  # mean of terminal wealth
    # params["W_T_std"] =  params["W_paths_std"][0,-1]  # stdev of terminal wealth
    #

    # TERMINAL WEALTH: possible modification for possibly cash withdrawal at T^-
    W_T = g.copy()  #terminal wealth
    params["W_T"] = W_T.copy()

    #Write OVER params["W_T"] if we need to do the cash withdrawal case
    if params["obj_fun"] == "one_sided_quadratic_target_error": #only in this case
        if params["obj_fun_cashwithdrawal_TrueFalse"] is True:  #Check if we want values *after* cash withdrawal

            W_T_cashwithdraw = np.minimum(g, params["obj_fun_W_target"])
            params["W_T_cashwithdraw"] = W_T_cashwithdraw

            W_T = W_T_cashwithdraw.copy()   #Overwrite terminal wealth in this case

            params["W_T"] = W_T_cashwithdraw.copy()



    #Add summary stats to params dictionary (based on results AFTER cash withdrawal, if applicable)
    W_T_stats_dict = fun_W_T_stats.fun_W_T_summary_stats(W_T)
    #params.update(W_T_stats_dict)   #unpacks the dictionary "W_T_stats_dict" into "params"

    # Appends the whole dictionary as well, useful for outputting results
    del W_T_stats_dict["W_T_summary_stats"]     #don't need the summary stats pd.DataFrame in dictionary
    params["W_T_stats_dict"] = W_T_stats_dict


    params["NN_object"] = NN_object #also append NN object used to obtain the results

    #Update gradient of terminal wealth w.r.t. NN parameter vector theta (if required)
    if output_Gradient == True:
        params["W_T_grad_g_theta"] = grad_g.copy()

        if params["obj_fun"] == "te_stochastic": # If we need to keep track of the gradient at each t_n as well
            #First update with terminal wealth gradient:
            grad_g_TEcomponents[:,:, -1] = grad_g.copy()

            #Copy over to params
            params["grad_g_TEcomponents"] = grad_g_TEcomponents.copy()


    #Add Feature paths summary stats: [Mean,Stdev] for each feature, calculated over *all* paths and time periods
    Feature_phi_stats_dict = {}
    for feature_index in np.arange(0, N_phi, 1):  # loop over feature index
        key = "feature_" + params["feature_order"][feature_index] + "_mean_stdev"
        mean = np.mean(params["Feature_phi_paths"][:, :, feature_index])
        stdev = np.std(params["Feature_phi_paths"][:, :, feature_index], ddof=1)
        Feature_phi_stats_dict.update({key :[mean,stdev]})

    params["Feature_phi_stats_dict"] = Feature_phi_stats_dict.copy()    #Copy over


    return params

