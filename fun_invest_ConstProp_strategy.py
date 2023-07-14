import numpy as np
import pandas as pd
import fun_Data__assign
import fun_utilities    #used to get smooth approx to absolute value function
import copy

def invest_ConstProp_strategy(prop_const, params, train_test_Flag = "train"):
    #Computes wealth using constant proportion strategy

    #OUTPUT
    # params dictionary as setup in the main code
    # params["W"]: contains paths of outcomes of W(t_n+) using constant prop strategy
    #              W.shape = (N_d, N_rb+1) since it contains paths, not returns
    #  params["W_T"] = terminal wealth as a vector (one entry for each path)

    # if params["TransCosts_TrueFalse"] is True, also adds:
    #   params["TransCosts_cum"] = vector of cum trans costs over [0,T], for each path
    #   params["TransCosts_cum_with_interest"] = vector of cum trans costs over [0,T} for each path, WITH interest
    #   params["TransCosts_cum_mean"] = np.mean(TransCosts_cum)
    #   params["TransCosts_cum_with_interest_mean"] = np.mean(TransCosts_cum_with_interest)

    # params["W_paths_mean"]: contains MEAN of W(t_n+) across sample paths using constant prop strategy
    #                W.shape = (1, N_rb+1) since it is the mean of paths, not returns
    # params["W_paths_std"]: contains STANDARD DEVIATION of W(t_n+) across sample paths using constant prop strategy
    #                W.shape = (1, N_rb+1) since it is the mean of paths, not returns

    #Summary stats for terminal wealth:
    #       params["W_T_summary_stats"]  = pandas.DataFrame with the summary results
    #       params["W_T_mean"] = mean of W_T
    #       params["W_T_median"] = median of W_T
    #       params["W_T_std"] = standard deviation of W_T
    #       params["W_T_pctile_1st"] = 1st percentile of W_T
    #       params["W_T_pctile_5th"]  = 5th percentile of W_T
    #       params["W_T_CVAR_1_pct"] = 1% CVAR
    #       params["W_T_CVAR_5_pct"] = 5% CVAR


    #INPUTS:
    #   prop_const = vector with constant proportions to invest in each asset over [0,T]
    #   prop_const[i] = constant proportion to invest in asset index i \in {0,1,...,N_a -1} over [0,T]
    #                   Order corresponds to asset index in sample return paths in params["Y"][:, :, i]
    #   train_test_Flag = "train" or "test". By default we do this on the training data set


    params = copy.deepcopy(params) #Create a copy

    # Set values of params["N_d"], params["Y"] and params["TradSig"] populated with train or test values
    params = fun_Data__assign.set_training_testing_data(train_test_Flag, params,
                                                        called_from = "invest_ConstProp_strategy")

    #Get stuff for convenience
    N_d = params["N_d"]
    N_a = params["N_a"]

    # Append for output
    params["strategy_description"] = "invest_ConstProp_strategy"
    params["prop_const"] = prop_const


    #Check that prop_const sums to 1, and that it is the correct size
    if np.abs(sum(prop_const)-1) >= 1e-06:
        raise ValueError("PVS error: Vector prop_const of constant proportions does not add to 1.")

    if len(prop_const) != params["N_a"]:
        raise ValueError("PVS error: Vector prop_const does not specify proportion for each of 'N_a' assets.")

    #initialize wealth at t_n^-
    W_end = params["W0"] * np.ones(N_d)

    #Append wealth W outcomes to params, and initialize
    params["W"] = np.zeros([N_d, params["N_rb"]+1])
    #   W contains PATHS, so W.shape = (N_d, N_rb+1)
    params["W_paths_mean"] = np.zeros([1, params["N_rb"] + 1])
    params["W_paths_std"] = np.zeros([1, params["N_rb"] + 1])

    #---------------------------------------------------------------------------
    # TRANSACTION COSTS: Initialize
    if params["TransCosts_TrueFalse"] is True:

        #For convenience get these
        TransCosts_r_b = params["TransCosts_r_b"]
        TransCosts_propcost = params["TransCosts_propcost"]
        TransCosts_lambda = params["TransCosts_lambda"]
        delta_t = params["delta_t"]

        #Matrix of amounts in each asset at time t_n^-
        Amounts_end = np.zeros([N_d,N_a])
        Amounts_end[:,0] =  params["W0"]   #Initial wealth assumed to be in the CASH account

        #Transaction costs due from PREVIOUS rebalancing event
        C_due = np.zeros(N_d)  #Initialize to make loop work out

        #Initialize cumulative transaction costs
        TransCosts_cum = C_due.copy()    #Cumulative transaction costs initialize
        TransCosts_cum_with_interest = C_due.copy() #Cumulative transaction costs INCLUDING interest initialize

    #---------------------------------------------------------------------------
    # TIMESTEPPING
    for n_index in np.arange(0, params["N_rb"],1): #Loop over time periods (columns)
        #n_index = n - 1

        if params["TransCosts_TrueFalse"] is True:
            # Add cash injection to get wealth at t_{n-1}^+
            # - repay borrowed transaction costs from PREVIOUS rebalancing event with interest
            W_start = W_end + params["q"][n_index] - np.exp(TransCosts_r_b*delta_t)*C_due

            #Update cumulative transaction costs
            TransCosts_cum = TransCosts_cum + C_due
            TransCosts_cum_with_interest = TransCosts_cum_with_interest + np.exp(TransCosts_r_b*delta_t)*C_due

        else:   #No transaction costs
            #Add cash injection to get wealth at t_{n-1}^+
            W_start = W_end + params["q"][n_index]

        #Update W to contain W(t_n+)
        params["W"][:,n_index] = W_start
        params["W_paths_mean"][:,n_index] = np.mean(W_start)

        if np.std(W_start) > 0.0:   #to correct for problems if only one data path
            params["W_paths_std"][:, n_index] = np.std(W_start, ddof = 1) #ddof=1 for (N_d -1) in denominator
        else:
            params["W_paths_std"][:, n_index] = np.std(W_start)

        # Update W_end using prop_const[k] and return paths in params["Y"][:,:,k]
        W_end = np.zeros(W_start.shape) #Initialize

        #Initialize Transaction Costs stuff
        if params["TransCosts_TrueFalse"] is True:
            Amounts_plus = np.zeros([N_d, N_a])  # Initialize amount in each asset *after* rebalancing
            delta_Amounts = np.zeros([N_d, N_a])  # Change in amount invested in each asset
            abs_delta_Amounts = np.zeros([N_d, N_a])  # Abs value of change in Amounts
            C_due = np.zeros(N_d)  # Transaction costs from THIS rebal event


        #Loop over assets
        for i in np.arange(0,params["N_a"],1):
            # params["Y"][:, n, i] = Return, along sample paths :, over time period (t_n, t_n+1), for asset i
            W_end = W_end + prop_const[i] * np.multiply(W_start, params["Y"][:,n_index,i])

            if params["TransCosts_TrueFalse"] is True:
                #Calculate amount in each asset *after* rebalancing
                Amounts_plus[:,i] = prop_const[i] * W_start
                delta_Amounts[:,i] = Amounts_plus[:,i] - Amounts_end[:,i]
                abs_delta_Amounts[:,i], _ = fun_utilities.abs_value_smooth_quad(x = delta_Amounts[:,i],
                                                                             lambda_param = TransCosts_lambda,
                                                                             output_Gradient = False)
                #Calculate cumulative transaction costs for THIS rebal event
                if i>0: #Cash will always be asset i == 0, and changes in cash incurs zero transaction costs
                    C_due = C_due + TransCosts_propcost*abs_delta_Amounts[:,i]

                #Update Amounts before *NEXT* rebal event
                Amounts_end[:,i] =   np.multiply(Amounts_plus[:,i], params["Y"][:,n_index,i])

        #END: Loop over assets

    #END: Loop over rebalancing events

    #Pay transaction costs from final rebalancing event, and update cumulative TCs
    if params["TransCosts_TrueFalse"] is True:
        W_end = W_end - np.exp(TransCosts_r_b*delta_t)*C_due

        # Update cumulative transaction costs
        TransCosts_cum = TransCosts_cum + C_due
        TransCosts_cum_with_interest = TransCosts_cum_with_interest + np.exp(TransCosts_r_b * delta_t) * C_due



    #Finally, update terminal wealth (wealth at t_N_rb^-
    params["W"][:, params["N_rb"]] = W_end
    params["W_T"] = W_end.copy()    # output terminal wealth vector

    #Update cumulative transaction costs if applicable
    if params["TransCosts_TrueFalse"] is True:
        params["TransCosts_cum"] = TransCosts_cum.copy()
        params["TransCosts_cum_with_interest"] = TransCosts_cum_with_interest.copy()
        params["TransCosts_cum_mean"] = np.mean(TransCosts_cum)
        params["TransCosts_cum_with_interest_mean"] = np.mean(TransCosts_cum_with_interest)

    #Update the path statistics
    params["W_paths_mean"][:, params["N_rb"]] = np.mean(W_end)

    if np.std(W_end) > 0.0:
        params["W_paths_std"][:, params["N_rb"]] = np.std(W_end, ddof=1)  # ddof=1 for (N_d -1) in denominator
    else:
        params["W_paths_std"][:, params["N_rb"]] = np.std(W_end)


    import fun_W_T_stats
    W_T_stats_dict = fun_W_T_stats.fun_W_T_summary_stats(W_end)

    #Add summary stats to params dictionary
    params.update(W_T_stats_dict) #unpacks the dictionary "W_T_stats_dict" into "params"

    # Appends the whole dictionary as well, useful for outputting results
    del W_T_stats_dict["W_T_summary_stats"]     #don't need the summary stats pd.DataFrame in dictionary
    params["W_T_stats_dict"] = W_T_stats_dict
    
    import fun_Q_T_stats
    Q_T_stats_dict = fun_Q_T_stats.fun_Q_T_summary_stats(W_end)

    #Add summary stats to params dictionary
    params.update(Q_T_stats_dict) #unpacks the dictionary "W_T_stats_dict" into "params"

    # Appends the whole dictionary as well, useful for outputting results
    del Q_T_stats_dict["Q_T_summary_stats"]     #don't need the summary stats pd.DataFrame in dictionary
    params["Q_T_stats_dict"] = Q_T_stats_dict

    return params
