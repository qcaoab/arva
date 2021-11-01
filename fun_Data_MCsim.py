#OBJECTIVE: Creates training/testing datasets for NN based on parameteric model
# TRAINING data: appends the following to params
#   params["Y_train"][j, n, i] = Return, along sample path j, over time period (t_n, t_n+1), for asset i
#       -- IMPORTANT: params["Y_train"][j, n, i] entries are basically (1 + return), so it is ready for multiplication with start value
#   params["Y_order_train"][i] = column name of asset i used for identification

#Similar for TESTING data, with suffix "_test"


import math
import pandas as pd
import numpy as np
import os
from fun_Data_MCsim_model_params import get_model_params
import datetime #used to timestamp.csv files if needed



def func_SimLogJumpMult(NrSims, dt, model_params):
    # Function simulates the log jump multipliers for NrSims simulation paths

    #INPUTS:
    #   model_params = functions_Params.params (Object)
    #   dt = timestep length used in simulation


    # Simulate total NUMBER of jumps in S over interval of length dt for each simulated path
    #   Each entry gives number of jumps for that path in [0,dt]
    pi_dt = np.random.poisson(lam = model_params["lambda_jump"] * dt, size=NrSims)

    # Initialize sum Log jump multipliers
    log_jump_mult_sum = np.zeros(NrSims)

    for i in np.arange(1, NrSims+1,1): #loop over pi_dt

        log_jump_mult = 0   #Initialize in case no jumps (would be the case for GBM)

        #Jump arrival times does NOT matter: Only dealing with ONE sim time interval at a time

        if  "merton" in model_params["model_ID"]:
            # if merton model:
            #     params_dict["m_tilde"] = m_tilde  # mean of log jump mult dist
            #     params_dict["gamma_tilde"] = gamma_tilde  # stdev of log jump mult dist

            log_jump_mult = np.random.normal(loc = model_params["m_tilde"], \
                                             scale = model_params["gamma_tilde"], \
                                             size = pi_dt[i-1])

        if "kou" in model_params["model_ID"]:
            # if kou model:
            #     params_dict["nu"] = nu  # prob of UP jump
            #     params_dict["zeta1"] = zeta1  # expo param of UP jump
            #     params_dict["zeta2"] = zeta2  # expo param of DOWN jump

            # Generate random nrs in [0,1], one for each jump
                        #If random nr <= kou_nu, classify it as UP jump
            jump_up_or_down = (np.random.uniform(size=pi_dt[i-1]) <= model_params["nu"])

            # Generate  uniform RVs used to get exponential RVs
            temp_U =  np.random.uniform(size = pi_dt[i-1])

            #Proceed for the moment as if ALL jumps are both up and down
            temp_UPjumps = -np.log(temp_U) / model_params["zeta1"]
            temp_DOWNjumps = -np.log(temp_U) / model_params["zeta2"]

            #Get jumps right by multiplying elementwise with indicators
            log_jump_mult = np.multiply(jump_up_or_down, temp_UPjumps ) \
                            - np.multiply(1 - jump_up_or_down,temp_DOWNjumps )
                            # The MINUS is important!


        #Sum all of the log jump multipliers for this sim, in this time interval
        log_jump_mult_sum[i-1] = sum(log_jump_mult)


    return log_jump_mult_sum


def MCsim (MCsim_info, params):
    #IMPORTANT! model_IDs in MCsim_info must be in same order as params["asset_basket"]["basket_columns"]


    #INPUTS:
    # MCsim_model_info = dictionary with info such as model_ID_XX for each asset and corr_matrix
    # model_IDs:
    #   MCsim_info["model_ID_XX"] should correspond to model for params["asset_basket"]["basket_columns"][XX]
    # params = params dictionary as setup in main code with all the info


    #---------------------------------------------------------------------------------
    #General setup

    N_d = MCsim_info["N_d"]     #Nr of simulation paths
    purpose = MCsim_info["purpose"]    #"train" or "test"


    #If output_csv == TRUE, will output .csv files with data
    output_csv = params["output_csv_data_training_testing"]  #output .csv files with the bootstrapped data


    #Get main structural parameters
    T = params["T"]  # Investment time horizon, in years
    N_rb = params["N_rb"]  # Nr of equally-spaced rebalancing events in [0,T]
    delta_t = params["delta_t"]  # Rebalancing time interval, should be some integer multiple of data_delta_t
    N_a = params["N_a"] #Number of assets


    #---------------------------------------------------------------------------------
    #Get model params
    real_or_nominal = params["real_or_nominal"]  # "real" or "nominal"

    model_params = {}
    sim_drift_logS = {} #drift of log S used in the simulation
    sim_sig  = {}

    for asset_index in np.arange(0,N_a,1):
        model_ID = MCsim_info["model_ID_" + str(asset_index)]
        model_params[asset_index] = get_model_params(model_ID= model_ID,
                                                     real_or_nominal = real_or_nominal)

        sim_drift_logS[asset_index] = model_params[asset_index]["mu"] \
                                      - (model_params[asset_index]["lambda_jump"] * model_params[asset_index]["kappa"]) \
                                      - 0.5 * (model_params[asset_index]["sig"] ** 2)
        sim_sig[asset_index] = model_params[asset_index]["sig"]


    corr_matrix =  MCsim_info["corr_matrix"]
    L_cholesky = np.linalg.cholesky(corr_matrix)    #lower-triangular Cholesky factor of corr_matrix

    # --------------------------------------------------------------------------------------
    # PATH SIMULATION - returns will be backed out from paths


    # Sim_paths initialize (init val doesn't matter - will just use to get *returns*)
    #   Sim_paths,shape = (N_d,N_a)
    #   Sim_paths[:,asset_index] = all the simulated values for asset_index at that time instant
    Sim_paths = np.ones([N_d,N_a])  * 100.

    # Initialize ASSET RETURNS output data:
    #   Y.shape = (N_d, N_rb, N_a)
    Y =  np.zeros([N_d, N_rb, N_a])
    Y_order = params["asset_basket"]["basket_columns"]   #to keep record of order of columns


    #Loop over rebalancing events
    for n_index in np.arange(0,N_rb,1):

        # From previous timestep
        Sim_paths_prev = Sim_paths.copy()

        # Simulate UNCORRELATED standard normal RVs for diffusion part
        Z = np.random.normal(loc=0.0, scale=1.0, size=(N_d,N_a))

        # CORRELATED normal RVs for diffusion part
        X = np.matmul(Z,np.transpose(L_cholesky))

        #Check correlations if needed
        #test_corr = np.corrcoef(np.transpose(X))    #Transpose so each *row* is variable

        #Do each asset separately
        for asset_index in np.arange(0,N_a,1):

            # Simulate nr of jumps and LOG jump multipliers and add together
            if model_params[asset_index]["lambda_jump"] <= 1e-9:    #Take care of cases where we model risk-free asset
                log_jump_mult_sum = np.zeros(N_d)
            else:
                log_jump_mult_sum = func_SimLogJumpMult(NrSims=N_d,
                                                    dt=delta_t,
                                                    model_params=model_params[asset_index])

            # Path simulation
            multiplier =  np.exp(sim_drift_logS[asset_index] * delta_t \
                                           + sim_sig[asset_index] * math.sqrt(delta_t) * X[:, asset_index] \
                                           + log_jump_mult_sum
                                 )

            Sim_paths[:, asset_index] = np.multiply(Sim_paths_prev[:,asset_index],multiplier)


            # Calculate (1+ return) for output
            Y[:,n_index,asset_index] = np.divide(Sim_paths[:, asset_index], Sim_paths_prev[:, asset_index])

        # ----------------
        #Update user on MC sim progress
        print( str( (n_index+1)/N_rb * 100) + "% of MC simulations done.")


    #---------------------------------------------------------------------------------
    #Write out to .csv files if required

    if output_csv:

        # Get datetime stamp
        date_time = datetime.datetime.now()
        date_time_str = date_time.strftime("%m-%d-%Y__%H_%M_%S")

        # Write out the bootstrap settings:
        temp = pd.DataFrame(MCsim_info.values(), index=MCsim_info.keys())

        temp.to_csv("z_MCsim_info_ " + date_time_str + ".csv", header=False, index=True)

        for asset_index in np.arange(0,N_a,1):

            # first convert to dataframe
            temp = pd.DataFrame(Y[:,:,asset_index])

            # Construct filename using column name

            filename = "z_MCsim_" + Y_order[asset_index]

            temp.to_csv(filename + date_time_str + ".csv", header=False, index=False)

            # Each outputted .csv file same format as Y[:,:,asset_index]

    # ---------------------------------------------------------------------------------
    #Set final output (no trading signals here)

    params["Y_"+ purpose] = Y.copy()
    params["Y_order_" + purpose] = Y_order


    return params





