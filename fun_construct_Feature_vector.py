import numpy as np
import torch

    
def construct_Feature_vector(params,                        # params dictionary as per MAIN code
                             n,                             # n is rebalancing event number n = 1,...,N_rb,
                                                            #   used to calculate time-to-go
                             wealth_n,                      #  Wealth vector W(t_n^+), usually of length params["N_d"]
                                                            #       *after* contribution at t_n but
                                                            #       *before* rebalancing at time t_n for (t_n, t_n+1)
                             feature_calc_option = None     # Set calc_option = "matlab" to match matlab code
                             ):

    #OUTPUTS: phi = standardized feature vector
    # - phi.shape(len(wealth_n), N_phi)
    #       in usual applications, len(wealth_n) = params["N_d"], but could be different for e.g. heatmaps of optimal control


    #INDEX of rebalancing event in the data
    n_index = n - 1  # index of rebalancing event in the data


    # Define local copies for ease of reference
    N_rb = params["N_rb"]  # Nr of equally-spaced rebalancing events in [0,T]
    N_a = params["N_a"]  # Nr of assets = nr of output nodes
    N_phi = params["N_phi"]  # Nr of features, i.e. the number of input nodes
    N_phi_standard = params["N_phi_standard"]   #Nr of *standard* features, EXCLUDING trading signals
    #N_d = params["N_d"]  # Nr of training data return sample paths
    T = params["T"]  # Terminal time
    delta_t = params["delta_t"]  # time interval between rebalancing events
    pytorch = params["NN_training_options"]["pytorch"]


    # --------------------------- CONSTRUCT FEATURE VECTOR and standardize ---------------------------

    if N_phi >= 2: #At a minimum, we have time-to-go and wealth
        # Time to go:

        t_n = n_index * delta_t  # time of rebalancing event
        time_to_go = (T - t_n) * np.ones(len(wealth_n))
        time_to_go_std = time_to_go / T
        
        if pytorch:
            time_to_go_std = torch.tensor(time_to_go_std, device = params["device"])

        if feature_calc_option == "matlab":
            # Time to go as defined in Matlab (used here if we want to get exactly the same results)
            t_n_plus_1 = n * delta_t
            time_to_go = (T - t_n_plus_1) * np.ones(len(wealth_n))
            time_to_go_std = time_to_go / T



        # Wealth:

        # Get benchmark values for wealth standardization
        # - These values will always be TRAINING dataset values
        # - They are effectively just values used for the standardization of NN features
        # - (This also aligns to Matlab code)

        if "benchmark_W_mean_train" not in params.keys():
            raise ValueError("PVS error in 'construct_Feature_vector' for feature vector standardization: "
                             "params['benchmark_W_mean_train'] missing. ")
        elif "benchmark_W_std_train" not in params.keys():
            raise ValueError("PVS error in 'construct_Feature_vector' for feature vector standardization: "
                             "params['benchmark_W_std_train'] missing. ")

        else:  # Values are all there
            benchmark_W_mean_train = params["benchmark_W_mean_train"][:, n_index]
            benchmark_W_std_train = params["benchmark_W_std_train"][:, n_index]

        if benchmark_W_std_train == 0:  # Correct division by zero (for example no variance at time zero)
            benchmark_W_std_train = 1.0

        if pytorch:
            benchmark_W_mean_train = torch.tensor(benchmark_W_mean_train, device = params["device"])
            benchmark_W_std_train = torch.tensor(benchmark_W_std_train, device=params["device"])
            wealth_n = torch.tensor(wealth_n, device=params["device"])
            
        wealth_std= (wealth_n - benchmark_W_mean_train) / benchmark_W_std_train

       

        #Add BENCHMARK wealth in the case of obj_fun in ["ads_stochastic", "qd_stochastic", "ir_stochastic", "te_stochastic"]
        if params["obj_fun"] in ["ads_stochastic", "qd_stochastic", "ir_stochastic", "te_stochastic"]:

            if "benchmark_W_paths" not in params.keys():
                #This would be paths on whatever dataset we are using; training OR testing!
                # Actual wealth paths for this dataset (benchmark strategy)
                raise ValueError("PVS error in 'construct_Feature_vector': missing benchmark_W_paths")

            else: #If benchmark values are there
                #   Stochastic benchmark relies on wealth for this dataset, training or testing
                #   But same STANDARDIZATION is used as for TRAINING
                benchmark_W_mean_train = params["benchmark_W_mean_train"][:, n_index]
                benchmark_W_std_train = params["benchmark_W_std_train"][:, n_index]

                if benchmark_W_std_train == 0:  # Correct division by zero (for example no variance at time zero)
                    benchmark_W_std_train = 1.0

                benchmark_W_n = params["benchmark_W_paths"][:, n_index]

                benchmark_wealth_std = (benchmark_W_n - benchmark_W_mean_train) / benchmark_W_std_train


    # --------------------------- CONSTRUCT FEATURE VECTOR for OUTPUT---------------------------
    # Construct feature vector at time t_n+ for NN

    
    #First 2 features always "wealth" and "time-to-go"
    if pytorch:
        phi = torch.zeros([len(wealth_n), N_phi], device=params["device"])  # Initialize to get shape right
    else:
        phi = np.zeros([len(wealth_n), N_phi])  # Initialize to get shape right
        
    #Order of first 2 matches the matlab order (my original notes has the first 2 reversed)
    phi[:, 0] = time_to_go_std
    phi[:, 1] = wealth_std


    #Add benchmark
    if params["obj_fun"] in ["ads_stochastic", "qd_stochastic", "ir_stochastic", "te_stochastic"]:
        phi[:, 2] = benchmark_wealth_std


    #Add trading signals
    if params["use_trading_signals_TrueFalse"] == True:
        #   params["TradSig"][j, n, i] = Point-in time observation for trade signal i, along sample path j, at rebalancing time t_n;
        #                               can only rely on time series observations <= t_n

        for i in np.arange(N_phi_standard, N_phi, 1):
            trad_sig_index = i - N_phi_standard
            trad_sig_mean = params["TradSig_MEAN_train"][trad_sig_index]
            trad_sig_stdev = params["TradSig_STDEV_train"][trad_sig_index]


            #Always use TRAINING data mean/stdev for standardization
            # params["TradSig_MEAN_train"] and params["TradSig_STDEV_train"] will be
            # in the right order. This is calculated in fun_Data_bootstrap
            phi[:, i] =  (params["TradSig"][:, n_index, trad_sig_index] - trad_sig_mean)/trad_sig_stdev
            # Note params["TradSig"] is either for the training or testing data, set in fun_Data.set_training_testing_data


    return phi




