#NN control FunctionHeatmaps gives the heatmaps of values of the NN control as a FUNCTION of features
# - it  calculates the NN outputs for various inputs
# - whether these inputs actually appear (in that range) in the training or testing data

from fun_invest_NN_strategy import invest_NN_strategy
from fun_construct_Feature_vector import construct_Feature_vector
import class_Neural_Network
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import sys
import pandas as pd
import datetime
import copy
import torch
from w_constraint_activations import custom_activation

def fun_Heatmap_NN_control_basic_features(params,  #params dictionary with *trained* NN parameters and setup as in main code
                                W_num_pts = 3001, #number of points for wealth grid
                                W_min = 0.0,       #minimum for the wealth grid
                                W_max = 3000.0,     #maximum for the wealth grid
                                save_Figures = False,   #Saves figures in format specified below
                                save_Figures_format = "png",
                                fig_filename_prefix = "zHeatmap_asset_",
                                feature_calc_option = None, # Set calc_option = "matlab" to match matlab code, None to match my notes
                                xticklabels = 12, #e.g. xticklabels=6 means we are displaying only every 6th xaxis label to avoid overlapping
                                yticklabels = 50, #e.g. yticklabels = 500 means we are displaying every 500th label
                                cmap = "Reds",   #e.g. "Reds" or "rainbow" etc colormap for sns.heatmap
                                heatmap_cbar_limits= [0.0, 1.0],  # list in format [vmin, vmax] for heatmap colorbar/scale
                                output_HeatmapData_Excel = False    #If TRUE, output the heatmap grid data to Excel, naming uses fig_filename_prefix
                               ):

    #OBJECTIVE: Plots heatmap of optimal fraction of wealth in EACH ASSET as a function of time-to-go and wealth
    #           i.e. basic features only, NO trading signals




    use_PyTorch = False
    #Check if function heatmap is going to work for this objective
    obj_fun = params["obj_fun"]
    if obj_fun in ["ads_stochastic", "qd_stochastic", "ir_stochastic", "te_stochastic"]:
        raise ValueError("PVS error in function heatmap code: Objective function not coded.")




    basket_asset_names = params["asset_basket"]["basket_timeseries_names"] #Get asset names
    basket_asset_columns = params["asset_basket"]["basket_columns"]    #Get asset column names
    Y_order_train = params["Y_order_train"] #Order in which asset columns was used for training

    #match up the asset names with the order in which they were used for training (should be the same)
    asset_names = []    #asset_names will match up with Y_order_train columns
    for col in Y_order_train:
        idx = basket_asset_columns.index(col)   #look up the index of col in basket_asset_columns
        asset_names.append(basket_asset_names[idx]) #get the asset name corresponding to this index

    # Assign weights matrices and bias vectors in NN_object using given value of NN_theta
    #NN_object = params["NN_object"]
    if len(params["NN_object"]) > 1:
        NN_object = params["NN_object"][1]
        NN_object_w = params["NN_object"][0]
        q_max = 60
        q_min = 35
        use_PyTorch = True # At this point, it is synonymous with 2 NNs trained in PyTorch.
        #@todo: fix this problem above
    else:
        NN_object.theta = params["res_BEST"]["NN_theta"]
        NN_object.unpack_NN_parameters()

    if use_PyTorch:
        asset_names.append("Withdrawals")

    #Set up grids for feature vector evaluation
    W_grid = np.linspace(start=W_min, stop=W_max, num=W_num_pts)       #wealth grid


    n_index_grid = np.arange(start=0, stop = params["N_rb"], step = 1) #rebalancing events index grid
    # n_index_grid.shape = (N_rb, )

    n_index_mesh , W_mesh = np.meshgrid(n_index_grid, W_grid)
    # n_index_mesh.shape = (W_num_pts, N_rb)
    # W_mesh.shape = (W_num_pts, N_rb)

    if output_HeatmapData_Excel is True: #Output wealth and time mesh just once, since they stay the same for all assets

        # TIME mesh output
        time_mesh = n_index_mesh * params["delta_t"] #Used to output data to Matlab
        df_time_mesh = pd.DataFrame(time_mesh)
        df_time_mesh.to_excel(fig_filename_prefix + "_FunctionHeatmap_TIME_mesh.xlsx", index=False, header=False)

        # WEALTH mesh output
        df_W_mesh = pd.DataFrame(W_mesh)
        df_W_mesh.to_excel(fig_filename_prefix + "_FunctionHeatmap_WEALTH_mesh.xlsx", index=False, header=False)


    if use_PyTorch:
        asset_loop_vector = np.arange(0, params["N_a"]+1,1) 
    else:
        asset_loop_vector = np.arange(0, params["N_a"],1)

    for asset_index in asset_loop_vector: # asset index \in {0,...,N_a-1}

        z_NNopt_prop_mesh = np.zeros(W_mesh.shape) #will contain NN-optimal proportion of wealth in asset_index

        for n_index in n_index_grid:    #Loop through n_index
            n = n_index + 1 #rebalancing number since n_index = n - 1

            wealth_n = W_mesh[:, n_index]
            if use_PyTorch:
                wealth_n = torch.as_tensor(wealth_n, device = params["device"])

            # ---------------------------Get standardized feature vector ---------------------------
            phi = construct_Feature_vector(params = params,  # params dictionary as per MAIN code
                                           n = n,  # n is rebalancing event number n = 1,...,N_rb, used to calculate time-to-go
                                           wealth_n = wealth_n,  # Wealth vector W(t_n^+), *after* contribution at t_n
                                           # but *before* rebalancing at time t_n for (t_n, t_n+1)
                                           feature_calc_option = feature_calc_option # Set calc_option = "matlab" to match matlab code
                                           )

            # --------------------------- NNCONTROL  ---------------------------
            # Get proportions to invest in each asset at time t_n^+
            #   a_t_n[j,asset_index] = proportion to invest in asset with index asset_index for j-th entry of vector wealth_n

            #a_t_n_output, _, _ = NN_object.forward_propagation(phi=phi)
            #z_NNopt_prop_mesh[:, n_index] = a_t_n_output[:, asset_index]
            if use_PyTorch is True:
                if asset_index == asset_loop_vector[-1]:
                    a_t_n_output= torch.squeeze(NN_object_w.forward(phi))
                    
                    q_n = custom_activation(a_t_n_output, wealth_n, params)

                    withdrawal_q = (q_n-q_min) / (q_max - q_min)
                    withdrawal_q = torch.nan_to_num(withdrawal_q, nan = 0.0, posinf = 0.0, neginf= 0.0)
                    z_NNopt_prop_mesh[:, n_index] = withdrawal_q.detach().to('cpu').numpy()
                else:
                    a_t_n_output = NN_object.forward(phi)
                    z_NNopt_prop_mesh[:, n_index] = a_t_n_output[:, asset_index].detach().to('cpu').numpy()
            else:
                a_t_n_output, _, _ = NN_object.forward_propagation(phi=phi)
                z_NNopt_prop_mesh[:, n_index] = a_t_n_output[:, asset_index]

        #Sort out time-to-go on x-axis labels
        time_to_go = params["T"] - n_index_grid*params["delta_t"]
        #time_to_go = time_to_go.astype(int)  #Convert to integers
        time_to_go = np.round(time_to_go, decimals=1)
        time_to_go = time_to_go.tolist()

        #Sort out wealth on y-axis labels
        W_grid_list = W_grid.tolist()

        plt.figure()
        sns.set(font_scale=0.85)
        sns.set_style("ticks")
        
        print(W_grid_list[::yticklabels])
        h = sns.heatmap(data =z_NNopt_prop_mesh,
                        yticklabels=W_grid_list[::yticklabels],
                        xticklabels = time_to_go[::xticklabels],#time_to_go,
                        vmin= heatmap_cbar_limits[0], vmax= heatmap_cbar_limits[1],
                        cbar_kws={'label': 'Proportion of wealth'},
                        cmap= cmap, cbar = True)
        h.set_xticks(np.arange(0,len(time_to_go),xticklabels))
        h.set_yticks(np.arange(0, len(W_grid_list), yticklabels))
        h.set_yticklabels(W_grid_list[::yticklabels])
        h.set_xticklabels(time_to_go[::xticklabels])
        h.invert_yaxis()
        plt.title('Optimal control (NN) as proportion of wealth in asset ' + asset_names[asset_index])
        plt.ylabel('Wealth')
        plt.xlabel('Time-to-go (years)')


        # Output heatmap data for THIS asset
        if output_HeatmapData_Excel is True:
            df_z_NNopt_prop_mesh = pd.DataFrame(z_NNopt_prop_mesh)
            Excel_filename = fig_filename_prefix + "_FunctionHeatmap_Asset_" + str(asset_index) \
                             + "_" + asset_names[asset_index] + ".xlsx"
            df_z_NNopt_prop_mesh.to_excel(Excel_filename, index=False, header=False)


        #Save figure
        if save_Figures:

            timestamp = datetime.datetime.now().strftime('%Y-%m-%d_%H_%M')

            fig_filename = fig_filename_prefix +  "timestamp_" + timestamp + "_Heatmap_asset_" \
                           +  str(asset_index) + "_" + asset_names[asset_index] \
                           + "." + save_Figures_format
            plt.savefig(fig_filename, format = save_Figures_format, bbox_inches = "tight")


    #plt.show()
    plt.close(fig="all")


def fun_Heatmap_NN_control_histpath_TradSig(params,  #params dictionary with *trained* NN parameters and setup as in main code
                                            W_num_pts = 3001,  #number of points for wealth grid
                                            W_min = 0.0,  #minimum for the wealth grid
                                            W_max = 3000.0,  #maximum for the wealth grid
                                            save_Figures = False,  #Saves figures in format specified below
                                            save_Figures_format = "png",
                                            fig_filename_prefix = "",
                                            feature_calc_option = None,  # Set calc_option = "matlab" to match matlab code, None to match my notes
                                            yyyymm_path_start = None,  #yyyymm START of HISTORICAL PATH for trading signals
                                            # yyyymm END is *calculated* based on params info
                                            xticklabels = 12,  #e.g. xticklabels=6 means we are displaying only every 6th xaxis label to avoid overlapping
                                            yticklabels=50, # e.g. yticklabels = 500 means we are displaying every 500th label
                                            cmap="Reds",  # e.g. "Reds" or "rainbow" etc colormap for sns.heatmap
                                            heatmap_cbar_limits= [0.0, 1.0]  # list in format [vmin, vmax] for heatmap colorbar/scale
                                            ):

    #OBJECTIVE: Plots heatmap of optimal fraction of wealth in EACH ASSET as a function of time-to-go and wealth
    #           for a given trade signal historical path starting at yyyymm_path_start



    # Create local copy of params
    params_temp = copy.deepcopy(params)


    #Check if function heatmap is going to work for this objective
    obj_fun = params_temp["obj_fun"]
    if obj_fun in ["ads_stochastic", "qd_stochastic", "ir_stochastic", "te_stochastic"]:
        raise ValueError("PVS error in function heatmap code: Objective function not coded.")



    T = params_temp["T"]
    delta_t = params_temp["delta_t"]  # rebalancing time interval


    #-----------------------------------------------------------------------------------------
    #Get historical path of trade signals:

    # Get market data = "bootstrap_source_data" in order to get a path of trading signals
    # Note that the actual bootstrapped data for training + testing would use a SUBSET of the dates here
    # - so we can use true out-of-sample market data points if we wanted to
    df_source_data = params_temp["bootstrap_source_data"]
    df_source_data_yyyymm_start = np.min(df_source_data.index)  # beginning of source data
    df_source_data_yyyymm_end = np.max(df_source_data.index)  # end of source data

    # get interval between df_source_data points
    source_data_delta_t = params_temp["asset_basket_data_settings"]["data_read_delta_t"]


    # Number of consecutive observations from df_source_data we need to construct a path
    nobs = int(T / source_data_delta_t)

    if yyyymm_path_start == None:
        raise ValueError("PVS error: need yyyymm_path_start for function 'fun_Heatmap_NN_control_histpath_TradSig'.")

    idx_start = df_source_data.index.get_loc(yyyymm_path_start)
    idx_end = idx_start + nobs  #Note: idx_end will be EXCLUDED below


    try:
        print("Start month for historical path: " + str(df_source_data.iloc[idx_start].name))
        print("End month for historical path: " + str(df_source_data.iloc[idx_end-1].name))
    except:
        raise ValueError("PVS error: End date out of bounds. Select earlier yyyymm_path_start.")


    # Select subset of source data with the historical path of interest
    df_paths = df_source_data.iloc[idx_start:idx_end].copy()    #idx_end will be EXCLUDED

    # Get row indices of at intervals required for portfolio rebalancing events
    row_indices = np.arange(0, nobs, delta_t / source_data_delta_t)
    row_indices.astype(int)

    # Get those entries corresponding to rebalancing events
    df_paths_rebal_events = df_paths.iloc[row_indices].copy()

    # Extract trading signals: Note that it will ALREADY be in the RIGHT ORDER as used for training
    df_TradSig_rebal_events = df_paths_rebal_events[params_temp["TradSig_order_train"]].copy()
    n_tradsig = len(params_temp["TradSig_order_train"])   #number of trade signals

    # WRITE OVER THE EXISTING "TradSig" data entries in params_temp
    del params_temp["TradSig"]

    #   params_temp["TradSig"][j, n, i] = Point-in time observation for trade signal i, along sample path j, at rebalancing time t_n;
    #                               can only rely on time series observations <= t_n

    # REPLACE params_temp["TradSig"]
    params_temp["TradSig"] = np.zeros([W_num_pts, params_temp["N_rb"], n_tradsig])

    for i in np.arange(0,n_tradsig,1):
        #Each sample path will be the SAME value of the historical trade signal
        # - this will ensure feature vector code for heatmap will work as below
        params_temp["TradSig"][:,:,i] = df_TradSig_rebal_events[params_temp["TradSig_order_train"][i]].copy()

    #Get trade signal names
    basket_tradsig_names = params_temp["trading_signal_basket"]["basket_timeseries_names"] #Get asset names
    basket_tradsig_columns = params_temp["TradSig_order_train"]    #Get asset column names

    tradsig_names = []    #tradsig_names will give the trade signal names
    for col in df_TradSig_rebal_events.columns:
        idx = basket_tradsig_columns.index(col)   #look up the index of col in basket_asset_columns
        tradsig_names.append(basket_tradsig_names[idx]) #get the asset name corresponding to this index

    #-------------------------------------------------------------------------------------
    #Plot historical path of trade signals all on one graph
    fig_title = "Trading signals: historical values observed at each rebalancing event \n" +    \
                str(min(df_TradSig_rebal_events.index)) + " to " + str(max(df_TradSig_rebal_events.index))
    def x_axis_tttgo(n_index): #converts date index to time-to-go
        ttgo = params_temp["T"] - n_index*params_temp["delta_t"]
        return ttgo

    x_axis_dates = pd.to_datetime(df_TradSig_rebal_events.index, format='%Y%m', errors='coerce')
    x_axis_timetogo = [int(x_axis_tttgo(n)) for n in np.arange(0,len(x_axis_dates) , 1)]


    #Split Sharpe and non-Sharpe ratios
    indices_NOTsharpe = [i for i in np.arange(0,n_tradsig,1) if "Sharpe" not in basket_tradsig_columns[i]]
    indices_sharpe = [i for i in np.arange(0, n_tradsig, 1) if "Sharpe" in basket_tradsig_columns[i]]

    fig, ax = plt.subplots(constrained_layout = True)
    ax2 = ax.twinx()  # instantiate a second axes that shares the same x-axis

    for i in indices_NOTsharpe: #Plot NON-Sharpe ratios on LHS vertical axis
        ax.plot(x_axis_timetogo, df_TradSig_rebal_events[basket_tradsig_columns[i]], label = tradsig_names[i])

    for i in indices_sharpe: #Plot Sharpe ratios on RHS vertical axis
        ax2.plot(x_axis_timetogo, df_TradSig_rebal_events[basket_tradsig_columns[i]], "r--", label=tradsig_names[i])



    #ax.axhline(y = 0.0, color = "k", linestyle = '--', linewidth= 0.5)
    ax.set_title(fig_title, fontsize=11)
    #ax.legend(loc = "lower left")
    ax.set_xlim(max(x_axis_timetogo), 0)    #ensures axis is in right direction
    ax.set_xticks(x_axis_timetogo, minor=True)
    ax.set_ylabel("Other trading signals", fontsize=10)
    ax.set_xlabel("Time-to-go", fontsize=10)
    ax.grid(b = True, which="both", axis="x" )

    #ax2.legend(loc="upper right")
    ax2.set_ylabel('Rolling Sharpe ratio')  # we already handled the x-label with ax1

    # Save figure
    if save_Figures:

        timestamp = datetime.datetime.now().strftime('%Y-%m-%d_%H_%M')
        fig_filename = fig_filename_prefix + "timestamp_" + timestamp + "_Heatmap_HistoricalPaths_trading_signals" \
                       + "_yyyymm_path_start_" + str(yyyymm_path_start) \
                       + "." + save_Figures_format

        plt.savefig(fig_filename, format=save_Figures_format)

    #-----------------------------------------------------------------------------------------
    #ASSET INFO:

    basket_asset_names = params_temp["asset_basket"]["basket_timeseries_names"] #Get asset names
    basket_asset_columns = params_temp["asset_basket"]["basket_columns"]    #Get asset column names
    Y_order_train = params_temp["Y_order_train"] #Order in which asset columns was used for training

    #match up the asset names with the order in which they were used for training (should be the same)
    asset_names = []    #asset_names will match up with Y_order_train columns
    for col in Y_order_train:
        idx = basket_asset_columns.index(col)   #look up the index of col in basket_asset_columns
        asset_names.append(basket_asset_names[idx]) #get the asset name corresponding to this index


    #-----------------------------------------------------------------------------------------
    #HEATMAP construction:
    # Assign weights matrices and bias vectors in NN_object using given value of NN_theta
    NN_object = params_temp["NN_object"]
    NN_object.theta = params_temp["res_BEST"]["NN_theta"]
    NN_object.unpack_NN_parameters()

    #Set up grids for feature vector evaluation
    W_grid = np.linspace(start=W_min, stop=W_max, num=W_num_pts)       #wealth grid

    n_index_grid = np.arange(start=0, stop = params_temp["N_rb"], step = 1) #rebalancing events index grid
    # n_index_grid.shape = (N_rb, )

    n_index_mesh , W_mesh = np.meshgrid(n_index_grid, W_grid)
    # n_index_mesh.shape = (W_num_pts, N_rb)
    # W_mesh.shape = (W_num_pts, N_rb)

    for asset_index in np.arange(0, params_temp["N_a"],1): # asset index \in {0,...,N_a-1}

        z_NNopt_prop_mesh = np.zeros(W_mesh.shape) #will contain NN-optimal proportion of wealth in asset_index

        for n_index in n_index_grid:    #Loop through n_index
            n = n_index + 1 #rebalancing number since n_index = n - 1

            # ---------------------------Get standardized feature vector ---------------------------
            phi = construct_Feature_vector(params = params_temp,  # params dictionary with replaced trade signal data
                                           n = n,  # n is rebalancing event number n = 1,...,N_rb, used to calculate time-to-go
                                           wealth_n = W_mesh[:, n_index],  # Wealth vector W(t_n^+), *after* contribution at t_n
                                           # but *before* rebalancing at time t_n for (t_n, t_n+1)
                                           feature_calc_option = feature_calc_option # Set calc_option = "matlab" to match matlab code
                                           )

            # --------------------------- NNCONTROL  ---------------------------
            # Get proportions to invest in each asset at time t_n^+
            #   a_t_n[j,asset_index] = proportion to invest in asset with index asset_index for j-th entry of vector wealth_n

            a_t_n_output, _, _ = NN_object.forward_propagation(phi=phi)

            z_NNopt_prop_mesh[:, n_index] = a_t_n_output[:, asset_index]


        #Sort out time-to-go on x-axis labels
        time_to_go = params["T"] - n_index_grid*params["delta_t"]
        #time_to_go = time_to_go.astype(int)  #Convert to integers
        time_to_go = np.round(time_to_go, decimals=1)
        time_to_go = time_to_go.tolist()

        #Sort out wealth on y-axis labels
        W_grid_list = W_grid.tolist()

        plt.figure()
        sns.set(font_scale=0.85)
        sns.set_style("ticks")

        h = sns.heatmap(data =z_NNopt_prop_mesh,
                        yticklabels=W_grid_list[::yticklabels],
                        xticklabels = time_to_go[::xticklabels],
                        vmin= heatmap_cbar_limits[0], vmax= heatmap_cbar_limits[1],
                        cbar_kws={'label': 'Proportion of wealth'},
                        cmap= cmap, cbar = True)
        h.set_xticks(np.arange(0,len(time_to_go),xticklabels))
        h.set_yticks(np.arange(0, len(W_grid_list), yticklabels))
        h.invert_yaxis()
        plt.title('Optimal control (NN) as proportion of wealth in asset ' + asset_names[asset_index])
        plt.ylabel('Wealth')
        plt.xlabel('Time-to-go (years)')


        #Save figure
        if save_Figures:

            timestamp = datetime.datetime.now().strftime('%Y-%m-%d_%H_%M')

            fig_filename = fig_filename_prefix + "timestamp_" + timestamp + "_Heatmap_asset_" \
                           +  str(asset_index) + "_" + asset_names[asset_index] \
                           + "_yyyymm_path_start_" + str(yyyymm_path_start) \
                           + "." + save_Figures_format
            plt.savefig(fig_filename, format = save_Figures_format, bbox_inches = "tight")


    #plt.show()
    plt.close(fig="all")