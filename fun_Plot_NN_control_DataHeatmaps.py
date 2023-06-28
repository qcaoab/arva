

#NN control DataHeatmaps gives the heatmaps of values of the NN control seen on the given data
# - it does NOT calculate the "theoretical" NN outputs for various inputs
# - but instead bins the actual NN output values as seen on the actual training/testing dataset
#   according to the wealth values along each path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import datetime
import pandas as pd
from constraint_activations import w_custom_activation


def plot_DataHeatmaps(
        params, #params dictionary with *trained* NN results and setup as in main code
        y_bin_min=-200.,  # left endpoint of first y-axis bin (wealth or wealth difference)
        y_bin_max=3000.,  # right endpoint of last y-axis bin (wealth or wealth difference)
        delta_y_bin =200.0,  # bin width for y-axis bins
        save_Figures = False,   #Saves figures in format specified below
        save_Figures_format="png",
        fig_filename_prefix="z_DataHeatmap_asset_",
        xticklabels=12,  # e.g. xticklabels=6 means we are displaying only every 6th xaxis label to avoid overlapping
        yticklabels=1,  # e.g. yticklabels = 500 means we are displaying every 500th label
        cmap="Reds",  # e.g. "Reds" or "rainbow" etc colormap for sns.heatmap
        heatmap_cbar_limits= [0.0, 1.0],  # list in format [vmin, vmax] for heatmap colorbar/scale
        output_HeatmapData_Excel=False # If TRUE, output the heatmap grid data to Excel, naming uses fig_filename_prefix
):


    obj_fun = params["obj_fun"]
    basket_asset_names = params["asset_basket"]["basket_timeseries_names"] #Get asset names
    basket_asset_columns = params["asset_basket"]["basket_columns"]    #Get asset column names
    Y_order_train = params["Y_order_train"] #Order in which asset columns was used for training

    #match up the asset names with the order in which they were used for training (should be the same)
    asset_names = []    #asset_names will match up with Y_order_train columns
    for col in Y_order_train:
        idx = basket_asset_columns.index(col)   #look up the index of col in basket_asset_columns
        asset_names.append(basket_asset_names[idx]) #get the asset name corresponding to this index

    asset_names.append("Total Proportion")


    #Construct data for DataHeatmaps
    if obj_fun in ["ads_stochastic", "qd_stochastic", "ir_stochastic", "te_stochastic"]:
        params = construct_data_for_DataHeatmaps__ads_stochastic(
                    params=params,  # params dictionary with *trained* NN results and setup as in main code
                    W_diff_min= y_bin_min,
                    W_diff_max= y_bin_max,
                    delta_W_diff= delta_y_bin
                    )

        DataHeatmaps_bin_left_edges = params["DataHeatmaps_W_diff_bin_left_edges"]
        ylabel_text = "W(t) minus benchmark W(t)"

    else: #For other objectives
        params = construct_data_for_DataHeatmaps(
                                    params = params,  #params dictionary with *trained* NN results and setup as in main code
                                    W_min = y_bin_min,   #left endpoint of first W bin
                                    W_max = y_bin_max,   #right endpoint of last W bin
                                    delta_W = delta_y_bin   #bin width for W bins
                                )

        DataHeatmaps_bin_left_edges = params["DataHeatmaps_W_bin_left_edges"]
        ylabel_text = "Wealth"

    #Get asset proportions
    DataHeatmaps_Asset_props = params["DataHeatmaps_Asset_props"]


    #Set up meshgrids for plot

    n_index_grid = np.arange(start=0, stop = params["N_rb"], step = 1) #rebalancing events index grid
    # n_index_grid.shape = (N_rb, )

    n_index_mesh , y_mesh = np.meshgrid(n_index_grid, DataHeatmaps_bin_left_edges)
    # n_index_mesh.shape = (W_num_pts, N_rb)
    # y_mesh.shape = (W_num_pts, N_rb)

    if output_HeatmapData_Excel is True: #Output wealth and time mesh just once, since they stay the same for all assets

        # TIME mesh output
        time_mesh = n_index_mesh * params["delta_t"] #Used to output data to Matlab
        df_time_mesh = pd.DataFrame(time_mesh)
        df_time_mesh.to_excel(fig_filename_prefix + "_DataHeatmap_TIME_mesh.xlsx", index=False, header=False)

        # WEALTH mesh output
        df_y_mesh = pd.DataFrame(y_mesh)
        df_y_mesh.to_excel(fig_filename_prefix + "_DataHeatmap_Yaxis_mesh.xlsx", index=False, header=False)



            #Sort out time-to-go on x-axis labels

    # n_index_grid.shape = (N_rb, )
    time_to_go = params["T"] - n_index_grid*params["delta_t"]
    #time_to_go = time_to_go.astype(int)  #Convert to integers
    time_to_go = np.round(time_to_go, decimals=1)
    time_to_go = time_to_go.tolist()

    for asset_index in np.arange(0, params["N_a"]+1,1): # asset index \in {0,...,N_a-1} and withdrawal strategy N_a

        # --------------------------- PROPORTIONS IN EACH ASSET  ---------------------------
        if asset_index == params["N_a"]:
            z_Data = params["DataHeatmaps_exploration_props"]
        else:
            z_Data = DataHeatmaps_Asset_props[:,:,asset_index]
        # DataHeatmaps_Asset_props[j,n,i] =  (for wealth in bin j) *average* proportion invested in asset i at rebal time t_n


        # Output heatmap data for THIS asset
        if output_HeatmapData_Excel is True:
            df_z_Data = pd.DataFrame(z_Data)
            Excel_filename = fig_filename_prefix + "_DataHeatmap_Asset_" + str(asset_index) \
                             + "_" + asset_names[asset_index] + ".xlsx"
            df_z_Data.to_excel(Excel_filename, index=False, header=False)


        # --------------------------- Construct heatmap  figure---------------------------
        #Avoids cutting off the xlabel and sides of box
        from matplotlib import rcParams
        rcParams.update({'figure.autolayout': True})

        plt.figure()
        sns.set(font_scale=0.85)
        sns.set_style("ticks")

        #Set title and colorbar label
        plt.title('NN training dataset: Proportion of wealth in asset ' + asset_names[asset_index])
        
        cbar_kws_label = "Proportion of wealth"

        h = sns.heatmap(data =z_Data,
                        yticklabels= DataHeatmaps_bin_left_edges[::yticklabels],
                        xticklabels = time_to_go[::xticklabels],#time_to_go,
                        vmin= heatmap_cbar_limits[0], vmax= heatmap_cbar_limits[1],
                        cbar_kws={'label': cbar_kws_label},
                        cmap=cmap, cbar = True)
        h.set_xticks(np.arange(0,len(time_to_go),xticklabels))
        h.set_yticks(np.arange(0, len(DataHeatmaps_bin_left_edges), yticklabels))
        h.invert_yaxis()

        
        plt.ylabel(ylabel_text)
        plt.xlabel('Time-to-go (years)')

        # Save figure
        if save_Figures:

            timestamp = datetime.datetime.now().strftime('%Y-%m-%d_%H_%M')

            fig_filename = fig_filename_prefix + "timestamp_" + timestamp + "_DataHeatmap_asset_" \
                           + str(asset_index) + "_" + asset_names[asset_index] \
                           + "_[" + str(params["obj_fun_rho"]) + "]"\
                           + "." + save_Figures_format

            plt.savefig(fig_filename, format=save_Figures_format, bbox_inches="tight")

    # plt.show()
    plt.close(fig="all")



def construct_data_for_DataHeatmaps(
        params,  #params dictionary with *trained* NN results and setup as in main code
        W_min = -200.,   #left endpoint of first W bin
        W_max = 3000.,   #right endpoint of last W bin
        delta_W = 200.0   #bin width for W bins
    ):

    #Objective: Construct data used for the DataHeatmap plots

    #OUTPUTS:
    # Appends fields to the params dictionary:
    # params["DataHeatmaps_W_bin_left_edges"] = left edges of bins for wealth
    # params["DataHeatmaps_Asset_props"]:
    #       #DataHeatmaps_Asset_props[j,n,i] =  (for wealth in bin j) *average* proportion invested in asset i at rebal time t_n

    #-----------------------------------------------
    # Get basic info
    N_d = params["N_d"]
    N_rb = params["N_rb"]
    N_a = params["N_a"]


    #Construct bins for W
    W_bin_left_edges = np.arange(W_min, W_max, delta_W)
    count_W_bins = W_bin_left_edges.shape[0]


    #Initialize outputs
    Asset_Heatmaps = np.empty([count_W_bins,  N_rb, N_a])
    Asset_Heatmaps[:] = np.NaN
    #Asset_Heatmaps[j,n,i] =  (for wealth in bin j) *average* proportion invested in asset i at rebal time t_n

    Data_exploration_Heatmaps = np.empty([count_W_bins,  N_rb])
    Data_exploration_Heatmaps[:] = np.NaN
    #Data_exploration_Heatmaps[j,n] =  (for wealth in bin j) *average* proportion of times we visit this point in the grid


    for n_index in np.arange(0, N_rb, 1): #Loop over rebalancing times

        #Get wealth vector at time (t_n^+)
        
        #changed to w_allocation to ensure this is wealth after withdrawals
        W_t_n = params["W_allocation"][:,n_index] #W to contain the wealth *after* contribution at t_n

        bin_count = 0
        #Loop through W bins
        for bin_index in np.arange(0,count_W_bins, 1):

            #Get bin left edge
            bin_left_edge = W_bin_left_edges[bin_index]

            #Get bin right edge
            if bin_index < count_W_bins-1:
                bin_right_edge = W_bin_left_edges[bin_index + 1]
            else:
                bin_right_edge = W_max

            # Get *indices* of wealth values in W_t_n that would fall in each bin
            indices_bin = np.where((W_t_n>= bin_left_edge) & (W_t_n < bin_right_edge) )

            if len(indices_bin[0])>0:   #if there are actually wealth values in this bin
                rows_bin = indices_bin[0]   #get the row numbers (i.e. training data paths) where wealth values are in each bin
                bin_count += len(rows_bin)

                Data_exploration_Heatmaps[bin_index, n_index] = len(rows_bin) / N_d
                #----------------------------------------
                #Average out the proportions invested in each asset for wealth in this bin
                for asset_index in np.arange(0,N_a,1):

                    #Get NN proportions actually invested in asset_index for wealth values in the bin
                    NN_prop_asset_index = params["NN_asset_prop_paths"][rows_bin, n_index, asset_index]
                    # Recall:
                    # params["NN_asset_prop_paths"].shape = [N_d, N_rb+1, N_a]  Paths for the proportions or wealth in each asset for given dataset
                    #   params["NN_asset_prop_paths"][j, n, i]:
                    #       for i <= N_a-1 (i is *index*): Proportion of wealth t_n^+ invested in asset i
                    #       at rebal time t_n along sample path j

                    #Get average of proportions for this bin for output
                    Asset_Heatmaps[bin_index, n_index, asset_index] = np.mean(NN_prop_asset_index)
                    # Asset_Heatmaps[j,n,i] =  (for wealth in bin j) *average* proportion invested in asset i at rebal time t_n

            # if bin_index == count_W_bins - 1:
                    # print(bin_count)


    params["DataHeatmaps_W_bin_left_edges"] = W_bin_left_edges
    params["DataHeatmaps_Asset_props"] = Asset_Heatmaps
    params["DataHeatmaps_exploration_props"] = Data_exploration_Heatmaps

    return params




def construct_data_for_DataHeatmaps__ads_stochastic(
        params,  #params dictionary with *trained* NN results and setup as in main code
        W_diff_min = -200.,   #left endpoint of first W_diff bin
        W_diff_max = 1000.,   #right endpoint of last W_diff bin
        delta_W_diff = 5.0   #bin width for W bins
    ):

    #Goal: Construct data used for the DataHeatmap plots for objective "ads_stochastic" with NO trade signals
    # - NOTE: we use this code for ["ads_stochastic", "qd_stochastic", "ir_stochastic", "te_stochastic"]
    #INPUTS: 'W_diff' refers to W(t) using NN minus W(t) using benchmark

    #OUTPUTS:
    # Appends fields to the params dictionary:
    # params["DataHeatmaps_W_diff_bin_left_edges"] = left edges of bins for wealth difference at time t
    # params["DataHeatmaps_Asset_props"]:
    #       #DataHeatmaps_Asset_props[j,n,i] =  (for wealth in bin j) *average* proportion invested in asset i at rebal time t_n

    #-----------------------------------------------
    # Get basic info
    N_d = params["N_d"]
    N_rb = params["N_rb"]
    N_a = params["N_a"]


    #Construct bins for W
    W_diff_bin_left_edges = np.arange(W_diff_min, W_diff_max, delta_W_diff)
    count_W_diff_bins = W_diff_bin_left_edges.shape[0]


    #Initialize outputs
    Asset_Heatmaps = np.empty([count_W_diff_bins,  N_rb, N_a])
    Asset_Heatmaps[:] = np.NaN
    #Asset_Heatmaps[j,n,i] =  (for wealth in bin j) *average* proportion invested in asset i at rebal time t_n


    for n_index in np.arange(0, N_rb, 1): #Loop over rebalancing times

        #Get wealth vector at time (t_n^+) using NN
        W_t_n = params["W"][:,n_index] #W to contain the wealth *after* contribution at t_n

        #Get wealth vector at time (t_n^+) using BENCHMARK
        W_t_n_benchmark = params["benchmark_W_paths"][:, n_index]  # W to contain the wealth *after* contribution at t_n

        #Calculate difference
        W_t_n_diff = W_t_n - W_t_n_benchmark

        #Loop through W bins
        for bin_index in np.arange(0,count_W_diff_bins, 1):

            #Get bin left edge
            bin_left_edge = W_diff_bin_left_edges[bin_index]

            #Get bin right edge
            if bin_index < count_W_diff_bins-1:
                bin_right_edge = W_diff_bin_left_edges[bin_index + 1]
            else:
                bin_right_edge = W_diff_max

            # Get *indices* of wealth values in W_t_n_diff that would fall in each bin
            indices_bin = np.where((W_t_n_diff>= bin_left_edge) & (W_t_n_diff < bin_right_edge) )

            if len(indices_bin[0])>0:   #if there are actually wealth diff values in this bin
                rows_bin = indices_bin[0]   #get the row numbers (i.e. training data paths) where wealth diff values are in each bin

                #----------------------------------------
                #Average out the proportions invested in each asset for wealth in this bin
                for asset_index in np.arange(0,N_a,1):

                    #Get NN proportions actually invested in asset_index for wealth values in the bin
                    NN_prop_asset_index = params["NN_asset_prop_paths"][rows_bin, n_index, asset_index]
                    # Recall:
                    # params["NN_asset_prop_paths"].shape = [N_d, N_rb+1, N_a]  Paths for the proportions or wealth in each asset for given dataset
                    #   params["NN_asset_prop_paths"][j, n, i]:
                    #       for i <= N_a-1 (i is *index*): Proportion of wealth t_n^+ invested in asset i
                    #       at rebal time t_n along sample path j

                    #Get average of proportions for this bin for output
                    Asset_Heatmaps[bin_index, n_index, asset_index] = np.mean(NN_prop_asset_index)
                    # Asset_Heatmaps[j,n,i] =  (for wealth in bin j) *average* proportion invested in asset i at rebal time t_n


    params["DataHeatmaps_W_diff_bin_left_edges"] = W_diff_bin_left_edges
    params["DataHeatmaps_Asset_props"] = Asset_Heatmaps

    return params
