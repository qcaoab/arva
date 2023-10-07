# OBJECTIVES: Output **and* PLOTS PERCENTILES of NN control (proportions of wealth invested in each asset)
#               and PERCENTILES OF WEALTH on given dataset over time

# Outputs plots of:
# - Wealth percentiles over time
# - Proportion of wealth in each asset percentiles over time
# - Withdrawal amount percentiles over time

import copy
import pandas as pd
import numpy as np
import datetime
import matplotlib.pyplot as plt


def get_df_Pctile_paths( params # dictionary with parameters and results from NN investment (TRAINING or TESTING)
                        ):
    #Objective: appends the dataframe of percentiles of asset proportions and wealth over time to params
    # - NO plotting or excel output here

    #Returns, params: ADDED field: params["df_pctiles_ALL"]


    params = copy.deepcopy(params)


    df_pctiles_ALL = output_Pctile_paths(params = params)   #use defaults to get just dataframe


    #Add extra fields to params
    params["df_pctiles_ALL"] = df_pctiles_ALL

    return params

def output_Pctile_paths(params,  # dictionary with parameters and results from NN investment (TRAINING or TESTING)
                        pctiles = None,  #E.g. [20,50,80] List of percentiles to output and/or plot
                        output_Excel =False,  # write the result to Excel
                        filename_prefix_for_Excel ="z_",  # used if output_Excel is True
                        save_Figures = False,  # Plots and save figures in format specified below
                        save_Figures_format="png",
                        fig_filename_prefix="z_",
                        W_max = 2000., #Maximum wealth value for wealth percentiles graph
                        ):

    #OUTPUTS:
    # if output_Excel is True: outputs a spreadsheet with paths of asset proportion percentiles and wealth percentiles over time
    # if save_Figures is True: outputs graphs of paths of asset proportion percentiles  and wealth percentiles over time

    #RETURNS:
    # df_pctiles_ALL: pandas.DataFrame with chosen percentiles for NN asset proportions and wealth over time on data in params

    if pctiles is None:
        pctiles = [20,50,80]    #set defaults


    #Get basic info
    N_d = params["N_d"]
    N_rb = params["N_rb"]
    N_a = params["N_a"]


    #--------------------------------------------------------------------
    #Asset names
    basket_asset_names = params["asset_basket"]["basket_timeseries_names"] #Get asset names
    basket_asset_columns = params["asset_basket"]["basket_columns"]    #Get asset column names
    Y_order_train = params["Y_order_train"] #Order in which asset columns was used for training

    #match up the asset names with the order in which they were used for training (should be the same)
    asset_names = []    #asset_names will match up with Y_order_train columns
    for col in Y_order_train:
        idx = basket_asset_columns.index(col)   #look up the index of col in basket_asset_columns
        asset_names.append(basket_asset_names[idx]) #get the asset name corresponding to this index

    asset_names.append("Withdrawals")

    #initialize
    pctiles_asset_ALL = {}  #will be the full dictionary written out

    withdraw_dummy = 1 if params["nn_withdraw"] else 0
    # --------------------------------------------------------------------
    # Loop over output nodes of NN  [Add one more loop execution to calc WEALTH percentiles]
    for node_index in np.arange(0,N_a+2,1): #Add one more loop execution to calc WEALTH percentiles


        if node_index < N_a+withdraw_dummy:  # ASSET nodes

            # Get ASSET proportion investment percentile time series for *THIS asset*
            pctiles_asset = get_dict_pctiles(data_set=params["NN_asset_prop_paths"][:,:,node_index],  # data for pctile calc on each *column*
                                         pctiles=pctiles,  # List of percentiles to calculate
                                             dict_key_prefix=asset_names[node_index]
                                         )
            # Recall:
            #   params["NN_asset_prop_paths"].shape = [N_d, N_rb+1, N_a]  Paths for the proportions or wealth in each asset for given dataset
            #       params["NN_asset_prop_paths"][j, n, i]: for i <= N_a-1 (i is *index*): Proportion of wealth t_n^+ invested in asset i
            #                                               at rebal time t_n along sample path j


            #Append to dictionary
            pctiles_asset_ALL.update(pctiles_asset)


            if save_Figures:
                #Avoids cutting off the xlabel and sides of box
                from matplotlib import rcParams
                rcParams.update({'figure.autolayout': True})


                #ASSET percentiles PLOT
                fig, ax = plt.subplots()
                x_axis_time = np.arange(0,N_rb,1) * params["delta_t"]   #No asset percentiles at terminal time

                for key in pctiles_asset.keys():
                    ax.plot(x_axis_time, pctiles_asset[key], label = key)

                ax.set_xlim(np.min(x_axis_time), np.max(x_axis_time))
                if node_index == N_a:
                    ax.set_ylim(30,65) #make slightly higher than 1.0 so we can see if all wealth is invested in an asset
                    ax.set_ylabel("Withdrawal amount", fontsize=10)
                    ax.set_xlabel("Time (years)", fontsize=10)
                    ax.legend(loc = "upper right")
                    ax.set_title("NN: Percentiles of " + asset_names[node_index], fontsize=11)

                else:
                    ax.set_ylim(0.,1.1) #make slightly higher than 1.0 so we can see if all wealth is invested in an asset
                    ax.set_ylabel("Proportion of wealth", fontsize=10)
                    ax.set_xlabel("Time (years)", fontsize=10)
                    ax.legend(loc = "upper right")
                    ax.set_title("NN: Percentiles proportion in asset " + asset_names[node_index], fontsize=11)


        if node_index == N_a + withdraw_dummy :  # Fake node index, just to use for Wealth  percentiles
            # Get WEALTH percentiles
            pctiles_wealth = get_dict_pctiles(data_set=params["W"],  # data for pctile calc on each *column*
                                            pctiles=pctiles,  # List of percentiles to calculate
                                            dict_key_prefix = "Wealth"
                                         )
            #Append to output dictionary
            pctiles_asset_ALL.update(pctiles_wealth)


            if save_Figures:
                #Avoids cutting off the xlabel and sides of box
                from matplotlib import rcParams
                rcParams.update({'figure.autolayout': True})


                #WEALTH percentiles PLOT
                fig, ax = plt.subplots()
                x_axis_time = np.arange(0,N_rb+1,1) * params["delta_t"]

                for key in pctiles_wealth.keys():
                    ax.plot(x_axis_time, pctiles_wealth[key], label = key)

                ax.set_xlim(np.min(x_axis_time), np.max(x_axis_time))
                ax.set_ylim(0.,W_max)
                ax.set_ylabel("Wealth", fontsize=10)
                ax.set_xlabel("Time (years)", fontsize=10)
                ax.legend(loc = "upper right")
                ax.set_title("NN: Percentiles of wealth", fontsize=11)


        #Save figure
        if save_Figures:

            timestamp = datetime.datetime.now().strftime('%Y-%m-%d_%H_%M')

            if node_index < N_a + withdraw_dummy: # ASSET node
                fig_filename = fig_filename_prefix +  "timestamp_" + timestamp + "_Pctiles_asset_" \
                               +  str(node_index) + "_" + asset_names[node_index] \
                                + "." + save_Figures_format

            elif node_index == N_a + withdraw_dummy: #fake node to deal with wealth
                fig_filename = fig_filename_prefix +  "timestamp_" + timestamp + "_Pctiles_Wealth" \
                                + "." + save_Figures_format

            plt.savefig(fig_filename, format = save_Figures_format, bbox_inches = "tight")

    #plt.show()
    plt.close(fig="all")

    # ------------------------------------------------------------------------------------------------
    # Define function return
    df_pctiles_ALL = pd.DataFrame.from_dict(pctiles_asset_ALL, orient="index")

    # ------------------------------------------------------------------------------------------------
    # OUTPUT to Excel if required
    if output_Excel:

        timestamp = datetime.datetime.now().strftime('%Y-%m-%d_%H_%M')
        filename = filename_prefix_for_Excel  + "timestamp_" + timestamp + "_Pctiles_ALL"

        df_pctiles_ALL.to_excel(filename + ".xlsx")


    return df_pctiles_ALL



def get_dict_pctiles(data_set, #data for pctile calc on each *column*, assumed to be different time, e.g. [N_d, N_rb + 1]
                     pctiles,    #List of percentiles to calculate
                     dict_key_prefix = ""   #Prefix for dictionary key (e.g. Asset name)
                     ):

    dict_pctiles = {}  #Dictionary with percentile time series to plot

    n_cols = data_set.shape[1]   #each *column* assumed to be different rebal time

    for pctile in pctiles:

        #Initialize timeseries_pctile: will contain the path of selected pctile
        timeseries_pctile = np.zeros(n_cols)

        #Loop through rebal times (columns) and get pctile for each rebal time
        for n_index in np.arange(0, n_cols):

            timeseries_pctile[n_index] = np.percentile(data_set[:,n_index], pctile) #get pctile

        #Append to output dictionary
        dict_pctiles[dict_key_prefix + "_pctile_" + str(pctile)] = timeseries_pctile


    return dict_pctiles