#OBJECTIVE: Creates training datasets for NN

#For the ASSET basket, the result will be
#   asset_expblksize_XX[j, n] = return, along sample path j, over time period (t_n, t_n+1),
#                                 for asset "ticker" and
#                                 for bootstrap_exp_block_size "expblksize" given by XX

# -- Shape for each asset's training data sample paths = (N_d,N_rb):
#                   N_d = nr of return paths for the training data,
#                   N_rb = nr of rebalancing events  (columns) for the training data
# -- Note that "returns" paths are basically (1+ return), so it is ready for multiplication with "start value"
#    to create price series

import pandas as pd
import numpy as np
import os
from fun_stationary_bootstrap import stationary_bootstrap   #bootstrapping code
import datetime #used to timestamp.csv files if needed


def run_bootstrap(df_all,   #pd.DataFrame with source data for bootstrapping
                bootstrap_settings, #dictionary with bootstrap settings
                params  #dictionary with the main investment parameters
                  ):

    # RETURNS "params" dictionary with ADDED fields ("purpose" will be either "train" or "test")
    # ASSET return data: always appended
    #   params["Y_"+ purpose] = Y.copy()
    #   params["Y_order_" + purpose] = Y_order
    #where
    # params["Y_"+ purpose][j, n, i] = Return, along sample path j, over time period (t_n, t_n+1), for asset i
    # -- IMPORTANT: params["Y_"+ purpose][j, n, i] entries are basically (1 + return), so it is ready for multiplication with start value
    # params["Y_order_" + purpose][i] = column name of asset i used for identification
    #
    # TRADING SIGNAL data: only appended if params["use_trading_signals_TrueFalse"] == True
    #     params["TradSig_" + purpose] = TradSig.copy()
    #     params["TradSig_order_" + purpose] = TradSig_order

    #where
    # params["TradSig_" + purpose][j, n, i] = Point-in time observation for trade signal i, along sample path j, at rebalancing time t_n;
    #                               can only rely on time series observations <= t_n
    # params["TradSig_order_" + purpose][i] = column name of trade signal i used for identification


    #--- BEFORE running code, decide if output neede for debugging!
    #If TRUE, will output bootstrapped data from every step below,
    #       for a randomly selected *single* path from [0, N_d]
    output_single_path_TEST = False  #Set to FALSE if happy with debugging of code
    if output_single_path_TEST is True:
        output_row_TEST = np.random.choice(np.arange(0,bootstrap_settings["N_d"],1).tolist(), size=1)
        df_step_0 = pd.DataFrame(output_row_TEST, columns=["output_row_TEST"], index=[0])
        df_step_0.to_excel("test_step_0_output_row.xlsx")




    #If output_csv == TRUE, will output .csv files with data
    output_csv = params["output_csv_data_training_testing"]  #output .csv files with the bootstrapped data

    #---------------------------------------------------------------------------------
    #Unpack bootstrap settings
    N_d = bootstrap_settings["N_d"]
    purpose = bootstrap_settings["purpose"] #train or test
    yyyymm_start = bootstrap_settings["data_bootstrap_yyyymm_start"]
    yyyymm_end = bootstrap_settings["data_bootstrap_yyyymm_end"]
    bootstrap_exp_block_size = bootstrap_settings["data_bootstrap_exp_block_size"]
    bootstrap_fixed_block = bootstrap_settings["data_bootstrap_fixed_block"]
    bootstrap_n_tot = bootstrap_settings["data_bootstrap_n_tot"]
    data_delta_t = bootstrap_settings["data_bootstrap_delta_t"]

    #---------------------------------------------------------------------------------
    #Get structural parameters

    T = params["T"]  #Investment time horizon, in years
    N_rb = params["N_rb"] #Nr of equally-spaced rebalancing events in [0,T]
    delta_t = params["delta_t"]  # Rebalancing time interval, should be some integer multiple of data_delta_t

    #for writing the bootstrap settings dictionary to .csv, it is useful to append this to bootstrap_settings
    bootstrap_settings["T"] = T
    bootstrap_settings["N_rb"] = N_rb
    bootstrap_settings["delta_t"] = delta_t

    #---------------------------------------------------------------------------------
    #Get column names to bootstrap
    column_names_all = df_all.columns.tolist()

    #Need to distinguish between assets and trading signals
    column_names_assets = params["asset_basket"]["basket_columns"]
    N_a = len(column_names_assets)  #number of assets

    if params["use_trading_signals_TrueFalse"] == True:
        column_names_trading_signals = params["trading_signal_basket"]["basket_columns"]
        N_tradingsignals = len(column_names_trading_signals)    #number of trading signals


    # ---------------------------------------------------------------------------------
    # Select dates of interest
    if yyyymm_end is not None:
        df_all = df_all.loc[df_all.index <= yyyymm_end].copy()
    if yyyymm_start is not None:
        df_all = df_all.loc[df_all.index >= yyyymm_start].copy()

    # ---------------------------------------------------------------------------------
    # DO BOOTSTRAPPING

    # Create output dictionary:
    #   For ASSETS (i.e. col in column_names_assets):
    #       each key is a column name i.e. corresponding to an asset return series
    #       each associated value is the (N_d, N_rb) set of (1+returns) used in training of NN for asset given by key


    #   For TRADING SIGNALS (i.e. col in column_names_trading_signals)
    #       each key is a column name i.e. corresponding to a time series
    #       each associated value is the (N_d, N_rb) set of POINT IN TIME observations as at the rebalancing time
    #       used for trading signals in the feature vector of the NN


    output_dict = {}

    for col in column_names_all:
        #Initialize
        output_dict.update({col : np.zeros([int(N_d),int(N_rb)])})

    #Input data for bootstrapping
    in_array = df_all.to_numpy()  # convert pd.DataFrame with historical data to numpy

    for output_row in np.arange(0, int(N_d), 1):  # Loop over number of output rows N_d
        # For each output_row, simultaneously construct that output row for all training datasets,
        # for all rebalancing events, for all assets

        out_array, block_numbers = stationary_bootstrap(in_array,
                                                            bootstrap_n_tot,
                                                            bootstrap_exp_block_size,
                                                            bootstrap_fixed_block)


        # ----------------
        #Update user Bootstrap progress
        if (N_d > 1000 ):
            if output_row in np.append(np.arange(0, N_d, int(0.05 * N_d)), N_d):
                print( str( (output_row)/N_d * 100) + "% of bootstrap sample done.")


        # convert out_array_ret to pd.DataFrame using same column names (no index date column!)
        df_out_array = pd.DataFrame(out_array, columns=column_names_all)


        #split in assets and trading signals
        df_out_array_assets = df_out_array[column_names_assets].copy()


        if params["use_trading_signals_TrueFalse"] == True:
            df_out_array_trading_signals = df_out_array[column_names_trading_signals].copy()


        # Output output_row_TEST if required
        if (output_single_path_TEST is True) and (output_row  == output_row_TEST):
            df_out_array.to_excel("test_step_1_df_out_array.xlsx")

            #Also get block numbers
            df_block_numbers = pd.DataFrame(block_numbers)
            df_block_numbers.to_excel("test_step_1_df_block_numbers.xlsx")

        #---------------------------------------------------------------
        # Construct "asset price" series for assets
        n_return_obs = out_array.shape[0]
        n_price_obs = n_return_obs + 1
        n_assets = len(column_names_assets)

        # construct price series starting from 100 using the returns
        # will have ONE EXTRA ROW compared to returns series
        out_array_prices = np.zeros([n_price_obs, n_assets])
        out_array_prices[0, :] = 100
        out_array_prices[1:n_price_obs, :] = 100 * (1 + df_out_array_assets).cumprod()

        df_out_array_asset_prices = pd.DataFrame(out_array_prices, columns=column_names_assets)

        # ---------------------------------------------------------------
        # Get row indices of prices at intervals required for training data
        row_indices = np.arange(0, n_price_obs, delta_t / data_delta_t)
        row_indices = row_indices.astype(int)

        #Get prices corresponding to row_indices, calculate returns (i.e. this accumulates the asset returns)
        df_out_array_asset_prices = df_out_array_asset_prices.iloc[row_indices].copy()
        df_bootstrapped_asset_returns = df_out_array_asset_prices.pct_change()
        # Drop the first row of NaN returns
        df_bootstrapped_asset_returns.drop(axis=0, index=df_bootstrapped_asset_returns.index[0], inplace=True)

        # Add one to get in right format for output (1 + return) for assets
        df_bootstrapped_asset_returns = df_bootstrapped_asset_returns + 1


        # ---------------------------------------------------------------
        # Get trading_signals  at intervals required for training data
        # - Trading signals are POINT IN TIME! No need for accumulation (like in the case of asset returns)
        #       over rebalancing time interval
        # - However, we need values of trading signals at the START of the time interval

        if params["use_trading_signals_TrueFalse"] == True:
            n_trading_signals = len(column_names_trading_signals)

            #Insert extra (first) row just to align the indices with the bootstrapped asset returns
            # - LAGGING of observations has been done in the processing of market data, so this will be right
            out_array_trading_signals = np.zeros([n_price_obs, n_trading_signals])
            out_array_trading_signals[0, :] = 0.0
            out_array_trading_signals[1:n_price_obs, :] = df_out_array_trading_signals

            df_out_array_trading_signals = pd.DataFrame(out_array_trading_signals, columns=column_names_trading_signals)

            #ROW indices for TRADING SIGNALS:
            # - Trading signals are POINT IN TIME (not accumulated over e.g. the year like returns)
            # - Market data calc does lagging of observations
            # - We want to get the signal at the *START* of the period of returns accumulation, *NOT* at the END!
            row_indices_TradSig = row_indices[:-1] + 1 # We have inserted a row to make this calc work out

            # Do NOT drop the first row of TRADING SIGNALS, we need it!
            df_bootstrapped_trading_signals = df_out_array_trading_signals.iloc[row_indices_TradSig].copy()


            # For merging with asset returns, *rename* the index column of df_bootstrapped_trading_signals
            df_bootstrapped_trading_signals.set_index(keys = df_bootstrapped_asset_returns.index.values,
                                                      inplace=True)


        # ---------------------------------------------------------------
        # MERGE asset_returns with trading_signals, if needed
        if params["use_trading_signals_TrueFalse"] == True:
            df_bootstrapped = df_bootstrapped_asset_returns.merge(df_bootstrapped_trading_signals,
                                                              how="inner", left_index=True, right_index=True)
        else:
            df_bootstrapped = df_bootstrapped_asset_returns.copy()

        # ---------------------------------------------------------------
        # Append to each ticker in output_dict data
        for col in column_names_all:
            output_dict[col][output_row, :] = np.transpose(df_bootstrapped[col].to_numpy())

        # ---------------------------------------------------------------
        # Output output_row_TEST if required
        if (output_single_path_TEST is True) and (output_row  == output_row_TEST):
            df_bootstrapped.to_excel("test_step_2_df_bootstrapped_path_final.xlsx")


    #END LOOP over number of output rows

    # --------------------------------------------------------------------------------------------------
    #Construct OUTPUT data

    # Initialize ASSET RETURNS output data: Y.shape = (N_d, N_rb, N_a)
    Y =  np.zeros([N_d, N_rb, N_a])
    Y_order = column_names_assets   #to keep record of order of columns, this is ensured with the population of Y below

    # Y[j, n, i] = Return, along sample path j, over time period (t_n, t_n+1), for asset i
    # -- IMPORTANT: Y[j, n, i] entries are basically (1 + return), so it is ready for multiplication with start value
    for asset_index in np.arange(0,N_a,1):
        asset_col = Y_order[asset_index]
        Y[:,:,asset_index] = output_dict[asset_col].copy()

    #Initialize TRADING SIGNALS output data
    # TradSig[j, n, i] = Point-in time observation for trade signal i, along sample path j, at rebalancing time t_n;
    #                      can only rely on time series observations < t_n
    if params["use_trading_signals_TrueFalse"] == True:
        TradSig =  np.zeros([N_d, N_rb, N_tradingsignals])
        TradSig_order = column_names_trading_signals    #to keep record of order of columns

        if purpose == "train":
            TradSig_MEAN_train = np.zeros(N_tradingsignals)
            TradSig_STDEV_train = np.zeros(N_tradingsignals)

        for tradsig_index in np.arange(0,N_tradingsignals,1):
            tradsig_col = TradSig_order[tradsig_index]
            TradSig[:,:,tradsig_index] = output_dict[tradsig_col].copy()

            # ------------------------------------------------------------------
            # Calculate values used in standardization of trading signals features
            # The actual stdization only happens when features are calculated
            if purpose == "train":
                TradSig_MEAN_train[tradsig_index] = np.mean(TradSig[:,:,tradsig_index])
                TradSig_STDEV_train[tradsig_index] = np.std(TradSig[:,:,tradsig_index], ddof=1)



    #---------------------------------------------------------------------------------
    #Write out to .csv files if required

    if output_csv == True:

        #Get datetime stamp
        date_time = datetime.datetime.now()
        date_time_str = date_time.strftime("%m-%d-%Y__%H_%M_%S")

        #Write out the bootstrap settings:
        temp = pd.DataFrame(bootstrap_settings.values(), index=bootstrap_settings.keys())

        temp.to_csv("zBootstrap_SETTINGS_ " + date_time_str + ".csv", header=False, index=True)

        for col in column_names_all:

            #first convert to dataframe
            temp = pd.DataFrame(output_dict[col])

            #Construct filename using column name

            filename = "zBootstrapped_" + col


            temp.to_csv(filename + date_time_str + ".csv", header = False, index= False)

            #Each outputted .csv file in this format
            #   ticker_expblksize_XX[j, n] = Return, along sample path j, over time period (t_n, t_n+1),
            #                                 for asset "ticker" and bootstrap_exp_block_size = XX

    # ---------------------------------------------------------------------------------
    #Finally, set RETURN values

    #- ASSET return data: always appended
    params["Y_"+ purpose] = Y.copy()
    params["Y_order_" + purpose] = Y_order

    # - TRADING SIGNAL data: only appended if params["use_trading_signals_TrueFalse"] == True
    if params["use_trading_signals_TrueFalse"] == True:
        params["TradSig_" + purpose] = TradSig.copy()
        params["TradSig_order_" + purpose] = TradSig_order

        if purpose== "train":
            params["TradSig_MEAN_train"] = TradSig_MEAN_train.copy()
            params["TradSig_STDEV_train"] = TradSig_STDEV_train.copy()

    return params


