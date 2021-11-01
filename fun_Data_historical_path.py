
import copy
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd



def assign_historical_path_to_params_for_testing(params,  #params dictionary with market data and investment params setup as in main code
                                    yyyymm_path_start  # yyyymm START of HISTORICAL PATH for returns and trading signals
                                                # yyyymm END is *calculated* based on params info (T, delta_t)
                                    ):

    # Assign single (subset of) historical path to params TESTING data in order to evaluate NN strategy
    #   - Needs to run this after bootstrapping SETUP, since it uses some params set (e.g. order of training)
    #               during the preparation for bootstrapping

    #OUTPUT:    essentially replaces the testing dataset in params and returns params_new
    # return params_new
    # updated fields:
    #  params_new["test_TrueFalse"] = True
    #  params_new["N_d_test"] = 1   #Only ONE historical path (subset)
    #  params_new["Y_test"][0,n,i] = (1 + return), along a SINGLE historical sample path,
    #                                over time period (t_n, t_n+1), for asset i
    #  params_new["Y_order_test"] = params["Y_order_train"]

    #  if params_new["use_trading_signals_TrueFalse"] == True:
    #       params_new["TradSig_order_test"] = params["TradSig_order_train"]
    #       params_new["TradSig_test"][0,n,i] = Point-in time observation for trade signal i,
    #                                           along a SINGLE historical sample path at rebalancing time t_n
    # params_new["df_historical_path_ret"] = pd.DataFrame of historical path of returns & trading signals used

    params_new = copy.deepcopy(params)  #create copy

    #Create NEW TESTING dataset

    params_new["test_TrueFalse"] = True #set flag for testing
    params_new["N_d_test"] = 1   #one historical path


    #Create local variables for convenience
    column_names_assets = params["Y_order_train"]
    N_d = params_new["N_d_test"]
    N_a = params_new["N_a"]
    N_rb = params_new["N_rb"]

    #Trading signals
    if params_new["use_trading_signals_TrueFalse"] == True:
        column_names_trading_signals = params_new["TradSig_order_train"]
        N_tradingsignals = len(column_names_trading_signals)  # number of trading signals


    #Get historical path using the function
    df_historical_path_ret = get_historical_path_returns(params_new, yyyymm_path_start)
    #   NOTE: - asset returns reported in row yyyymm is from the BEGINNING of that month to the NEXT index (next event)
    #              i.e. it is FORWARD looking
    #               also, asset returns are (1 + return) as required for training + evaluation
    #        - trading signals is point in time: reported in yyyymm is at the BEGINNING of that month for the PRECEDING months
    #              i.e. it is BACKWARD looking
    params_new["df_historical_path_ret"] = df_historical_path_ret   #for output

    #------------------------------------------------------------------------------
    # ASSET returns along historical path
    # Initialize ASSET RETURNS output data: Y.shape = (N_d, N_rb, N_a)
    Y =  np.zeros([N_d, N_rb, N_a])
    Y_order = column_names_assets   #to keep record of order of columns, this is ensured with the population of Y below

    # Y[j, n, i] = Return, along sample path j, over time period (t_n, t_n+1), for asset i
    # -- IMPORTANT: Y[j, n, i] entries are basically (1 + return), so it is ready for multiplication with start value
    for asset_index in np.arange(0,N_a,1):
        asset_col = Y_order[asset_index]
        Y[0,:,asset_index] = df_historical_path_ret[asset_col].to_numpy()

    params_new["Y_order_test"] = Y_order
    params_new["Y_test"] = Y.copy()             #params_new["Y_test"][0,n,i]


    #------------------------------------------------------------------------------
    #TRADING SIGNALS for the historical path
    # TradSig[j, n, i] = Point-in time observation for trade signal i, along sample path j, at rebalancing time t_n;
    #                      can only rely on time series observations <= t_n
    if params_new["use_trading_signals_TrueFalse"] == True:
        TradSig =  np.zeros([N_d, N_rb, N_tradingsignals])
        TradSig_order = column_names_trading_signals   #to keep record of order of columns
        for tradsig_index in np.arange(0,N_tradingsignals,1):
            tradsig_col = TradSig_order[tradsig_index]
            TradSig[:,:,tradsig_index] = df_historical_path_ret[tradsig_col].to_numpy()

        params_new["TradSig_order_test"] = TradSig_order
        params_new["TradSig_test"] = TradSig.copy()


    return params_new



def get_historical_path_returns(params,  #params dictionary WITH MARKET DATA in params["bootstrap_source_data"]
                                 #  and investment params setup as in main code
                        yyyymm_path_start  # yyyymm START of HISTORICAL PATH for trading signals
                                                # yyyymm END is *calculated* based on params info (T, delta_t)
                       ):
    #return df_historical_path_ret

    #OBJECTIVE: Gets historical path of (1 + asset return) for each asset, and point-in-time trading signals
    #            starting from yyyymm_path_start, at the same intervals as rebalancing events
    #            until final rebalancing event

    #NOTE: asset returns reported in row yyyymm is from the BEGINNING of that month to the NEXT index (next event)
    #              i.e. it is FORWARD looking
    #               also, asset returns are (1 + return) as required for training + evaluation
    #      trade signals is point in time: reported in yyyymm is at the BEGINNING of that month for the PRECEDING months
    #              i.e. it is BACKWARD looking


    T = params["T"]
    delta_t = params["delta_t"]  # rebalancing time interval


    # -----------------------------------------------------------------------------------------
    # Get historical path of trade signals:

    # Get market data = "bootstrap_source_data" in order to get a path of trading signals
    # Note that the actual bootstrapped data for training + testing would use a SUBSET of the dates here
    # - so we can use true out-of-sample market data points if we wanted to
    df_source_data = params["bootstrap_source_data"]
    df_source_data_yyyymm_start = np.min(df_source_data.index)  # beginning of source data
    df_source_data_yyyymm_end = np.max(df_source_data.index)  # end of source data

    # get interval between df_source_data points
    source_data_delta_t = params["asset_basket_data_settings"]["data_read_delta_t"]


    # Number of consecutive observations from df_source_data we need to construct a path
    nobs = int(T / source_data_delta_t)

    idx_start = df_source_data.index.get_loc(yyyymm_path_start)
    idx_end = idx_start + nobs  # Note: idx_end will be EXCLUDED below


    try:
        temp = df_source_data.iloc[idx_end -1].name
    except:
        raise ValueError("PVS error in 'get_historical_path_returns': End date out of bounds. Select earlier yyyymm_path_start.")
        sys.exit(1)

    # Select subset of source data with the historical path of interest
    df_paths = df_source_data.iloc[idx_start:idx_end].copy()  # idx_end will be EXCLUDED

    # Get row indices of at intervals required for portfolio rebalancing events
    row_indices_rebal = np.arange(0, nobs, delta_t / source_data_delta_t)
    row_indices_rebal.astype(int)

    #Get rebal events index names
    rebal_events_index_names = df_paths.iloc[row_indices_rebal].index

    # Extract trading signals: Recall that these are POINT IN TIME
    if params["use_trading_signals_TrueFalse"] == True:

        tradsig_columns = params["trading_signal_basket"]["basket_columns"]

        # Get those entries corresponding to rebalancing events (point in time)
        df_TradSig_rebal_events = df_paths.iloc[row_indices_rebal][tradsig_columns].copy()


    # Extract asset returns (same frequency as source data)
    asset_columns = params["asset_basket"]["basket_columns"]
    df_asset_returns = df_paths[asset_columns].copy()

    # ---------------------------------------------------------------
    # Construct "asset price" series for assets
    n_return_obs = df_asset_returns.shape[0]
    n_price_obs = n_return_obs + 1
    n_assets = df_asset_returns.shape[1]    #nr of columns

    # construct price series starting from 100 using the returns
    # will have ONE EXTRA ROW compared to returns series
    out_array_prices = np.zeros([n_price_obs, n_assets])
    out_array_prices[0, :] = 100
    out_array_prices[1:n_price_obs, :] = 100 * (1 + df_asset_returns).cumprod()


    df_asset_prices = pd.DataFrame(out_array_prices, columns=asset_columns)
    #           Column names a bit confusing (returns) but convenient to keep using them

    #Get row indices, INCLUDING first row, required for rebalancing events
    row_indices_prices = np.arange(0, n_price_obs, delta_t / source_data_delta_t)
    row_indices_prices.astype(int)

    #Calculate asset returns over each rebalancing period
    df_asset_prices = df_asset_prices.iloc[row_indices_prices].copy()
    df_asset_returns = df_asset_prices.pct_change(periods=1)

    df_asset_returns.drop(index = 0, axis=0, inplace= True) #drop the first row with nan values
    df_asset_returns.index = rebal_events_index_names.copy()    #rename to yyyymm

    #Get one plus return for assets
    df_asset_returns = df_asset_returns + 1

    #Construct output dataframe
    if params["use_trading_signals_TrueFalse"] == True:
        df_historical_path_ret = df_asset_returns.merge(right = df_TradSig_rebal_events, how = "inner",
                                                        left_index = True, right_index = True)
    else:
        df_historical_path_ret = df_asset_returns.copy()


    return df_historical_path_ret


