

import numpy as np
import fun_Objective_functions
import fun_utilities
import fun_output_results
import pandas as pd
import datetime
import fun_output_results
import fun_Data_historical_path
import fun_invest_ConstProp_strategy    #for adding constant prop strategy results to the specific training paths

#--------------------------------------------------------------------------
#Output to Excel for testing:
# - NN weights matrices
# - benchmark wealth W mean and stdev for (standardized) feature calc
# - Paths from the TRAINING dataset giving the Top X and bottom X *objective function* and *terminal wealth* values:
#   -- Trading signal paths (features), if applicable
#   -- Asset returns
# - Paths for ROLLING HISTORICAL test giving Top X and bottom X *terminal wealth* values:


def output_training_results_TopBottom_paths(params_TRAIN, #params dictionary AFTER training of NN
                             top_bottom_rows_count = 1, #Number of top and bottom objective function and wealth values of interest
                             output_Excel = False,
                             filename_prefix_for_Excel = "z_"  #used if output_Excel== True
                             ):

    # return dict_dfs_ALL
    dict_dfs_ALL = {}   #initialize dictionary of dfs to write out to Excel or return


    #Get summary of results on the training dataset
    df_summary_results = fun_output_results.output_results_NN(params_TRAIN= params_TRAIN,
                                                              params_TEST= None,
                                                              output_Excel = False)
    dict_dfs_ALL.update({"TRAINING_summary" : df_summary_results})

    # --------------------------------------------------------------------
    #Get NN parameters (weights matrices) from training
    NN = params_TRAIN["NN_object"]

    for layer_id in np.arange(1 ,NN.n_layers_total ,1):  # No weights matrix *into* layer_id=0
        desc = NN.layers[layer_id].description
        x_l = NN.layers[layer_id].x_l  # weights matrix INTO layer_id
        x_l_df = pd.DataFrame(x_l)

        #Write into dictionary
        dict_dfs_ALL.update({"NNweights_into_layer_id_" + str(layer_id) : x_l_df})


    # --------------------------------------------------------------------
    # TERMINAL WEALTH and OBJECTIVE FUNCTION values along EACH TRAINING PATH

    # Get terminal wealth for objective function evaluation
    # - note, we don't want the "after cash withdrawal" value
    # - we want the value on which the optimization is performed
    W_T_true = params_TRAIN["W"][:, -1].copy()

    #Objective function value
    obj_fun_W_T_true = fun_Objective_functions.fun_objective(params_TRAIN)[0]  # Just return function value, not gradients


    # --------------------------------------------------------------------
    # Paths with SMALLEST and LARGEST results: W_T and objective function
    # Get k *smallest* and *largest*  INDICES OF PATHS in no particular order of the Wealth and Objective function values
    # [i.e. associated wealth/objective function VALUES within these lists will not necessarily be sorted!]

    idx_W_T_bottom = fun_utilities.np_array_indices_SMALLEST_k_values_no_sorting(array=W_T_true, k=top_bottom_rows_count)
    idx_W_T_top = fun_utilities.np_array_indices_LARGEST_k_values_no_sorting(array=W_T_true, k=top_bottom_rows_count)
    idx_ObjFun_bottom = fun_utilities.np_array_indices_SMALLEST_k_values_no_sorting(array=obj_fun_W_T_true, k=top_bottom_rows_count)
    idx_ObjFun_top = fun_utilities.np_array_indices_LARGEST_k_values_no_sorting(array=obj_fun_W_T_true, k=top_bottom_rows_count)


    #Get paths for each of these collections of indices

    dict_dfs_W_T_bottom = construct_dict_of_dfs_for_each_path_index(params_TRAIN,
                               path_indices = idx_W_T_bottom,
                               dict_key_prefix = "W_T_bottom_idx_")


    dict_dfs_W_T_top = construct_dict_of_dfs_for_each_path_index(params_TRAIN,
                               path_indices = idx_W_T_top,
                               dict_key_prefix = "W_T_top_idx_")

    dict_dfs_ObjFun_bottom = construct_dict_of_dfs_for_each_path_index(params_TRAIN,
                                                                    path_indices=idx_ObjFun_bottom,
                                                                    dict_key_prefix="ObjFun_bottom_idx_")

    dict_dfs_ObjFun_top = construct_dict_of_dfs_for_each_path_index(params_TRAIN,
                                                                 path_indices=idx_ObjFun_top,
                                                                 dict_key_prefix="ObjFun_top_idx_")

    # append to the output dictionary
    dict_dfs_ALL.update(dict_dfs_W_T_bottom)
    dict_dfs_ALL.update(dict_dfs_W_T_top)
    dict_dfs_ALL.update(dict_dfs_ObjFun_bottom)
    dict_dfs_ALL.update(dict_dfs_ObjFun_top)


    if output_Excel:    #Output contents of dict_dfs_ALL to Excel sheet
                        #contents of each dict_dfs_ALL.keys() will be in a separate worksheet, same workbook!

        timestamp = datetime.datetime.now().strftime('%Y-%m-%d_%H_%M')
        filename = filename_prefix_for_Excel + "_TRAINING_TopBottomPaths_and_NNweights" + ".xlsx"

        with pd.ExcelWriter(filename) as writer:    #we want to write out to multiple sheets

            for key in dict_dfs_ALL.keys():
                    sheet_name = key
                    sheet_data = dict_dfs_ALL[key]
                    sheet_data.to_excel(writer, sheet_name=sheet_name, header=True, index=True)


    return dict_dfs_ALL





def construct_dict_of_dfs_for_each_path_index(params_TRAIN,      #params dict *after* training of NN
                               path_indices,    #np.array of training path indices
                                                #this is the "j" values in
                                                # params["TradSig_train"][j, n, i] and params["Y_train"][j, n, i]
                               dict_key_prefix  # prefix for dictionary keys, used to interpret path index, e.g. "W_T_top"
                            ):
    #return dict_dfs

    #dict_dfs = dictionary of pd.Dataframes.
    # - each df will correspond to the paths (asset returns, training signals and benchmark wealth info for features)
    # of ONE index entry in path_indices

    dict_dfs = {}

    for idx in path_indices:  # Index of training PATH

        df = pd.DataFrame() #each df will be appended to the dictionary

        dict_key =  dict_key_prefix + str(idx)  # index appended to key to ensure unique key

        #--------------------------------------------------------------------------
        # Loop through assets and get paths, making sure it is in the right order, and append to df
        #   params["Y_train"][j, n, i] = Return, along sample path j, over time period (t_n, t_n+1), for asset i
        #       -- IMPORTANT: params["Y_train"][j, n, i] entries are basically (1 + return), so it is ready for multiplication with start value
        #   params["Y_order_train"][i] = column name of asset i used for identification

        for asset_idx in np.arange(0 ,params_TRAIN["N_a"] ,1):
            col_name = params_TRAIN["Y_order_train"][asset_idx]
            t_series = params_TRAIN["Y_train"][idx ,: ,asset_idx].copy()
            df[col_name] = t_series

        # --------------------------------------------------------------------------
        # Get trade signal data, making sure it is in the right order, and append to df
        #   params["TradSig_train"][j, n, i] = Point-in time observation for trade signal i, along sample path j, at rebalancing time t_n;
        #                               can only rely on time series observations <= t_n
        #   params["TradSig_order_train"][i] = column name of trade signal i used for identification
        if params_TRAIN["use_trading_signals_TrueFalse"] == True:
            N_tradsig = len(params_TRAIN["TradSig_order_train"])
            for tradsig_idx in np.arange(0, N_tradsig ,1):
                col_name = params_TRAIN["TradSig_order_train"][tradsig_idx]
                t_series = params_TRAIN["TradSig_train"][idx, :, tradsig_idx].copy()
                df[col_name] = t_series

        # --------------------------------------------------------------------------
        # Append "NaN" row to returns /features to ensure terminal wealth values can be added
        df = df.append(pd.Series(dtype="float64"), ignore_index=True)

        # --------------------------------------------------------------------------
        # WEALTH - benchmark strategy
        # Append the benchmark mean and stdev for features
        # Get benchmark strategy wealth + stdev info to calculate features
        benchmark_W_mean = (np.transpose(params_TRAIN["benchmark_W_mean"])).flatten()
        benchmark_W_std = (np.transpose(params_TRAIN["benchmark_W_std"])).flatten()

        df["benchmark_W_mean"] = benchmark_W_mean
        df["benchmark_W_std"] = benchmark_W_std

        # --------------------------------------------------------------------------
        # WEALTH - NN strategy
        df["W_NN_strategy"] = params_TRAIN["W"][idx ,:]

        # --------------------------------------------------------------------------
        # WEALTH - Benchmark strategy

        # Constant proportion strategy implemented on the TRAINING data
        params_CP_TRAIN = fun_invest_ConstProp_strategy.invest_ConstProp_strategy(prop_const=params_TRAIN["benchmark_prop_const"],
                                                                                  params=params_TRAIN,
                                                                                  train_test_Flag="train")
        df["W_benchmark_strategy"] = params_CP_TRAIN["W"][idx ,:]


        dict_dfs.update({dict_key : df})

    return dict_dfs