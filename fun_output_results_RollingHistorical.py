
import fun_test_NN
import fun_invest_ConstProp_strategy
import fun_Data_historical_path
import datetime
import pandas as pd
import numpy as np
import fun_utilities
import fun_output_results


def output_results_RollingHistorical(params_TRAIN, #params dictionary AFTER training of NN
                                     top_bottom_rows_count = 1, #Number of top and bottom wealth values of interest
                                     fixed_yyyymm_list = None, #[198912, 199001] LIST of yyyymm_start months of interest
                                     output_Excel=False,
                                     output_only_for_fixed = False, #output ONLY results for fixed_yyyymm_list
                                     filename_prefix_for_Excel="z_"  # used if output_Excel== True
                                     ):

    #Objective: Rolling historical path assessment: test NN strategy and benchmark strategy on actual single
                                    # historical data path, starting in each month and investing for the duration

    if fixed_yyyymm_list is None:
        if output_only_for_fixed is True:
            raise ValueError("PVSerror in output_results_RollingHistorical: fixed_yyyymm_list is None but output_only_for_fixed is True.")

    dict_dfs_ALL = {}   #each df in this dictionary will be written out to Excel if output_Excel = True

    # --------------------------------------------------------------------
    #Get summary of results on the training dataset
    df_summary_results = fun_output_results.output_results_NN(params_TRAIN= params_TRAIN,
                                                              params_TEST= None,
                                                              output_Excel = False)
    dict_dfs_ALL.update({"TRAINING_summary" : df_summary_results})

    # --------------------------------------------------------------------
    # ROLLING HISTORICAL TEST for all available starting dates
    W_paths_HIST_test_NN_dict = {}  # Dict with HISTORICAL WEALTH paths obtained starting at month yyyymm_path_start
    # in the historical data and investing according to the trained NN strategy
    # format {yyyymm_path_start: params_HIST_test_NN["W"]}

    W_paths_HIST_test_BENCHMARK_dict = {} # Dict with HISTORICAL WEALTH paths obtained starting at month yyyymm_path_start
    # in the historical data and investing according to the BENCHMARK strategy

    W_T_HIST_test_NN_dict = {}  # {yyyymm_path_start: params_HIST_test_NN["W_T"].flatten()} #just the terminal wealth
    W_T_HIST_test_benchmark_dict = {}  # {yyyymm_path_start: params_HIST_test_benchmark["W_T"].flatten()} #just the terminal wealth


    #Paths for each yyyymm path start
    NN_asset_prop_paths_dict = {}   #paths of asset proportions for each yyyymm path start
    Feature_phi_paths_dict = {} #paths of features for each yyyymm path start
    PRP_scores_paths_dict = {}  #paths of PRP scores for each yyyymm path start


    yyyymm_available = params_TRAIN["bootstrap_source_data"].index  # All available start dates
    yyyymm_start_min = int(min(yyyymm_available))  # Can change this
    yyyymm_start_max = yyyymm_available[-int(12 * params_TRAIN["T"])]  # Start dates up to and incl yyyymm_start_max
    # will have a full path of outcomes in the historical data

    #If we are interested only in the fixed dates, replace yyyymm_available
    if output_only_for_fixed is True:
        yyyymm_available = fixed_yyyymm_list
    else:
        yyyymm_available = [x for x in yyyymm_available if (x <= yyyymm_start_max and x >= yyyymm_start_min)]


    for yyyymm in yyyymm_available:
        yyyymm_path_start = yyyymm

        #Note: Single path is assigned to the TESTING data
        params_HIST_test = fun_Data_historical_path.assign_historical_path_to_params_for_testing(params=params_TRAIN,
                                                                                                 yyyymm_path_start=yyyymm_path_start)

        # Constant proportion strategy [BENCHMARK] implemented on the historical path data
        params_HIST_test_benchmark = fun_invest_ConstProp_strategy.invest_ConstProp_strategy(prop_const= params_TRAIN["benchmark_prop_const"] ,
                                                                                             params=params_HIST_test,
                                                                                             train_test_Flag="test")




        #Benchmark strategy results
        W_paths_HIST_test_BENCHMARK_dict.update({yyyymm_path_start: params_HIST_test_benchmark["W"].flatten()})
        W_T_HIST_test_benchmark_dict.update({yyyymm_path_start: params_HIST_test_benchmark["W_T"].flatten()})

        # Also add terminal wealth vector from constant proportion strategy (for ADS and IR objectives)
        if params_HIST_test["obj_fun"] in ["ads_stochastic", "qd_stochastic", "ir_stochastic", "te_stochastic"]:
            params_HIST_test["benchmark_W_T_vector_test"] = params_HIST_test_benchmark["W_T"].copy()  # terminal wealth as a vector (one entry for each path)
            params_HIST_test["benchmark_W_paths_test"] = params_HIST_test_benchmark["W"].copy()

        # NN results on historical path data:
        #   NN_asset_prop_paths and PRPscores will contain only results for this SINGLE historical path
        params_HIST_test_NN = fun_test_NN.test_NN(F_theta=params_TRAIN["F_theta"],
                                                  NN_object= params_TRAIN["NN_object"],
                                                  params=params_HIST_test
                                                  )

        #NN strategy results: Wealth paths
        W_paths_HIST_test_NN_dict.update({yyyymm_path_start: params_HIST_test_NN["W"].flatten()})
        W_T_HIST_test_NN_dict.update({yyyymm_path_start: params_HIST_test_NN["W_T"].flatten()})


        #Asset prop paths
        temp_col_names = params_HIST_test_NN["asset_basket"]["basket_timeseries_names"]
        temp_col_names = [("Prop_" + col) for col in temp_col_names]
        df_temp = pd.DataFrame(np.squeeze(params_HIST_test_NN["NN_asset_prop_paths"]), columns=temp_col_names)

        NN_asset_prop_paths_dict.update({yyyymm_path_start: df_temp})

        #Feature paths  Feature_phi_paths_dict
        temp_col_names = params_HIST_test_NN["feature_order"]
        temp_col_names = [("Feature_" + col) for col in temp_col_names]
        df_temp = pd.DataFrame(np.squeeze(params_HIST_test_NN["Feature_phi_paths"]), columns=temp_col_names)

        Feature_phi_paths_dict.update({yyyymm_path_start: df_temp})


        #PRP scores
        if params_HIST_test_NN["PRP_TrueFalse"] is True:
            temp_col_names = params_HIST_test_NN["feature_order"]
            temp_col_names = [("PRPscore_" + col) for col in temp_col_names]
            df_temp = pd.DataFrame(np.squeeze(params_HIST_test_NN["PRPscores"]), columns=temp_col_names)

            PRP_scores_paths_dict.update({yyyymm_path_start: df_temp})




    # - end of loop
    # Index = yyyymm_path_start, Wealth path in COLUMNS
    W_paths_HIST_test_NN_df = pd.DataFrame.from_dict(data=W_paths_HIST_test_NN_dict, orient="index")
    W_paths_HIST_test_BENCHMARK_df = pd.DataFrame.from_dict(data=W_paths_HIST_test_BENCHMARK_dict, orient="index")

    # Terminal wealth
    W_T_HIST_test_NN_df = pd.DataFrame.from_dict(data=W_T_HIST_test_NN_dict, orient="index", columns=["NN_strategy"])
    W_T_HIST_test_benchmark_df = pd.DataFrame.from_dict(data=W_T_HIST_test_benchmark_dict, orient="index",
                                                        columns=["Benchmark_strategy"])
    W_T_HIST_test_df = W_T_HIST_test_NN_df.merge(right=W_T_HIST_test_benchmark_df, how="inner", left_index=True,
                                                 right_index=True)

    #Append for output
    dict_dfs_ALL.update({"W_T_rolling" : W_T_HIST_test_df})
    dict_dfs_ALL.update({"W_paths__NNstrategy_rolling": W_paths_HIST_test_NN_df})

    # --------------------------------------------------------------------
    # ROLLING HISTORICAL analysis: Top and bottom W_T values


    # Get terminal wealth for objective function evaluation
    # - note, we don't want the "after cash withdrawal" value
    # - therefore, use paths rather than W_T_df
    W_T_RollingHistorical = W_paths_HIST_test_NN_df.to_numpy()
    W_T_RollingHistorical = W_T_RollingHistorical[:,-1].copy()    #get only last column


    dict_idx_for_looping = {}   #will be of format {"dict_key_prefix": np.array of indices}

    #Top and bottom values
    if output_only_for_fixed is False: #Do Top and Bottom paths only if we don't want JUST the fixed_yyyymm_list
        idx_W_T_RollingHistorical_bottom = fun_utilities.np_array_indices_SMALLEST_k_values_no_sorting(
                                            array=W_T_RollingHistorical, k=top_bottom_rows_count)
        idx_W_T_RollingHistorical_top = fun_utilities.np_array_indices_LARGEST_k_values_no_sorting(
                                            array=W_T_RollingHistorical, k=top_bottom_rows_count)

        dict_idx_for_looping.update({"W_T_bottom_idx_" : idx_W_T_RollingHistorical_bottom})
        dict_idx_for_looping.update({"W_T_top_idx_": idx_W_T_RollingHistorical_top})

    #Also find indices of particular interest
    if fixed_yyyymm_list is not None:
        lst = []
        for yyyymm in fixed_yyyymm_list:
            #get location (index) for each yyyymm in fixed_yyyymm_list
            lst.append(W_paths_HIST_test_NN_df.index.get_loc(yyyymm))
        lst = np.array(lst)

        dict_idx_for_looping.update({"Selected_idx_": lst})




    for key in dict_idx_for_looping.keys():
        dict_key_prefix = key
        idx_lst = dict_idx_for_looping[key]

        for idx in idx_lst:

            dict_key = dict_key_prefix +  str(idx)

            # Get the yyyymm for the index
            yyyymm_path_start = W_T_HIST_test_df.iloc[idx].name

            # Get historical path for idx
            df_historical_path_ret = fun_Data_historical_path.get_historical_path_returns(params_TRAIN, yyyymm_path_start=yyyymm_path_start)


            # Append "NaN" row to returns /features to ensure terminal wealth values can be added
            df_historical_path_ret = df_historical_path_ret.append(pd.Series(dtype="float64", name="T"), ignore_index=False)

            # Get benchmark mean and stdev used to calculate FEATURES
            benchmark_W_mean_train = (np.transpose(params_TRAIN["benchmark_W_mean_train"])).flatten()
            benchmark_W_std_train = (np.transpose(params_TRAIN["benchmark_W_std_train"])).flatten()

            df_Benchmark_MeanStd= pd.DataFrame()
            df_Benchmark_MeanStd["benchmark_W_mean_train"] = benchmark_W_mean_train
            df_Benchmark_MeanStd["benchmark_W_std_train"] = benchmark_W_std_train
            df_Benchmark_MeanStd.set_index(df_historical_path_ret.index.values, inplace=True) #Use same index as for df_historical_path_ret

            #Append benchmark mean and stdev for feature calc
            df_historical_path_ret = df_historical_path_ret.merge(df_Benchmark_MeanStd, how="inner", left_index = True, right_index = True)

            # Get wealth path using the NN strategy for this index
            df_W_path_idx = pd.DataFrame()
            df_W_path_idx["W_NN_strategy"] = W_paths_HIST_test_NN_df.iloc[idx]

            # Append wealth path using the benchmark strategy for this index
            df_W_path_idx["W_benchmark_strategy"] = W_paths_HIST_test_BENCHMARK_df.iloc[idx]

            #Get index
            df_W_path_idx.set_index(df_historical_path_ret.index.values, inplace=True)


            # Append benchmark mean and stdev
            df_historical_path_ret = df_historical_path_ret.merge(df_W_path_idx, how="inner", left_index = True, right_index = True)


            #Proportions in each asset along this historical path
            # - Also append "NaN" row at end for merging, and then merge with results
            df_NN_asset_prop_paths = NN_asset_prop_paths_dict[yyyymm_path_start]
            df_NN_asset_prop_paths = df_NN_asset_prop_paths.append(pd.Series(dtype="float64", name="T"),
                                                                   ignore_index=False)
            df_NN_asset_prop_paths.index = df_historical_path_ret.index.copy() #Set index for merging

            df_historical_path_ret = df_historical_path_ret.merge(df_NN_asset_prop_paths, how="inner", left_index=True,
                                                                  right_index=True)


            #Features along this historical path
            df_Feature_phi_paths = Feature_phi_paths_dict[yyyymm_path_start]
            df_Feature_phi_paths = df_Feature_phi_paths.append(pd.Series(dtype="float64", name="T"),
                                                                   ignore_index=False)
            df_Feature_phi_paths.index = df_historical_path_ret.index.copy() #Set index for merging

            df_historical_path_ret = df_historical_path_ret.merge(df_Feature_phi_paths, how="inner", left_index=True,
                                                                  right_index=True)


            #PRP scores along this historical path, if applicable
            if params_HIST_test_NN["PRP_TrueFalse"] is True:
                df_PRP_scores_paths = PRP_scores_paths_dict[yyyymm_path_start]
                df_PRP_scores_paths = df_PRP_scores_paths.append(pd.Series(dtype="float64", name="T"),
                                                                       ignore_index=False)
                df_PRP_scores_paths.index = df_historical_path_ret.index.copy() #Set index for merging

                df_historical_path_ret = df_historical_path_ret.merge(df_PRP_scores_paths, how="inner",
                                                                      left_index=True,
                                                                      right_index=True)


            #Append for output
            dict_dfs_ALL.update({dict_key : df_historical_path_ret})



    # --------------------------------------------------------------------
    #Get NN parameters (weights matrices) from training
    NN = params_TRAIN["NN_object"]

    for layer_id in np.arange(1 ,NN.n_layers_total ,1):  # No weights matrix *into* layer_id=0
        desc = NN.layers[layer_id].description
        x_l = NN.layers[layer_id].x_l  # weights matrix INTO layer_id
        x_l_df = pd.DataFrame(x_l)

        #Write into dictionary
        dict_dfs_ALL.update({"NNweights_into_layer_id_" + str(layer_id) : x_l_df})


    if output_Excel:
        # Write out
        timestamp = datetime.datetime.now().strftime('%Y-%m-%d_%H_%M')
        filename = filename_prefix_for_Excel + "timestamp_" + timestamp + "_HistoricalRolling_from_yyyymm" + ".xlsx"

        with pd.ExcelWriter(filename) as writer:    #we want to write out to multiple sheets
            for key in dict_dfs_ALL.keys():
                    sheet_name = key
                    sheet_data = dict_dfs_ALL[key]
                    sheet_data.to_excel(writer, sheet_name=sheet_name, header=True, index=True)


    return dict_dfs_ALL, W_T_HIST_test_df, W_paths_HIST_test_NN_df
