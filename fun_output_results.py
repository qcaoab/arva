#OBJECTIVE: output full results from NN portfolio optimization problem to pd.Series
# and write out to Excel if required


import pandas as pd
import numpy as np
import datetime
import fun_utilities
import matplotlib.pyplot as plt
import os

def output_W_T_vectors(params_TRAIN,         # dictionary with parameters and results from NN TRAINING
                      params_TEST = None,    # dictionary with parameters and results from NN TESTING
                      params_BENCHMARK_train = None,     #dictionary with benchmark strategy results and info on the TRAINING dataset
                      params_BENCHMARK_test = None,     #dictionary with benchmark strategy results and info on the TESTING dataset
                      output_Excel = False,             # write the result to Excel
                      filename_prefix_for_Excel = "z_"  #used if output_Excel== True
                      ):
    #Objective: Create single Excel spreadsheet with max 4 columns: ["W_T_train", "W_T_test", "W_T_benchmark_train", "W_T_benchmark_test"]
    # - W_T displayed might be  *after* cash withdrawal, if one sided quadratic check params["obj_fun_cashwithdrawal_TrueFalse"]

    #CHECK if TESTING data is supplied when necessary
    if params_TRAIN["test_TrueFalse"] is True:
        if params_TEST is None:
            raise ValueError("PVSerror in 'output_results_NN': if params_TRAIN['test_TrueFalse'] == True, then"
                             "we need params_TEST as input to this function.")

    #---------------------------------------------------------------------------------
    #TRAINING results
    #convert first to pd.Series in order to handle different length vectors W_T
    W_T_train = pd.Series(params_TRAIN["W_T"])
    df_output = pd.DataFrame(data = W_T_train, columns=["W_T_train"], index=None)

    #---------------------------------------------------------------------------------
    #  TESTING results
    if params_TEST is not None:  #only if we have testing results
        W_T_test = pd.Series(params_TEST["W_T"])
        df_output["W_T_test"] = W_T_test


    #---------------------------------------------------------------------------------
    #  BENCHMARK results
    if params_BENCHMARK_train is not None:
        W_T_benchmark_train = pd.Series(params_BENCHMARK_train["W_T"])
        df_output["W_T_benchmark_train"] = W_T_benchmark_train

    if params_BENCHMARK_test is not None:
        W_T_benchmark_test = pd.Series(params_BENCHMARK_test["W_T"])
        df_output["W_T_benchmark_test"] = W_T_benchmark_test


    # ------------------------------------------------------------------------------------------------
    # OUTPUT to Excel if required
    if output_Excel:
        timestamp = datetime.datetime.now().strftime('%Y-%m-%d_%H_%M')
        filename = filename_prefix_for_Excel + "timestamp_" + timestamp + "_W_T_vectors"
        df_output.to_excel(filename + ".xlsx", header= True, index=False)

    return df_output

def output_W_T_histogram_and_cdf(df_W_T_vectors,  #df_output from the function output_W_T_vectors
                          bins = None,  # "bins =" for np.histogram, can also be a list of bin edges
                          percentage_or_count = "percentage",  #set to "count" for number of obs in each bin,
                          # "percentage" for % of total nr of obs in each bin; #NOT used for CDF!
                          output_Excel=False,  # write the result to Excel
                          filename_prefix_for_Excel="z_"  # used if output_Excel== True
                          ):

    #OBJECTIVE: Calculates histogram and CDF for the W_T vectors based on the function output_W_T_vectors

    #If not provided, we need to explicitly fix bin edges, because all the W_T_vectors will use same bin edges!
    if bins is None:
        bins = np.linspace(start=0.0, stop=2000., num=101)
    elif np.array(bins).size <= 1: #if bins is just a value (integer)
        bins = np.linspace(start=0.0, stop=2000., num=int(bins))

    #Identify the right edges of the bins
    bins_right_edges = bins[1:]
    df_hist = pd.DataFrame(data=bins_right_edges, columns=["bins_right_edges"])

    df_cdf = pd.DataFrame(data=bins_right_edges, columns=["bins_right_edges"])

    #-----------------------------------------------------------------------------------
    #HISTOGRAMS:
    for col in df_W_T_vectors.columns:

        #Get denominator right, since training and testing datasets might have different lengths
        col_obs = np.sum(1 - np.isnan(df_W_T_vectors[col])) #Exclude "nan" values from number of observations

        #Histogram counts
        counts_hist, _ = np.histogram(df_W_T_vectors[col], bins= bins)

        #Percentages for cdf and/or histogram
        values_hist_perc = counts_hist / col_obs    #make sure *all* data is used for the count in the denominator

        #Calculate counts or % of total
        if percentage_or_count == "percentage":
            values_hist = values_hist_perc
        elif percentage_or_count == "count":
            values_hist = counts_hist

        df_hist[col] = values_hist
        df_cdf[col] =  df_hist[col].cumsum()


    #Save histograms
    if output_Excel:
        timestamp = datetime.datetime.now().strftime('%Y-%m-%d_%H_%M')
        
        filename = filename_prefix_for_Excel + "timestamp_" + timestamp + "_W_T_histogram_and_cdf" \
                       + ".xlsx"
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        
        with pd.ExcelWriter(filename) as writer:
            df_hist.to_excel(writer, sheet_name="hist_" + percentage_or_count, header=True, index=True)
            df_cdf.to_excel(writer, sheet_name="CDF", header=True, index=True)

    return df_hist, df_cdf




def output_results_NN(params_TRAIN,         # dictionary with parameters and results from NN TRAINING
                      params_TEST = None,   # dictionary with parameters and results from NN TESTING
                      params_BENCHMARK_train = None,     #dictionary with benchmark strategy results and info on the TRAINING dataset
                      params_BENCHMARK_test = None,     #dictionary with benchmark strategy results and info on the TRAINING dataset
                      output_Excel = False,             # write the result to Excel
                      filename_prefix_for_Excel = "z_"  #used if output_Excel== True
                      ):


    # ------------------------------------------------------------------------------------------------
    #  Portfolio problem: get Main structural parameters
    output_main_setup = {}

    output_main_setup.update({"Main_setup": "**********************************"})
    output_main_setup.update({"T"  :  params_TRAIN["T"]  })
    output_main_setup.update({"N_rb": params_TRAIN["N_rb"] })
    output_main_setup.update({"delta_t": params_TRAIN["delta_t"]})
    output_main_setup.update({"W0": params_TRAIN["W0"]})
    output_main_setup.update({"average_q": np.mean(params_TRAIN["q"])})

    output_main_setup.update({"": ""})  # Clear line

    # ------------------------------------------------------------------------------------------------
    # Objective function settings and parameters
    output_obj_fun = {}
    output_obj_fun.update({"OBJECTIVE_training": "**********************************"})

    temp_keys =  [key for key in params_TRAIN.keys() if "obj_fun" in key]
    for key in temp_keys:
        output_obj_fun.update({key : params_TRAIN[key]})

    output_obj_fun.update({"": ""})  # Clear line

    # ------------------------------------------------------------------------------------------------
    #  Transaction costs:
    #-- for convenience, this is appended to the output_obj_fun dictionary
    if params_TRAIN["TransCosts_TrueFalse"] is True:
        output_obj_fun.update({"Transaction_costs": "**********************************"})
        output_obj_fun.update({"TransCosts_TrueFalse" : params_TRAIN["TransCosts_TrueFalse"]})
        output_obj_fun.update({"TransCosts_r_b": params_TRAIN["TransCosts_r_b"]})
        output_obj_fun.update({"TransCosts_propcost": params_TRAIN["TransCosts_propcost"]})
        output_obj_fun.update({"TransCosts_lambda": params_TRAIN["TransCosts_lambda"]})
        output_obj_fun.update({" ": ""})  # Clear line

    # ------------------------------------------------------------------------------------------------
    #  Confidence penalty:
    #-- for convenience, this is appended to the output_obj_fun dictionary
    if params_TRAIN["ConfPenalty_TrueFalse"] is True:
        output_obj_fun.update({"Confidence_penalty": "**********************************"})
        output_obj_fun.update({"ConfPenalty_TrueFalse" : params_TRAIN["ConfPenalty_TrueFalse"]})
        output_obj_fun.update({"ConfPenalty_lambda": params_TRAIN["ConfPenalty_lambda"]})
        output_obj_fun.update({"ConfPenalty_n_H": params_TRAIN["ConfPenalty_n_H"]})
        output_obj_fun.update({" ": ""})  # Clear line

    # ------------------------------------------------------------------------------------------------
    #  Underlying assets:
    output_asset_basket = {}

    output_asset_basket.update({"Underlying_ASSETS": "**********************************"})
    output_asset_basket.update({"N_a": params_TRAIN["N_a"]})
    output_asset_basket.update({"real_or_nominal":params_TRAIN["real_or_nominal"]})

    #We don't want the basket DATA here, just the info
    temp_keys = [key for key in params_TRAIN["asset_basket"].keys() if "basket_" in key]
    for key in temp_keys:
        output_asset_basket.update({key: params_TRAIN["asset_basket"][key]})

    output_asset_basket.update({"_____": ""})  # Clear line

    #Asset basket data read settings
    if "asset_basket_data_settings" in params_TRAIN.keys(): #Might not be in there, e.g. if we did MC sim
        output_asset_basket.update({"Asset_basket_MARKET_DATA": "**********************************"})
        for key in params_TRAIN["asset_basket_data_settings"]:
            output_asset_basket.update({key: params_TRAIN["asset_basket_data_settings"][key]})

        output_asset_basket.update({"": ""})  # Clear line


    # ------------------------------------------------------------------------------------------------
    #  Features and Trading signals:
    output_features = {}

    output_features.update({"FEATURES": "**********************************"})

    output_features.update({"N_phi": params_TRAIN["N_phi"]})
    output_features.update({"use_trading_signals_TrueFalse": params_TRAIN["use_trading_signals_TrueFalse"]})

    if  params_TRAIN["use_trading_signals_TrueFalse"] == True:
        # We don't want the basket DATA here, just the info
        temp_keys = [key for key in params_TRAIN["trading_signal_basket"].keys() if "basket_" in key]
        for key in temp_keys:
            output_features.update({key: params_TRAIN["trading_signal_basket"][key]})

        output_features.update({"_____": ""})  # Clear line
        #Trading signal basket data read settings
        if "trading_signal_basket_data_settings" in params_TRAIN.keys():
            output_features.update({"Trading_signal_basket_MARKET_DATA": "**********************************"})
            for key in params_TRAIN["trading_signal_basket_data_settings"]:
                output_features.update({key: params_TRAIN["trading_signal_basket_data_settings"][key]})


    output_features.update({"": ""})  # Clear line

    # ------------------------------------------------------------------------------------------------
    #  BOOTSTRAP and/or MC simulation SETTINGS
    output_bootstrap_settings = {}

    output_bootstrap_settings.update({"Settings_TRAINING_DATA": "**********************************"})
    output_bootstrap_settings.update({"output_csv_data_training_testing":  params_TRAIN["output_csv_data_training_testing"]})
    output_bootstrap_settings.update({"data_source_Train": params_TRAIN["data_source_Train"]})

    if "bootstrap_settings_train" in params_TRAIN.keys():   #Bootstrap data
        for key in params_TRAIN["bootstrap_settings_train"].keys():
            suffix = "_train"
            output_bootstrap_settings.update({key + suffix: params_TRAIN["bootstrap_settings_train"][key]})

    elif "MCsim_info_train" in  params_TRAIN.keys():  #Simulated data
        for key in params_TRAIN["MCsim_info_train"].keys():
            suffix = "_train"
            output_bootstrap_settings.update({key + suffix: params_TRAIN["MCsim_info_train"][key]})

    if params_TRAIN["test_TrueFalse"] == True:
        suffix = "_test"
        output_bootstrap_settings.update({"_____": ""})  # Clear line
        output_bootstrap_settings.update({"Settings_TESTING_DATA": "**********************************"})
        output_bootstrap_settings.update({"data_source_Test": params_TRAIN["data_source_Test"]})

        if "bootstrap_settings_test" in params_TRAIN.keys():  # Bootstrap data
            for key in params_TRAIN["bootstrap_settings_test"].keys():
                output_bootstrap_settings.update({key + suffix: params_TRAIN["bootstrap_settings_test"][key]})

        elif "MCsim_info_test" in  params_TRAIN.keys():  #Simulated data
            for key in params_TRAIN["MCsim_info_test"].keys():
                output_bootstrap_settings.update({key + suffix: params_TRAIN["MCsim_info_test"][key]})

    output_bootstrap_settings.update({"": ""})  # Clear line


    # ------------------------------------------------------------------------------------------------
    #  Get Neural Network setup
    output_NN_setup = {}

    output_NN_setup.update({"NEURAL_NETWORK_setup": "**********************************"})
    output_NN_setup.update({"lambda_reg": params_TRAIN["lambda_reg"]})
    output_NN_setup.update({"n_nodes_input": params_TRAIN["N_phi"]})
    output_NN_setup.update({"n_layers_hidden": params_TRAIN["N_L"]})
    output_NN_setup.update({"n_nodes_output": params_TRAIN["N_a"]})

    NN_object = params_TRAIN["NN_object"]

    output_NN_setup.update({"NN_object.theta_length": NN_object.theta_length})

    for layer_id in np.arange(0,params_TRAIN["N_L"]+2,1):
        #get info
        description = NN_object.layers[layer_id].description
        n_nodes = NN_object.layers[layer_id].n_nodes
        activation = NN_object.layers[layer_id].activation

        #write out
        output_NN_setup.update({"layer_id_" + str(layer_id) + "_description": description})
        output_NN_setup.update({"layer_id_" + str(layer_id) + "_n_nodes": n_nodes})
        output_NN_setup.update({"layer_id_" + str(layer_id) + "_activation": activation})

    output_NN_setup.update({"": ""})  # Clear line

    # ------------------------------------------------------------------------------------------------
    #  Main NN training options
    output_NN_training_options = {}

    output_NN_training_options.update({"NN_training_options": "**********************************"})
    output_NN_training_options.update({"methods" : params_TRAIN["NN_training_options"]["methods"]})

    if params_TRAIN["preTrained_TrueFalse"] is False:   #If we needed to TRAIN the NN
        output_NN_training_options.update({"itbound_SGD_algorithms": params_TRAIN["NN_training_options"]["itbound_SGD_algorithms"]})
        output_NN_training_options.update({"batchsize": params_TRAIN["NN_training_options"]["batchsize"]})
        output_NN_training_options.update({"nit_running_min": params_TRAIN["NN_training_options"]["nit_running_min"]})
        output_NN_training_options.update({"nit_IterateAveragingStart": params_TRAIN["NN_training_options"]["nit_IterateAveragingStart"]})

    output_NN_training_options.update({"": ""})  # Clear line

    # ------------------------------------------------------------------------------------------------
    #  Training results
    output_results_TRAIN = ({"TRAINING_RESULTS": "**********************************"})

    #   Convert to comma-separated string to write out and read in if needed
    theta0 =  fun_utilities.np_1darray_TO_string_comma_separated(x = params_TRAIN["theta0"])

    output_results_TRAIN.update({"theta0": theta0})

    temp_keys = params_TRAIN["res_BEST"].keys()
    temp_keys = [key for key in temp_keys if "W_T_" not in key]
    temp_keys = [key for key in temp_keys if "summary_df" not in key]

    for key in temp_keys:

        if ("F_theta" in key) or ("NN_theta" in key):
            #   Convert to comma-separated string to write out and read in if needed
            key_data = fun_utilities.np_1darray_TO_string_comma_separated(x=params_TRAIN["res_BEST"][key])
            output_results_TRAIN.update({key: key_data })

        else:
            output_results_TRAIN.update({key: params_TRAIN["res_BEST"][key]})

    suffix = "_train"  # to avoid duplicates in final results
    for key in params_TRAIN["W_T_stats_dict"].keys():  # Get terminal wealth stats
        output_results_TRAIN.update(
            {key + suffix: params_TRAIN["W_T_stats_dict"][key]})


    #Add checks to training results if MEAN-CVAR objective
    if "mean_cvar" in params_TRAIN["obj_fun"]: #both single level and bilevel formulation

        xi = params_TRAIN["xi"] #will be optimal value
        alpha = params_TRAIN["obj_fun_alpha"]
        W_T = params_TRAIN["W_T"]   #terminal wealth vector

        # Calculate theoretical values for mean-cvar
        check_alpha_VAR = xi**2
        check_alpha_CVAR = np.mean(xi**2 + (1/alpha) *  \
                              np.minimum(W_T - xi**2, 0) )

        output_results_TRAIN.update({"check_alpha_VAR" + suffix : check_alpha_VAR})
        output_results_TRAIN.update({"check_alpha_CVAR" + suffix: check_alpha_CVAR})

    #Append AVERAGE proportion percentiles over time invested in each asset on the TRAINING dataset
    df_prop = params_TRAIN["df_pctiles_ALL"]
    df_index = df_prop.index.values.tolist()    #row headers
    prefix = "Avg_"
    for idx in df_index:
        if "Wealth" not in idx:     #Don't want to output wealth percentiles again
            output_results_TRAIN.update({prefix + idx: df_prop.loc[idx].mean(skipna=True)})

    #Append Feature stats (mean,stdev) over all paths and over all time periods for each feature
    temp_dict = params_TRAIN["Feature_phi_stats_dict"]
    output_results_TRAIN.update({"---* NN feature path stats *--- in training data": "--------"})
    for key in temp_dict.keys():
        output_results_TRAIN.update({key : temp_dict[key]})

    #Append values for feature standardization
    if params_TRAIN["use_trading_signals_TrueFalse"] is True:
        output_results_TRAIN.update({"---* Values for feature stdization *--- training data": "--------"})

        for trad_sig_index in np.arange(0, len(params_TRAIN["TradSig_order_train"]), 1):
            key = "MEAN " + params_TRAIN["TradSig_order_train"][trad_sig_index]
            val = params_TRAIN["TradSig_MEAN_train"][trad_sig_index]
            output_results_TRAIN.update({key : val})

            key = "STDEV " + params_TRAIN["TradSig_order_train"][trad_sig_index]
            val = params_TRAIN["TradSig_STDEV_train"][trad_sig_index]
            output_results_TRAIN.update({key: val})



    #Append AVERAGE PRP score percentiles over time for each feature, in TRAINING dataset
    if "df_PRP_pctiles" in params_TRAIN.keys():
        df_prop = params_TRAIN["df_PRP_pctiles"]
        df_index = df_prop.index.values.tolist()  # row headers
        prefix = "Avg_"
        for idx in df_index:
            output_results_TRAIN.update({prefix + idx: df_prop.loc[idx].mean(skipna=True)})


    #Append average total transaction costs
    if params_TRAIN["TransCosts_TrueFalse"] is True:
        output_results_TRAIN.update({"TransCosts_total_mean" :  params_TRAIN["TransCosts_cum_mean"]})
        output_results_TRAIN.update({"TransCosts_total_with_interest_mean": params_TRAIN["TransCosts_cum_with_interest_mean"]})

    output_results_TRAIN.update({"": ""})  # Clear line

    # ------------------------------------------------------------------------------------------------
    #  TESTING results
    if params_TEST is not None:  #only if we have testing results

        output_results_TEST = {}
        output_results_TEST.update({"TESTING_RESULTS": "**********************************"})


        suffix = "_test"  # to avoid duplicates in final results
        for key in params_TEST["W_T_stats_dict"].keys():  # Get terminal wealth stats
            output_results_TEST.update(
                {key + suffix: params_TEST["W_T_stats_dict"][key]})


        # Append AVERAGE proportion percentiles over time invested in each asset on the TESTING dataset
        df_prop = params_TEST["df_pctiles_ALL"]
        df_index = df_prop.index.values.tolist()  # row headers
        suffix = "_test"  # to avoid duplicates in final results
        prefix = "Avg_"
        for idx in df_index:
            if "Wealth" not in idx:  # Don't want to output wealth percentiles again
                output_results_TEST.update({prefix + idx + suffix: df_prop.loc[idx].mean(skipna=True)})

        # Append Feature stats (mean,stdev) over all paths and over all time periods for each feature
        temp_dict = params_TEST["Feature_phi_stats_dict"]
        output_results_TEST.update({"---* NN feature path stats *--- in testing data": "--------"})
        for key in temp_dict.keys():
            output_results_TEST.update({key: temp_dict[key]})

        # Append AVERAGE PRP score percentiles over time for each feature, in TESTING dataset
        if "df_PRP_pctiles" in params_TEST.keys():
            df_prop = params_TEST["df_PRP_pctiles"]
            df_index = df_prop.index.values.tolist()  # row headers
            prefix = "Avg_"
            suffix = "_test"  # to avoid duplicates in final results
            for idx in df_index:
                output_results_TEST.update({prefix + idx + suffix: df_prop.loc[idx].mean(skipna=True)})


        #Append transaction costs if applicable
        if params_TEST["TransCosts_TrueFalse"] is True:
            output_results_TEST.update({"TransCosts_total_mean": params_TEST["TransCosts_cum_mean"]})
            output_results_TEST.update({"TransCosts_total_with_interest_mean": params_TEST["TransCosts_cum_with_interest_mean"]})

        #Output generalization error info
        output_results_TEST.update({"_____": ""})  # Clear line
        output_results_TEST.update({"TRAINING_vs_TESTING": "**********************************"})
        for key in params_TEST["gen_error_dict"].keys():  # get generalization error gen_error_dict
            output_results_TEST.update(
                {key: params_TEST["gen_error_dict"][key]})


        output_results_TEST.update({"": ""})  # Clear line

    # ------------------------------------------------------------------------------------------------
    #  BENCHMARK TRAINING results if supplied
    if params_BENCHMARK_train is not None:  #If benchmarks TRAINING results have been provided
        suffix = "_benchmark_train"   #to avoid duplicates in final results

        output_results_Benchmark = {}
        output_results_Benchmark.update({"BENCHMARK_strategy_TRAINING_data": "**********************************"})

        output_results_Benchmark.update({"strategy_description" + suffix: params_BENCHMARK_train["strategy_description"]})

        if params_BENCHMARK_train["strategy_description"] == "invest_ConstProp_strategy":   #if CP, output proportions
            output_results_Benchmark.update({"prop_const": params_BENCHMARK_train["prop_const"].tolist()})


        for key in params_BENCHMARK_train["W_T_stats_dict"].keys(): #Get terminal wealth stats
            output_results_Benchmark.update(
                {key + suffix: params_BENCHMARK_train["W_T_stats_dict"][key]})


        #Append transaction costs if applicable
        if params_BENCHMARK_train["TransCosts_TrueFalse"] is True:
            output_results_Benchmark.update({"TransCosts_total_mean": params_BENCHMARK_train["TransCosts_cum_mean"]})
            output_results_Benchmark.update({"TransCosts_total_with_interest_mean": params_BENCHMARK_train["TransCosts_cum_with_interest_mean"]})


        output_results_Benchmark.update({"":""})    #Clear line


    # ------------------------------------------------------------------------------------------------
    #  BENCHMARK TESTING results if supplied
    if params_BENCHMARK_test is not None:  #If benchmarks TESTING results have been provided
        suffix = "_benchmark_test"   #to avoid duplicates in final results

        output_results_Benchmark_test = {}
        output_results_Benchmark_test.update({"BENCHMARK_strategy_TESTING_data": "**********************************"})

        output_results_Benchmark_test.update({"strategy_description" + suffix: params_BENCHMARK_test["strategy_description"]})

        if params_BENCHMARK_test["strategy_description"] == "invest_ConstProp_strategy":   #if CP, output proportions
            output_results_Benchmark_test.update({"prop_const": params_BENCHMARK_test["prop_const"].tolist()})


        for key in params_BENCHMARK_test["W_T_stats_dict"].keys(): #Get terminal wealth stats
            output_results_Benchmark_test.update(
                {key + suffix: params_BENCHMARK_test["W_T_stats_dict"][key]})


        #Append transaction costs if applicable
        if params_BENCHMARK_test["TransCosts_TrueFalse"] is True:
            output_results_Benchmark_test.update({"TransCosts_total_mean": params_BENCHMARK_test["TransCosts_cum_mean"]})
            output_results_Benchmark_test.update({"TransCosts_total_with_interest_mean": params_BENCHMARK_test["TransCosts_cum_with_interest_mean"]})

        output_results_Benchmark_test.update({"":""})    #Clear line


    # ------------------------------------------------------------------------------------------------
    # COMBINE output dictionaries and merge

    # Convert dictionaries to pd.Series in order in which we want
    pd_Series_1 = pd.Series(data=output_main_setup)
    pd_Series_2 = pd.Series(data=output_obj_fun)
    pd_Series_3 = pd.Series(data=output_asset_basket)
    pd_Series_4 = pd.Series(data=output_features)
    pd_Series_5 = pd.Series(data = output_bootstrap_settings)
    pd_Series_6 = pd.Series(data=output_NN_setup)
    pd_Series_7 = pd.Series(data=output_NN_training_options)
    pd_Series_8 = pd.Series(data=output_results_TRAIN)

    output_full_results = pd.concat([pd_Series_1, pd_Series_2, pd_Series_3, pd_Series_4,
                                     pd_Series_5, pd_Series_6, pd_Series_7, pd_Series_8 ])

    if params_TEST is not None:
        pd_Series_X =  pd.Series(data=output_results_TEST)
        output_full_results = pd.concat([output_full_results, pd_Series_X ])


    if params_BENCHMARK_train is not None:
        pd_Series_Y = pd.Series(data=output_results_Benchmark)
        output_full_results = pd.concat([output_full_results, pd_Series_Y ])

    if params_BENCHMARK_test is not None:
        pd_Series_Z = pd.Series(data=output_results_Benchmark_test)
        output_full_results = pd.concat([output_full_results, pd_Series_Z ])



    # ------------------------------------------------------------------------------------------------
    # OUTPUT to Excel if required
    if output_Excel:
        timestamp = datetime.datetime.now().strftime('%Y-%m-%d_%H_%M')
        filename = filename_prefix_for_Excel + "timestamp_" + timestamp + "__output"
        output_full_results.to_excel(filename + ".xlsx")

    return output_full_results