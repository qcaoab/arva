#Wraps the training, testing of NN, as well as the outputs and analysis


import numpy as np
import pandas as pd
import json
import fun_train_NN
import fun_output_results
import fun_output_results_TopBottom
import fun_output_results_Pctiles
import fun_Plot_NN_control_FunctionHeatmaps
import fun_Plot_NN_control_DataHeatmaps
import fun_invest_ConstProp_strategy
import fun_eval_objfun_NN_strategy
import fun_W_T_stats
import class_NN_Pytorch
import export_control_from_NN


def RUN__wrapper_TWO_stage_optimization(
        params,  # dictionary as setup in the main code
        NN,  # object of class_Neural_Network with structure as setup in main code
        theta0,  # initial parameter vector (weights and biases) + other parameters for objective function
        NN_training_options,  # dictionary with options to train NN, specifying algorithms and hyperparameters
        output_parameters  # Dictionary with output parameters as setup in main code
):
    #Objective: Do TWO stage optimization for the following objectives:
    #"meancvarLIKE_constant_wstar"

    #--Loop over list of candidate wstar values and output the lowest one

    # Initialize running best wstar
    params_TRAIN_best = {}
    params_CP_TRAIN_best = {}
    params_TEST_best = {}
    params_CP_TEST_best = {}
    F_val_best = np.inf

    for wstar in params["obj_fun_LIST_constant_wstar_tested"]:
        params["obj_fun_constant_wstar"] = wstar  # CONSTANT wstar(s),
        # in usual mean-CVAR this would be candidate value for value-at-risk at level alpha

        print("wstar(nu): ", wstar)
        # For constant value of wstar: Do TRAINING and/or TESTING of NN, as well as OUTPUTS
        params_TRAIN, params_CP_TRAIN, params_TEST, params_CP_TEST = \
            RUN__wrapper_training_testing_NN(
                params=params,  # dictionary as setup in the main code
                NN=NN,  # object of class_Neural_Network with structure as setup in main code
                theta0=theta0,
                # initial parameter vector (weights and biases) + other parameters for objective function
                NN_training_options=NN_training_options
                # dictionary with options to train NN, specifying algorithms and hyperparameters
            )

        if params_TRAIN["F_val"] < F_val_best:  # Assign new running minimum
            F_val_best = params_TRAIN["F_val"]
            params_TRAIN_best = params_TRAIN.copy()
            params_CP_TRAIN_best = params_CP_TRAIN.copy()
            params_TEST_best = params_TEST.copy()
            params_CP_TEST_best = params_CP_TEST.copy()

    # END: Loop over obj_fun_LIST_constant_wstar_tested

    # Assign and output the dictionaries with value of wstar in constant_wstar_list giving SMALLEST objective function value
    params_TRAIN = params_TRAIN_best.copy()
    params_CP_TRAIN = params_CP_TRAIN_best.copy()
    params_TEST = params_TEST_best.copy()
    params_CP_TEST = params_CP_TEST_best.copy()

    #Delete values no longer necessary
    del params_TRAIN_best, params_CP_TRAIN_best, params_TEST_best, params_CP_TEST_best

    #Output only the best results
    RUN__wrapper_output(
        output_parameters=output_parameters,  # Dictionary with output parameters as setup in main code
        params_TRAIN=params_TRAIN,  # dictionary with parameters and results from NN TRAINING
        params_CP_TRAIN=params_CP_TRAIN,  # dictionary with benchmark strategy results and info on the TRAINING dataset
        params_TEST=params_TEST,  # dictionary with parameters and results from NN TESTING
        params_CP_TEST=params_CP_TEST  # dictionary with benchmark strategy results and info on the TESTING dataset
    )

    # Check if wstar value chosen occurs at an endpoint of LIST "obj_fun_LIST_constant_wstar_tested":
    # - if this is the case, warn the user.
    if params_TRAIN["obj_fun_constant_wstar"] in \
            [params["obj_fun_LIST_constant_wstar_tested"][0], params["obj_fun_LIST_constant_wstar_tested"][-1]]:
        print("WARNING for objective: meancvarLIKE_constant_wstar")
        print(">> wstar giving minimum F_val was at an ENDPOINT of the list of wstar values considered.")
        print(">> True wstar might be outside the range provided.")

    return params_TRAIN, params_CP_TRAIN, params_TEST, params_CP_TEST


def RUN__wrapper_ONE_stage_optimization(
        params,  # dictionary as setup in the main code
        NN_list,  # object of class_Neural_Network with structure as setup in main code 
        NN_training_options,  # dictionary with options to train NN, specifying algorithms and hyperparameters
        output_parameters      #Dictionary with output parameters as setup in main code
):
    #Objective: Do ONE stage optimization for the following objectives:
    #"mean_cvar_single_level",
    # "one_sided_quadratic_target_error",
    # "quad_target_error",
    # "huber_loss",
    # "ads"

    #Do TRAINING and/or TESTING of NN
    params_TRAIN, params_CP_TRAIN, params_TEST, params_CP_TEST = \
        RUN__wrapper_training_testing_NN(
            params=params,  # dictionary as setup in the main code
            NN_list=NN_list,  # object of class_Neural_Network with structure as setup in main code
            NN_training_options=NN_training_options
            # dictionary with options to train NN, specifying algorithms and hyperparameters
        )

    # #Do OUTPUTS
    params_TRAIN, params_CP_TRAIN, params_TEST, params_CP_TEST = \
        RUN__wrapper_output(
        output_parameters=output_parameters,  # Dictionary with output parameters as setup in main code
        params_TRAIN=params_TRAIN,  # dictionary with parameters and results from NN TRAINING
        params_CP_TRAIN=params_CP_TRAIN,
        # dictionary with benchmark strategy results and info on the TRAINING dataset
        params_TEST=params_TEST,  # dictionary with parameters and results from NN TESTING
        params_CP_TEST=params_CP_TEST  # dictionary with benchmark strategy results and info on the TESTING dataset
    )

    return params_TRAIN, params_CP_TRAIN, params_TEST, params_CP_TEST



def RUN__wrapper_training_testing_NN(
        params,    #dictionary as setup in the main code
        NN_list,  # object of class_Neural_Network with structure as setup in main code
        NN_training_options  #dictionary with options to train NN, specifying algorithms and hyperparameters
):
    #ALWAYS returns 4 dictionaries (whether testing or no testing):
    #return params_TRAIN, params_CP_TRAIN, params_TEST, params_CP_TEST
    # -> If no testing, then params_TEST = {} and params_CP_TEST = {}, but these are always returned

    # -----------------------------------------------------------------------------------------------
    # BENCHMARK: Constant proportion strategy on TRAINING data
    # -----------------------------------------------------------------------------------------------
    
    # pytorch to do: get outputs from pyt NN
    
    # Constant proportion strategy implemented on the TRAINING data
    params_CP_TRAIN = fun_invest_ConstProp_strategy.invest_ConstProp_strategy(prop_const=params["benchmark_prop_const"],
                                                                              params=params,
                                                                              train_test_Flag="train")

    # Print some key results from Benchmark strategy on TRAINING data
    print("-----------------------------------------------")
    print("Selected results: ConstProp_strategy on TRAINING dataset")
    print("constant withdrawal: ", params["withdraw_const"])
    print("constant allocation: ", params["benchmark_prop_const"])
    print("W_T_mean: " + str(params_CP_TRAIN["W_T_stats_dict"]["W_T_mean"]))
    print("W_T_median: " + str(params_CP_TRAIN["W_T_stats_dict"]["W_T_median"]))
    print("W_T_pctile_5: " + str(params_CP_TRAIN["W_T_stats_dict"]["W_T_pctile_5"]))
    print("W_T_CVAR_5_pct: " + str(params_CP_TRAIN["W_T_stats_dict"]["W_T_CVAR_5_pct"]))
    print("-----------------------------------------------")

    # Append constant benchmark results to the results summary file:
    
    with open(params["results_dir"]+"results_summary.json") as in_file:
        results_dict = json.load(in_file)
    
    results_dict["constant_benchmark_results"] = {"constant withdrawal": params["withdraw_const"],
                                                  "constant allocation": params["benchmark_prop_const"].tolist(),
                                                  "W_T_mean": params_CP_TRAIN["W_T_stats_dict"]["W_T_mean"],
                                                  "W_T_median": params_CP_TRAIN["W_T_stats_dict"]["W_T_median"],
                                                  "W_T_pctile_5": params_CP_TRAIN["W_T_stats_dict"]["W_T_pctile_5"],
                                                  "W_T_CVAR_5_pct": params_CP_TRAIN["W_T_stats_dict"]["W_T_CVAR_5_pct"]}
    
    with open(params["results_dir"]+"results_summary.json", "w") as out_file:
        json.dump(results_dict, out_file, indent = 6)
    out_file.close() 
  
    
    # Don't over write standardization values if sideloaded during initialization   
    if params["sideloaded_standardization"] == True:
        print("\n Side loaded standardization params from pre-trained model. \n")
        
    else:
        # Append to params results for "W_paths_mean" and "W_paths_std",
        # used for standardizing the feature vector for NN strategy
        params["benchmark_W_mean_train"] = params_CP_TRAIN["W_paths_mean"].copy()
        params["benchmark_W_std_train"] = params_CP_TRAIN["W_paths_std"].copy()

    #stats for post-withdrawal
    params["benchmark_W_mean_train_post_withdraw"] = params_CP_TRAIN["W_paths_mean_post_withdraw"].copy()
    params["benchmark_W_std_train_post_withdraw"] = params_CP_TRAIN["W_paths_std_post_withdraw"].copy()
    
    #Also add terminal wealth vector from constant proportion strategy (for ADS and IR objectives)
    if params["obj_fun"] in ["ads_stochastic", "qd_stochastic", "ir_stochastic", "te_stochastic"]:
        params["benchmark_W_T_vector_train"] =  params_CP_TRAIN["W_T"].copy()  #terminal wealth as a vector (one entry for each path)
        params["benchmark_W_paths_train"] = params_CP_TRAIN["W"].copy()

    
    #----------------------------------------------------------------------------------------
    # EXPORT CONTROL EXPLICITLY, if required
    #----------------------------------------------------------------------------------------
    
    if params["output_control"]:
        
        export_control_from_NN.export_controls(NN_list, params)
        
        print("Explicit control exported from preloaded model. Quitting program.")
        exit()
    # ----------------------------------------------------------------------------------------
    # TRAINING of NN
    # ----------------------------------------------------------------------------------------

    params_TRAIN = {}   #make sure it is empty
    res_BEST = {} #make sure it is empty
    
    params_TRAIN, res_adam = fun_train_NN.train_NN(
                                    NN_list = NN_list,
                                    params = params,
                                    NN_training_options = NN_training_options
                                    )


    # Print results of trained NN model:
    print("-----------------------------------------------")
    print("Selected results: NN-strategy-on-TRAINING dataset (temp implementation")
    print("W_T_mean: " + str(res_adam["temp_w_output_dict"]["W_T_mean"]))
    print("W_T_median: " + str(res_adam["temp_w_output_dict"]["W_T_median"]))
    print("W_T_pctile_5: " + str(res_adam["temp_w_output_dict"]["W_T_pctile_5"]))
    print("W_T_CVAR_5_pct: " + str(res_adam["temp_w_output_dict"]["W_T_CVAR_5_pct"]))
    if params["nn_withdraw"]:
        print("Average q (qsum/M+1): ", res_adam["q_avg"])
    print("Optimal xi: ", res_adam["optimal_xi"])
    print("Expected(across Rb) median(across samples) p_equity: ", res_adam["average_median_p"])
    print("obj fun: ", res_adam["objfun_final"])
    print("-----------------------------------------------")

        
    # Append NN results to the results summary file:
    in_file = open(params["results_dir"]+"results_summary.json")
    results_dict = json.load(in_file)
    in_file.close()
    
    results_dict["NN_results_kappa_" + str(params["obj_fun_rho"])] = {"kappa": params["obj_fun_rho"],
                                                  "W_T_mean": params_CP_TRAIN["W_T_stats_dict"]["W_T_mean"],
                                                  "W_T_median": params_CP_TRAIN["W_T_stats_dict"]["W_T_median"],
                                                  "W_T_pctile_5": params_CP_TRAIN["W_T_stats_dict"]["W_T_pctile_5"],
                                                  "W_T_CVAR_5_pct": params_CP_TRAIN["W_T_stats_dict"]["W_T_CVAR_5_pct"],
                                                  "optimal_xi": res_adam["optimal_xi"],
                                                  "obj_fun_value": res_adam["objfun_final"]}
    
    if params["nn_withdraw"]:
        results_dict["NN_results_kappa_" + str(params["obj_fun_rho"])]["Average q (qsum/M+1)"] = res_adam["q_avg"]
    
    out_file = open(params["results_dir"]+"results_summary.json", "w")
    json.dump(results_dict, out_file, indent = 6)
    out_file.close() 
  
    
    #----------------------------------------------------------------------------------------
    # TESTING of NN
    #----------------------------------------------------------------------------------------

    #Initialize testing outputs to avoid errors
    params_CP_TEST = {}
    params_TEST = {}

    #TESTING only when params["test_TrueFalse"] == True
    

    return params_TRAIN, params_CP_TRAIN, params_TEST, params_CP_TEST #, res_adam


def RUN__wrapper_output(
        output_parameters,      #Dictionary with output parameters as setup in main code
        params_TRAIN,           # dictionary with parameters and results from NN TRAINING
        params_CP_TRAIN,        #dictionary with benchmark strategy results and info on the TRAINING dataset
        params_TEST = None,     # dictionary with parameters and results from NN TESTING
        params_CP_TEST = None   #dictionary with benchmark strategy results and info on the TESTING dataset
):

    # -----------------------------------------------------------------------------------------------
    # Unpack output flags
    # -----------------------------------------------------------------------------------------------
    # Basic output params                                             #added kappa to output name
    code_title_prefix = output_parameters["code_title_prefix"]    # used as prefix for naming files when saving outputs
    output_results_Excel = output_parameters["output_results_Excel"]  # Output results summary to Excel

    save_Figures_format = output_parameters["save_Figures_format"] # format to save figures in, e.g. "png", "eps",

    # W_T, pdf and cdf
    output_W_T_vectors_Excel = output_parameters["output_W_T_vectors_Excel"]  # output the terminal wealth vectors (train, test and benchmark) for histogram construction
    output_W_T_histogram_and_cdf = output_parameters[ "output_W_T_histogram_and_cdf"]  # output Excel sheet with numerical histogram and CDF of terminal wealth (train, test and benchmark)
    output_W_T_histogram_and_cdf_W_max = output_parameters["output_W_T_histogram_and_cdf_W_max"] # Maximum W_T value for histogram and CDF.
    output_W_T_histogram_and_cdf_W_bin_width = output_parameters["output_W_T_histogram_and_cdf_W_bin_width"]  # Bin width of wealth for histogram and CDF.


    # NN detail
    output_TrainingData_NNweights_test = output_parameters["output_TrainingData_NNweights_test"]  # if true, outputs the training paths + features + NN weights required to
    # reproduce the top and bottom k results for the terminal wealth and objective function values

    # PERCENTILES:
    output_Pctiles_Excel = output_parameters["output_Pctiles_Excel"]  # If True, outputs Excel spreadsheet with NN pctiles of proportions in each asset and wealth over time
    output_Pctiles_Plots = output_parameters["output_Pctiles_Plots"]  # if True, plots the paths of output_Pctiles_Excel over time
    output_Pctiles_Plots_W_max = output_parameters["output_Pctiles_Plots_W_max"]  #Maximum y-axis value for WEALTH percentile plots
    output_Pctiles_list = output_parameters["output_Pctiles_list"]  # Only used if output_Pctiles_Excel or output_Pctiles_Plots is True, must be list, e.g.  [20,50,80]
    output_Pctiles_on_TEST_data = output_parameters["output_Pctiles_on_TEST_data"]  # Output percentiles for test data as well

    # Control heatmap params [used if save_Figures == True]
    save_Figures_FunctionHeatmaps = output_parameters["save_Figures_FunctionHeatmaps"] # save figures in save_Figures_format
    output_FunctionHeatmaps_Excel = output_parameters["output_FunctionHeatmaps_Excel"] # If TRUE, output the heatmap grid data to Excel

    save_Figures_DataHeatmaps = output_parameters["save_Figures_DataHeatmaps"]  # save figures in save_Figures_format
    output_DataHeatmaps_Excel = output_parameters["output_DataHeatmaps_Excel"]  # If TRUE, output the heatmap grid data to Excel

    heatmap_y_bin_min = output_parameters["heatmap_y_bin_min"]  # minimum for the y-axis grid
    heatmap_y_bin_max = output_parameters["heatmap_y_bin_max"] # maximum for the y-axis grid
    heatmap_y_num_pts = output_parameters["heatmap_y_num_pts"] # number of points for y-axis grid
    heatmap_xticklabels = output_parameters["heatmap_xticklabels"] # e.g. xticklabels=6 means we are displaying only every 6th xaxis label to avoid overlapping
    heatmap_yticklabels = output_parameters["heatmap_yticklabels"] # e.g. yticklabels = 10 means we are displaying every 10th label
    heatmap_cmap = output_parameters["heatmap_cmap"]  # e.g. "Reds" or "rainbow" etc colormap for sns.heatmap
    heatmap_cbar_limits = output_parameters["heatmap_cbar_limits"] # list in format [vmin, vmax] for heatmap colorbar/scale


    #----------------------------------------------------------------------------------------------------
    # OUTPUT full results to Excel (parameters + results)
    #----------------------------------------------------------------------------------------------------


    if output_results_Excel:

        #Add percentiles of asset proportions and wealth over time to params
        params_TRAIN = fun_output_results_Pctiles.get_df_Pctile_paths(params_TRAIN)

       
        #"TODO: Need to implement output
        # fun_output_results.output_results_NN(params_TRAIN = params_TRAIN,
        #                                      params_TEST=None,
        #                                      params_BENCHMARK_train=params_CP_TRAIN,
        #                                      params_BENCHMARK_test= None,
        #                                      output_Excel = output_results_Excel,
        #                                      filename_prefix_for_Excel=code_title_prefix
        #                                      )

    #----------------------------------------------------------------------------------------------------
    # OUTPUT terminal wealth vectors (W_T vectors) to .csv file AND/OR HISTOGRAM/CDF
    #   - Note: this might be is *after* cash withdrawal, if one sided quadratic check params["obj_fun_cashwithdrawal_TrueFalse"]
    #----------------------------------------------------------------------------------------------------


    if output_W_T_vectors_Excel or output_W_T_histogram_and_cdf:


        #Bins for histogram and CDF
        bins = np.linspace(start=0.0, stop= output_W_T_histogram_and_cdf_W_max ,
                           num= int(output_W_T_histogram_and_cdf_W_max/output_W_T_histogram_and_cdf_W_bin_width +1))

        
        df_W_T_vectors = fun_output_results.output_W_T_vectors(params_TRAIN=params_TRAIN,
                                                params_TEST=None,
                                                params_BENCHMARK_train=params_CP_TRAIN,
                                                params_BENCHMARK_test=None,
                                                output_Excel=output_W_T_vectors_Excel,
                                                filename_prefix_for_Excel=code_title_prefix
                                                )

        #Histogram and CDF: output to Excel spreadsheet
        fun_output_results.output_W_T_histogram_and_cdf(df_W_T_vectors = df_W_T_vectors,
                      bins = bins,  #bin edges or integer number of bins
                      percentage_or_count = "percentage", #not used for cdf
                      output_Excel=output_W_T_histogram_and_cdf,  # write the result to Excel
                      filename_prefix_for_Excel= code_title_prefix,
                      results_output = params_TRAIN["results_dir_kappa"]
                      )

    #----------------------------------------------------------------------------------------------------
    # NN control FUNCTION HEATMAPS (Optimal control as a function of features)
    #----------------------------------------------------------------------------------------------------
    # Will just plot this for the *last* of the batchsizes and itbounds above


    if save_Figures_FunctionHeatmaps or output_FunctionHeatmaps_Excel:

        
        fun_Plot_NN_control_FunctionHeatmaps.fun_Heatmap_NN_control_basic_features\
                                    (params = params_TRAIN,  #params dictionary with *trained* NN parameters and setup as in main code
                                    W_num_pts = heatmap_y_num_pts,  #number of points for wealth grid
                                    W_min = heatmap_y_bin_min,  #minimum for the wealth grid
                                    W_max = heatmap_y_bin_max,  #maximum for the wealth grid
                                    save_Figures = save_Figures_FunctionHeatmaps,  #Saves figures in format specified below
                                    save_Figures_format = save_Figures_format,
                                    fig_filename_prefix = params_TRAIN["results_dir_kappa"],
                                    feature_calc_option = None,  # Set calc_option = "matlab" to match matlab code, None to match my notes
                                    xticklabels = heatmap_xticklabels,  # e.g. xticklabels=6 means we are displaying only every 6th xaxis label to avoid overlapping
                                    yticklabels= heatmap_yticklabels,  #e.g. yticklabels = 500 means we are displaying every 500th label
                                    cmap= heatmap_cmap,  # e.g. "Reds" or "rainbow" etc colormap for sns.heatmap
                                    heatmap_cbar_limits= heatmap_cbar_limits,  # list in format [vmin, vmax] for heatmap colorbar/scale
                                    output_HeatmapData_Excel=output_FunctionHeatmaps_Excel  # If TRUE, output the heatmap grid data to Excel, naming uses fig_filename_prefix
                                    )


        
    # ----------------------------------------------------------------------------------------------------
    # NN control DATA HEATMAPS (Optimal control as realized on the training data)
    # ----------------------------------------------------------------------------------------------------
    if save_Figures_DataHeatmaps or output_DataHeatmaps_Excel:

        fun_Plot_NN_control_DataHeatmaps.plot_DataHeatmaps(
            params=params_TRAIN,  # params dictionary with *trained* NN results and setup as in main code
            y_bin_min = heatmap_y_bin_min,  # left endpoint of first y-axis bin
            y_bin_max=heatmap_y_bin_max,  # right endpoint of last W bin
            delta_y_bin = 5.0,  # bin width for W bins
            save_Figures=save_Figures_DataHeatmaps,  # Saves figures in format specified below
            save_Figures_format=save_Figures_format,
            fig_filename_prefix = params_TRAIN["results_dir_kappa"],
            xticklabels=heatmap_xticklabels,
            # e.g. xticklabels=6 means we are displaying only every 6th xaxis label to avoid overlapping
            yticklabels=heatmap_yticklabels,  # e.g. yticklabels = 500 means we are displaying every 500th label
            cmap= heatmap_cmap,  # e.g. "Reds" or "rainbow" etc colormap for sns.heatmap
            heatmap_cbar_limits=heatmap_cbar_limits,  # list in format [vmin, vmax] for heatmap colorbar/scale
            output_HeatmapData_Excel=output_DataHeatmaps_Excel # If TRUE, output the heatmap grid data to Excel, naming uses fig_filename_prefix
        )



    # ----------------------------------------------------------------------------------------------------
    # Output selected training PATHS and NN weights matrices for testing
    # ----------------------------------------------------------------------------------------------------

    # Output to Excel for testing:
    # - NN weights matrices
    # - benchmark wealth W mean and stdev for (standardized) feature calc
    # - Paths from the TRAINING dataset giving the Top X and bottom X *objective function* values:
    #   -- Trading signal paths (features), if applicable
    #   -- Asset returns

    if output_TrainingData_NNweights_test is True:

        top_bottom_rows_count = 2  # Number of top and bottom objective function and wealth values of interest

        fun_output_results_TopBottom.output_training_results_TopBottom_paths(params_TRAIN=params_TRAIN,
                                                                             top_bottom_rows_count=top_bottom_rows_count,
                                                                             output_Excel=output_TrainingData_NNweights_test,
                                                                             filename_prefix_for_Excel=code_title_prefix)




    #----------------------------------------------------------------------------------------------------
    #  NN control on data as percentiles (proportion in each asset, wealth) over time
    #----------------------------------------------------------------------------------------------------


    if output_Pctiles_Excel or output_Pctiles_Plots:

        fun_output_results_Pctiles.output_Pctile_paths(
                            params_TRAIN,  # dictionary with parameters and results from NN investment (TRAINING or TESTING)
                            pctiles=output_Pctiles_list,  # E.g. [20,50,80] List of percentiles to output and/or plot
                            output_Excel=output_Pctiles_Excel,  # write the result to Excel
                            filename_prefix_for_Excel=params_TRAIN["results_dir_kappa"],  # used if output_Excel is True
                            save_Figures=output_Pctiles_Plots,  # Plots and save figures in format specified below
                            save_Figures_format="png",
                            fig_filename_prefix=params_TRAIN["results_dir_kappa"],
                            W_max=output_Pctiles_Plots_W_max,  # Maximum wealth value for wealth percentiles graph
                            )


    #Output since some summary fields have been added
    return params_TRAIN, params_CP_TRAIN, params_TEST, params_CP_TEST

