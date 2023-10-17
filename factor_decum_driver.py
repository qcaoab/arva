
# Portfolio optimization problem using NN to model dual control for decumulation/multi factor problem 

run_on_my_computer = False   #True if running in my local desktop, False if running remotely
#-- this flag affects use of matplotlib.use('Agg') below


import pandas as pd
import numpy as np
import os
import datetime
import sys
import datetime
from pathlib import Path
import re
import json

#Import files needed (other files are imported within those files as needed)
import fun_Data_timeseries_basket
import fun_Data__bootstrap_wrapper
import fun_Data__MCsim_wrapper
import fun_train_NN_algorithm_defaults
import fun_RUN__wrapper
import fun_utilities
import class_NN_Pytorch
import torch
import manage_nn_models
import copy
import EF_plotter

if run_on_my_computer is True:  #handle importing of matplotlib

    #---- Check: current working directory (to ensure no conflicts with file names etc. ----
    # print("Current working directory is: " + str( os.getcwd()))
    # #If needed, make sure we are in the correct working directory
    # os.chdir("../NN_pension_DC_decumulation_v02")
    # print("After change - Current working directory is: " + str( os.getcwd()))

    # ---- Disable or enable Agg
    import matplotlib
    import matplotlib.pyplot as plt

else:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

#-----------------------------------------------------------------------------------------------
# Portfolio problem: Main structural parameters
#-----------------------------------------------------------------------------------------------
params = {} #Initialize empty dictionary 

# Name experiment for organizing output:
params["experiment_name"] = "chen_decum"    

# Record start time
now = datetime.datetime.now()
print ("Starting at: ")
start_time = now.strftime("%d-%m-%y_%H:%M:%S")
params["start_time"] = start_time
print(start_time)

# set working directory to be where this file is: 
params["working_dir"] = os.path.dirname(os.path.abspath(__file__)) + "/" # Find name of directory of where this code is being run. 
                                                                   # Subsequent file paths should be relative to this. 
os.chdir(params["working_dir"]) 

params["T"] = 30. #Investment time horizon, in years
params["N_rb"] = 30  #Nr of equally-spaced rebalancing events in [0,T]
          #Withdrawals, cash injections, AND rebalancing at times t_n = (n-1)*(T/N_rb), for n=1,...,N_rb
params["delta_t"] = params["T"] / params["N_rb"]    # Rebalancing time interval
params["W0"] = 1000.     # Initial wealth W0
# NOT IMPLEMENTED YET: Cash injections
params["q"] =  0. * np.ones(params["N_rb"])

# automatic flag to see if running on GPU machine.
device = 'cuda' if torch.cuda.is_available() else 'cpu'  
params["device"] = device

# Dynamic NN Withdrawal options
params["nn_withdraw"] = True # flag to include dynamic withdrawals as part of experiment. Turning this to "false" 
                             # will make the script bypass withdrawal network. 
params["q_min"] = 35.0 #min withdrawal per Rb
params["q_max"] = 60.0 #max withdrawal per Rb
# note: consraint function options for withdrawals and asset allocation are defined in NN SETUP section of this file. 


params["mu_bc"] = 0.00 #borrowing spread: annual additional rate on negative wealth (plus bond rate)
# ^TODO: need to implement borrowing interest when wealth is negative

#set seed for both pytorch and numpy
params["random_seed"] = 2
np.random.seed(params["random_seed"])
print("\n numpy seed: ", params["random_seed"], " \n")
torch.manual_seed(params["random_seed"])
print("\n pytorch seed: ", params["random_seed"], " \n")

# Transfer learning flags:
transfer_learn = True #if True, will use weights from previous tracing parameter to initialize NN model. 
transfer_learn_start = 0 # tracing param (i.e. kappa) index (starts at 1) to start transfer learning from  

# preload saved model TODO: combine these and maybe replace entirely?
preload = False

# Set directory to load models from. Will automatically select model of correct tracing parameter if directory of models is given. Otherwise, path must point to model file directly.  
load_model_dir = "saved_models_output/chen_decum_11-10-23_16:11:17" 

# Options for exporting control: This is for creating a control file that Prof. Forsyth can use for his C++ based simulation.
params["output_control"] = False
params["control_filepath"] = params["working_dir"] + "/control_files/feb14_kappa1_add_w1000.txt"
params["w_grid_min"] = 0
params["w_grid_max"] = 10000
params["nx"] = 4096

#Specify TRANSACTION COSTS parameters: NOT YET IMPLEMENTED. 
params["TransCosts_TrueFalse"] = False #If True, incorporate transaction costs
# - if TransCosts_TrueFalse == True, additional parameters will be used
if params["TransCosts_TrueFalse"] is True:
    params["TransCosts_r_b"] = 5/100 #(Real), cont comp. interest rate on borrowed transaction costs
    params["TransCosts_propcost"] = 0.5/100   #proportional TC in (0,1] of trading in any asset EXCEPT cash account
    params["TransCosts_lambda"] = 1e-6  #lambda>0 parameter for smooth quadratic approx to abs. value function

# Parameters for size of experiment:
# Shortcut names to set size of experiment
params["iter_params"] = "small" 

    # parameters for full training loop
if params["iter_params"] == "big":
    N_d_train = int(2.56* (10**5))   # number of random paths simulated or sampled
    itbound = 30000                  # number of training iterations
    batchsize = 1000                 # Mini-batch: number of paths in each Stochastic Gradient Descent iteration 
    nodes_nn = 8                     # number of nodes in each hidden layer for each NN
    layers_nn = 2                    # number of hidden layers for each NN
    biases_nn = True                 # flag to include or exclude biases in each NN
    adam_xi_eta = 0.04               # adam learning rate for xi (candidate VAR) for mean-cvar objective
    adam_nn_eta = 0.05               # adam learning rate for NN parameters
    
    # parameters for testing a trained model
if params["iter_params"] == "check": 
    N_d_train = int(2.56* (10**5)) 
    itbound = 10
    batchsize = 1000
    nodes_nn = 8
    layers_nn = 2
    biases_nn = True
    adam_xi_eta = 0.00  # learning rate set to zero to ensure no training happens when you are testing the model
    adam_nn_eta = 0.00

    # small training loop for testing code functionality -- should be able to get reasonable results from this
if params["iter_params"] == "small":
    N_d_train = int(2.56* (10**4)) 
    itbound = 2000
    batchsize = 1000
    nodes_nn = 8
    layers_nn = 2
    biases_nn = True
    adam_xi_eta = 0.04
    adam_nn_eta = 0.05

    # tiny training loop for debugging code -- results will be nonsense
if params["iter_params"] == "tiny":
    N_d_train = 10
    itbound = 5
    batchsize = 5
    nodes_nn = 8
    layers_nn = 2
    biases_nn = True
    adam_xi_eta = 0.05
    adam_nn_eta = 0.05
#----------------------------------------------

# Main settings for data: This is used for creating both training and testing data. 
params["N_d_train"] = N_d_train #Nr of data return sample paths to bootstrap
params["data_source_Train"] = "simulated" #"bootstrap" or "simulated" [data source for TRAINING data]

# TODO: create better output info about whether NN was trained/tested

#--------------------------------------
# ASSET BASKET: Specify basket of candidate assets, and REAL or NOMINAL data
params["asset_basket_id"] =  "B10_and_VWD"   # Pre-defined basket of underlying candidate assets 
                                             # See fun_Data_assets_basket.py for other asset basket options, and to add new 
                                             # asset baskets. 
params["add_cash_TrueFalse"] = False # This functionality is not implemented, but you can include cash by including T30 in the
                                     # asset basket. 

# Options for factor investing - Only relevant if using factor assets. 
params["factor_constraint"] = False
params["dynamic_total_factorprop"] = False
params["factor_constraints_dict"] = None  

params["real_or_nominal"] = "real" # "real" or "nominal" for asset data for wealth process: if "real", the asset data will be deflated by CPI


# Note: In a previous implementation for the dynamic NN investing ML model, Postdoc Pieter Van Staden had implemented 
# Functionality for calculating confidence penalties and including trading signals in the features of the NN.
# Neither of these have been implemented in this new Pytorch implementation. This is where you might specify parameters for
# that functionality if you were to add it to this implementation. 
#--------------------------------------
# CONFIDENCE PENALTY: 
#--------------------------------------
# TRADING SIGNALS:
#---------------------------------------


# -----------------------------------------------------------------------------------------------
#  OBJECTIVE FUNCTION:  CHOICE AND PARAMETERS
# -----------------------------------------------------------------------------------------------
params["obj_fun"] = "mean_cvar_single_level"  # see fun_Objective_functions.py to see other objective function options, or
                                              # to add additional objective functions. 
params["obj_fun_epsilon"] =  10**-6 # epsilon in Forsyth stablization term. Not really needed for NN approach, but we usually
                                    # want it so we are solving the same objective function. 
                                    
# XI training settings (candidate VAR) for mean CVAR objective function:

params["xi_0"] = 100. #initial value -- will be overwritten if pre-loading a model.
params["xi_lr"] = adam_xi_eta # learning rate for xi (separate from NN learning rate)

# Quadratic smoothing options for mean-cvar function. 
params["smooth_cvar_func"] = True
params["lambda_quad"] = 10**(-6)

# Note: The only objective functions that have been implemented here are the mean-cvar and linear shortfall functions.
# To add objective functions, they must be implemented in fun_Objective_functions.py. 

# Here are some notes on objective functions that have been used in previous implementations:
# STANDARD objective functions ofAdam W(T): obj_fun options could include:
# "mean_cvar_single_level",
# "one_sided_quadratic_target_error",
# "quad_target_error", "huber_loss"
# "meancvarLIKE_constant_wstar"   (NOT true mean-cvar)
# "ads" for constant target



# TRACING PARAMETERS (i.e., kappa for mean cvar objective functions)
# set k = 999. for Inf case!
tracing_parameters_to_run =  [1.0, 2.0]

#[0.05, 0.2, 0.5, 1.0, 1.5, 3.0, 5.0, 50.0]
 #[float(item) for item in sys.argv[1].split(",")]  #[0.1, 0.25, 0.4, 0.6, 0.7, 0.8, 0.9, 1.0] + np.around(np.arange(1.1, 3.1, 0.1),1).tolist() + [10.0]

# print("tracing parameter(s) entered from terminal: ", sys.argv[1])
#[float(item) for item in sys.argv[1].split(",")] #Must be LIST


# TRACING PARAMETERS interpreted as follows:
#   -> STANDARD objectives tracing parameters:
#   params["obj_fun_rho"] if obj_fun is in ["mean_cvar_single_level"], *larger* rho means we are looking for a *higher* mean
#       this "rho" is usually referred to as "\kappa" in Forsyth and Li's papers.
#   params["obj_fun_W_target"] if obj_fun is in ["one_sided_quadratic_target_error", "quad_target_error", "huber_loss", "ads"]


# Set objective function parameters [rarely changed]

# -----------------------------------------------------------------------------
if params["obj_fun"] == "mean_cvar_single_level":  # SINGLE level formulation, no Lagrangian
    params["obj_fun_alpha"] = 0.05  # alpha for alpha_CVaR, must be in DECIMAL format (not *100)
    params["obj_fun_lambda_smooth"] = 1e-07  # 1e-07  # Lambda for smoothed version of mean-CVAR objective,assumed =0 for no smoothing if not set
# -----------------------------------------------------------------------------
elif params["obj_fun"] == "meancvarLIKE_constant_wstar":  # NOT true mean-cvar!
    params["obj_fun_alpha"] = 0.05  # alpha for alpha_CVaR, must be in DECIMAL format (not *100)
    params["obj_fun_lambda_smooth"] = 1e-07  # 1e-07  # Lambda for smoothed version of mean-CVAR objective,assumed =0 for no smoothing if not set
    params["obj_fun_LIST_constant_wstar_tested"] = [784.]   #List of CONSTANT wstar(s),
    # in usual mean-CVAR this would be candidate value for value-at-risk at level alpha
    #Code below will loop over wstar in params["LIST_constant_wstar_tested"], and
    # optimize "mean-cvar" objective (minimize) for each constant wstar
    # -- ALL OUTPUTS (Training + Testing) will be chosen for value of wstar
    #       such that params_TRAIN["F_val"] is SMALLEST

#add parameters for new objective functions here


#-----------------------------------------------------------------------------------------------
# Experiment Output Options
#-----------------------------------------------------------------------------------------------


output_parameters = {}

#Basic output params
output_parameters["code_title_prefix"] = params["experiment_name"]+"_"+start_time+"/" # prefix to name results, i.e. plots and results summary # used as prefix for naming files when saving outputs

# directories to save results output and trained models
params["results_dir"] = "results_output/" + output_parameters["code_title_prefix"] 
params["saved_model_dir"] = "saved_models_output/" + output_parameters["code_title_prefix"] 
os.makedirs(params["saved_model_dir"], exist_ok=True)
os.makedirs(params["results_dir"], exist_ok=True)

output_parameters["output_results_Excel"] = True      #Output results summary to Excel

output_parameters["save_Figures_format"] = "png"  # if saving figs, format to save figures in, e.g. "png", "eps",

# W_T, pdf and cdf
output_parameters["output_W_T_vectors_Excel"] = False       #output the terminal wealth vectors (train, test and benchmark) for histogram construction
output_parameters["output_W_T_histogram_and_cdf"] = True        #output Excel sheet with numerical histogram and CDF of terminal wealth (train, test and benchmark)
output_parameters["output_W_T_histogram_and_cdf_W_max"] = 1000.   # Maximum W_T value for histogram and CDF.
output_parameters["output_W_T_histogram_and_cdf_W_bin_width"] = 5.   # Bin width of wealth for histogram and CDF.

#Output benchmark stats
output_parameters["output_W_T_benchmark_comparisons"] = False #If true, outputs W_T vs Benchmark differences and ratios


# Rolling historical test: NOT IMPLEMENTED IN PYTORCH YET----
output_parameters["output_Rolling_Historical_Data_test"] = False  #If true, test NN strategy and benchmark strategy on actual single
                                    # historical data path, starting in each month and investing for the duration
output_parameters["fixed_yyyymm_list"] = [198001, 198912, 199001] #Used for historical rolling test; LIST of yyyymm_start months of particular interest
output_parameters["output_Rolling_Historical_only_for_fixed"] = False  #if True, outputs rolling historical ONLY for output_parameters["fixed_yyyymm_list"]
#-----------------------------------------------------------

# NN detail
output_parameters["output_TrainingData_NNweights_test"] = False   #if true, outputs the training paths + features + NN weights required to
                                            # reproduce the top and bottom k results for the terminal wealth and objective function values

# PERCENTILES:
output_parameters["output_Pctiles_Excel"] = True #If True, outputs Excel spreadsheet with NN pctiles of proportions in each asset and wealth over time
output_parameters["output_Pctiles_Plots"] = True #if True, plots the paths of output_Pctiles_Excel over time
output_parameters["output_Pctiles_Plots_W_max"] = 2000. #Maximum y-axis value for WEALTH percentile plots
output_parameters["output_Pctiles_list"] = [5,50,95]  #Only used if output_Pctiles_Excel or output_Pctiles_Plots is True, must be list, e.g.  [20,50,80]
output_parameters["output_Pctiles_on_TEST_data"] = False #Output percentiles for test data as well

#Control heatmap params [used if save_Figures == True]
output_parameters["save_Figures_FunctionHeatmaps"] = True #if True, plot and save the function heatmap figures
output_parameters["output_FunctionHeatmaps_Excel"] = True  # If TRUE, output the heatmap grid data to Excel

output_parameters["save_Figures_DataHeatmaps"] = True   #if True, plot and save the data heatmap figures
output_parameters["output_DataHeatmaps_Excel"] = False   # If TRUE, output the heatmap grid data to Excel

output_parameters["heatmap_y_bin_min"] = 0.  # minimum for the y-axis grid of heatmap
output_parameters["heatmap_y_bin_max"] = 2000.0  # maximum for the y-axis grid of heatmap
output_parameters["heatmap_y_num_pts"] = int(output_parameters["heatmap_y_bin_max"] - output_parameters["heatmap_y_bin_min"])+1  # number of points for y-axis
output_parameters["heatmap_xticklabels"] = 5  # e.g. xticklabels=6 means we are displaying only every 6th xaxis label to avoid overlapping
output_parameters["heatmap_yticklabels"] = 500  # e.g. yticklabels = 10 means we are displaying every 10th label
output_parameters["heatmap_cmap"] = "rainbow"  # e.g. "Reds" or "rainbow" etc colormap for sns.heatmap
output_parameters["heatmap_cbar_limits"] = [0.0, 1.0]  # list in format [vmin, vmax] for heatmap colorbar/scale
output_parameters["percentile_cutoffs"] = True

#-----------------------------------------------------------------------------------------------
# Asset basket and Feature specification (also specify basket of trading signals, if applicable)
#-----------------------------------------------------------------------------------------------

#Construct asset basket:
# - this will also give COLUMN NAMES in the historical returns data to use
params["asset_basket"] = fun_Data_timeseries_basket.timeseries_basket_construct(
                            basket_type="asset",
                            basket_id=params["asset_basket_id"],
                            add_cash_TrueFalse=params["add_cash_TrueFalse"],
                            real_or_nominal = params["real_or_nominal"] )


#get index of B10 for borrowing cost (For when the 10 year treasury is used as proxy rate for reverse mortgage.)
params["b10_idx"] = params["asset_basket"]['basket_columns'].index('B10_real_ret')
#Assign number of assets based on basket information:
params["N_a"] = len(params["asset_basket"]["basket_columns"])   #Nr of assets = nr of output nodes


#Initialize number of NN input nodes: More input features would be needed if using trading signals or stochastic benchmark.  
params["N_phi"] =  2  #Nr of default features, i.e. the number of input nodes
params["feature_order"] = ["time_to_go", "stdized_wealth"]  #initialize order of the features

params["N_phi_standard"] = params["N_phi"] # Set nr of basic features *BEFORE* we add trading signals


#-----------------------------------------------------------------------------------------------
# Gathering Historical Market data: ASSETS and FEATURES (trading signals):
#   If market data required for bootstrapping, extracted, processed (e.g. inflation adjusted)
#   and prepared for bootstrapping here.
#-----------------------------------------------------------------------------------------------

#If market data is required (checked inside code), following fields appended/modified to params dictionary:
#          params["asset_basket"]: (existing field) modified by appending historical data
#                           and associated key stats (mean, stdev, corr matrix) to asset_basket
#          params["asset_basket_data_settings"]: new dictionary appended  historical data extraction settings for record
#          params["trading_signal_basket"]:  (existing field) modified by appending historical data
#                                   and associated key stats (mean, stdev, corr matrix) to trading_signal_basket
#                                   Trading signals constructed *lagged* to avoid look-ahead
#          params["trading_signal_basket_data_settings"]: new dictionary appended historical data extraction settings for record
#          params["bootstrap_source_data"]: (new field) pandas.DataFrame with time series ready for bootstrapping:
#                                           1) Inflation adjusted if necessary,
#                                           2) Trade signals and asset returns merged
#                                           3) NaNs removed (at start due to trade signal calculation)
#               for a given month, asset obs are at END of month, trade signals at BEGINNING of month

# Note: if real_or_nominal = "real" (assets or trade signals), the inflation-adjusted returns time series will be constructed here

params = fun_Data__bootstrap_wrapper.wrap_append_market_data(
                            params = params,  #params dictionary as in main code
                            data_read_yyyymm_start = 192607, #Start date to use for historical market data, set to None for data set start
                            data_read_yyyymm_end = 202212,  #End date to use for historical market data, set to None for data set end
                            data_read_input_folder = "Market_data", #folder name (relative path)
                            data_read_input_file = "_PVS_ALLfactors_CRSP_FF_data_202304_MCfactors", #just the filename, no suffix
                            data_read_input_file_type = ".xlsx",  # file suffix
                            data_read_delta_t = 1 / 12,  # time interval for returns data (monthly returns means data_delta_t=1/12)
                            data_read_returns_format = "percentages",  # 'percentages' = already multiplied by 100 but without added % sign
                                                                        # 'decimals' is percentages in decimal form
                            data_read_skiprows = 15 , # nr of rows of file to skip before start reading
                            data_read_index_col = 0,  # Column INDEX of file with yyyymm to use as index
                            data_read_header = 0,  # INDEX of row AFTER "skiprows" to use as column names
                            data_read_na_values = "nan" # how missing values are identified in the data
                            )
#          params["bootstrap_source_data"]: (new field) pandas.DataFrame with time series ready for bootstrapping:
#                                           1) Inflation adjusted if necessary,
#                                           2) Trade signals and asset returns merged
#                                           3) NaNs removed (at start due to trade signal calculation)
#               for a given month, asset obs are at END of month, trade signals at BEGINNING of month

#Output bootstrap source data to Excel, if needed
output_bootstrap_source_data = False
if output_bootstrap_source_data:
    df_temp = params["bootstrap_source_data"]
    df_temp.to_excel(output_parameters["code_title_prefix"] + "bootstrap_source_data.xlsx")
    

#-----------------------------------------------------------------------------------------------
# MARKET DATA GENERATOR: Source data for training\testing
#-----------------------------------------------------------------------------------------------

params["output_csv_data_training_testing"] = False  #if True, write out training/testing data to .csv files

if params["data_source_Train"] == "bootstrap":  # TRAINING data bootstrap using historical data

    # ----------------------------------------
    # TRAINING data bootstrapping
    # - Append bootstrapped data to "params" dictionary
    blocksize = 6
    print("Bootstrap block size: " + str(blocksize))
    params = fun_Data__bootstrap_wrapper.wrap_run_bootstrap(
        train_test_Flag = "train",                  # "train" or "test"
        params = params,                            # params dictionary as in main code
        data_bootstrap_yyyymm_start = 196307,       # start month to use subset of data for bootstrapping, CHECK DATA!
        data_bootstrap_yyyymm_end = 202212,         # end month to use subset of data for bootstrapping, CHECK DATA!
        data_bootstrap_exp_block_size = blocksize,  # Expected block size in terms of frequency of market returns data
                                                    # e.g. = X means expected block size is X months of returns
                                                    # if market returns data is monthly
        data_bootstrap_fixed_block = False,         # if False: use STATIONARY BLOCK BOOTSTRAP, if True, use FIXED block bootstrap
        data_bootstrap_delta_t = 1 / 12             # time interval for returns data (monthly returns means data_delta_t=1/12)
    )

    # ASSET return data: always appended (if market data required
    #   params["Y_train"][j, n, i] = Return, along sample path j, over time period (t_n, t_n+1), for asset i
    #       -- IMPORTANT: params["Y_train"][j, n, i] entries are basically (1 + return), so it is ready for multiplication with start value
    #   params["Y_order_train"][i] = column name of asset i used for identification

    # TRADING SIGNAL data: only appended if params["use_trading_signals_TrueFalse"] == True

    #   params["TradSig_train"][j, n, i] = Point-in time observation for trade signal i, along sample path j, at rebalancing time t_n;
    #                               can only rely on time series observations <= t_n
    #   params["TradSig_order_train"][i] = column name of trade signal i used for identification

    # ----------------------------------------

elif params["data_source_Train"] == "simulated":

# ----------------------------------------
    # TRAINING data simulation, "Synthetic Data"
    # - Append simulated data to "params" dictionary
    params = fun_Data__MCsim_wrapper.wrap_run_MCsim(
                train_test_Flag = "train",  # "train" or "test"
                params = params,  # params dictionary as in main code
                model_ID_set_identifier = "Forsyth_retirementDC_2020" # (see code)identifier for the collection of models AND correlations to use
                )
    # ASSET return data:
    #   params["MCsim_info_train"]: inputs used to get the MC simulation results
    #   params["Y_train"][j, n, i] = Return, along sample path j, over time period (t_n, t_n+1), for asset i
    #       -- IMPORTANT: params["Y_train"][j, n, i] entries are basically (1 + return), so it is ready for multiplication with start value
    #   params["Y_order_train"][i] = column name of asset i used for identification


#-----------------------------------------------------------------------------------------------
# NEURAL NETWORK (NN) SETUP
#-----------------------------------------------------------------------------------------------

# Build Withdrawal NN, if needed: NN_withdrawal
#---------------------------

if params["nn_withdraw"]:

    nn_options_q = {}              # _q indicates for withdrawals
    nn_options_q["nn_purpose"] = "withdrawal"
    nn_options_q["N_layers_hid"] = 2   # Nr of hidden layers of NN
                                # NN will have total layers 1 (input) + N_L (hidden) + 1 (output) = N_L + 2 layers in total
                                # layer_id list: [0, 1,...,N_L, N_L+1]

    nn_options_q["N_nodes_input"] = params["N_phi"]    # number of input nodes. By default, feature vector phi includes just wealth and time
    nn_options_q["N_nodes_hid"] = 8                   # number of nodes to add to N_a (number of assets) to set total nodes in each hidden layer
    nn_options_q["N_nodes_out"] = 1             # number of nodes in output layer (withdrawal NN only takes 1)
    nn_options_q["hidden_activation"] = "logistic_sigmoid"       # Type of activation function for hidden layers
    nn_options_q["output_activation"] = "none"                  # Type of activation function for output layer. 
                                                            # NOTE: When needing any kind of constraint activation function, this should be set to "none" so that the custom activation that encodes constraints can be applied in the fun_invest_NN.py file instead of in the PyTorch NN object itself. 
                                                            
    # Set custom activation function with constraint. This is not applied in PyTorch object, and is instead applied as separate function in the fun_invest_NN.py file.                                                       
    params["w_constraint_activation"] = "yy_fix_jan29"   #"yy_fix_jan29" is the standard withdrawal constraint used in Forsyth (2022) "Nasty" paper, as well as all decumulation problems worked on by Mohib Shirazi and Marc Chen. 

    nn_options_q["biases"] = True       # add biases
    #store options for record keeping
    params["nn_options_q"] = nn_options_q

    # NN initialization occurs in tracing parameter loop to ensure resetting/continuing parameters between runs

#-------------------------------------
# Build Allocation NN: NN_allocate 
#---------------------------
nn_options_p = {}              # _p indicates for allocation
nn_options_p["nn_purpose"] = "allocation"
nn_options_p["N_layers_hid"] = 2   # Nr of hidden layers of NN
                               # NN will have total layers 1 (input) + N_L (hidden) + 1 (output) = N_L + 2 layers in total
                               # layer_id list: [0, 1,...,N_L, N_L+1]

nn_options_p["N_nodes_input"] = params["N_phi"]    # number of input nodes. By default, feature vector phi includes just wealth and time
nn_options_p["N_nodes_hid"] = 8                   # number of nodes to add to N_a (number of assets) to set total nodes in each hidden layer
nn_options_p["N_nodes_out"] = params["N_a"]       # number of nodes to add to N_a (number of assets) to set total nodes in each hidden layer  
nn_options_p["hidden_activation"] = "logistic_sigmoid"       # Type of activation function for hidden layers
nn_options_p["output_activation"] = "softmax"                  # Type of activation function for output layer. 
                                                        # NOTE: When needing any kind of constraint activation function, this should be set to "none" so that the custom activation that encodes constraints can be applied in the fun_invest_NN.py file instead of in the PyTorch NN object itself. 

# Asset custom constraint functions:
# So far, these are only implemented for asset baskets including factor assets
params["factor_constraint"] = False # True or False TODO: switch to change between "indiv", "group", and "None"
params["dynamic_total_factorprop"] = False # True or False to switch on group constraint for all factor assets together
params["factor_constraints_dict"] = None
                                                        
nn_options_p["biases"] = True             # add biases
#store options for record keeping
params["nn_options_p"] = nn_options_p


#-----------------------------------------------------------------------------------------------
# CONSTANT BENCHMARK: Specify constant proportion strategy
#-----------------------------------------------------------------------------------------------

# The constant benchmark is used to sanity check NN results, validate data sets, create NN feature standardization parameters, etc. It runs automatically before every training loop or testing. 

#prop_const[i] = constant proportion to invest in asset index i \in {0,1,...,N_a -1}
#               Order corresponds to asset index in sample return paths in params["Y"][:, :, i]

#Equal Allocations across assets: automatic for number of assets. 
params["benchmark_prop_const"] = np.ones(params["N_a"]) * (1/ params["N_a"])  # automatically get equal proportions

# Can manually specific more complicated splits, for example:
# if params["N_a"] == 2:
#     #prop_const = np.ones(params["N_a"]) / params["N_a"] #Equal split
#     prop_const = np.array([0.6, 0.4])
# elif params["N_a"] == 5:
#     prop_const = np.array([0.1, 0.3, 0.36, 0.12, 0.12])

# Constant Withdrawal: Most common constant benchmark is Bengen 4% rule, which is 40 if starting wealth is 1000. 
params["withdraw_const"] = 40.0

# Set sideloading feature standardization flag to false by default. Don't change this! This will be updated later if needed. 
params["sideloaded_standardization"] = False


#----------------------------------------------------------------------------------------
# TRAINING, TESTING and OUTPUTS of NN
#----------------------------------------------------------------------------------------

#Specify dictionary training_options used in training of NN: Set default values
NN_training_options = fun_train_NN_algorithm_defaults.train_NN_algorithm_defaults()

#NOTE: NN_training_options["methods"] specifies algorithm(s) used to train NN:
    #       -> can specify multiple methods
    #       DEFAULT = **ALL** methods coded using both scipy.minimize and SGD algorithms
    #       *smallest* objective func value returned by ANY of the methods specified is used as the final value
    #        along with its associated parameter vector


#Override some of the NN_training_options set above if needed
NN_training_options["methods"] = ["Adam"]
NN_training_options["Adam_ewma_1"] = 0.9
NN_training_options["Adam_ewma_2"] = 0.998 #0.999
NN_training_options["Adam_eta"] = adam_nn_eta #override 0.1
NN_training_options["Adam_weight_decay"] = 1e-4
NN_training_options['nit_running_min'] = int(itbound / 100)  # nr of iterations at the end that will be used to get the running minimum for output
NN_training_options["itbound_SGD_algorithms"] = itbound
NN_training_options["batchsize"] = batchsize
NN_training_options["nit_IterateAveragingStart"] = int(itbound * 9 / 10)  # Start IA 90% of the way in
NN_training_options["running_min_from_avg"] = False #if true, take running min from avg
NN_training_options["running_min_from_sgd"] = True #if true, take running min from sgd, both can be true
NN_training_options["lr_schedule"] = True  #If true, set to divide lr by 10 at 70% and 97%

print("\n NN training settings: ")
print(NN_training_options)

# Save experiment parameters in results summary file before beginning training:----
result_summary = {}
exp_details = copy.deepcopy(params)

# delete all fields that can't be put into json
for key in params.keys():
    if fun_utilities.is_jsonable(exp_details[key]) == False:
        del exp_details[key]

# save to json
result_summary["exp_details"] = exp_details
out_file = open(params["results_dir"]+"summary_all_points.json", "w") 
json.dump(result_summary, out_file, indent = 6)
out_file.close() 
#---------------------------------------------------------------------------


# RUN EXPERIMENT
# -----------------------------------------------------------------------------
# Loop over tracing parameters [scalarization or wealth targets] and do training, testing and outputs
for i,tracing_param in enumerate(tracing_parameters_to_run): #Loop over tracing_params
    
    # setting tracing parameters to 999 sets up logic for kappa = Inf in other files. 
    if tracing_param == 999.:
        params["kappa_inf"] = True
    else:
        params["kappa_inf"] = False
        
    # Set directory to save results for each kappa point
    params["results_dir_kappa"] = "results_output/" + output_parameters["code_title_prefix"] + \
                                    "kappa_" + str(tracing_param) + "/"
    os.makedirs(params["results_dir_kappa"], exist_ok=True)
    
    # INITIALIZE NNs:-----------------------------------------
    
    
    if not preload:
        #withdrawal network
        if params["nn_withdraw"]:
            NN_withdraw = class_NN_Pytorch.pytorch_NN(nn_options_q)

            # print structure
            pd.set_option("display.max_rows", None, "display.max_columns", None)
            print(f"Intialized NN model structure for withdrawal: ")
            print(NN_withdraw.model) 
            
            # copy NN structures to correct device
            NN_withdraw.to(device)
            
        #allocation network
        NN_allocate = class_NN_Pytorch.pytorch_NN(nn_options_p)
        NN_allocate.to(device)
        # print structure
        print(f"Intialized NN model structure for allocation: ")
        print(NN_allocate.model)     
        
        # place in list object for convenience
        if params["nn_withdraw"]:
            NN_list = torch.nn.ModuleList([NN_withdraw, NN_allocate])
        else:
            NN_list = torch.nn.ModuleList([NN_allocate])
        #--------------------------------------------------------------
        
    
    #PRELOAD NN MODEL----------------------------------------------
    
    
    # manual preload: load at every tracing param, but is overridden by continuation learn if cont is set to true
    if preload:
        
        if os.path.isdir(load_model_dir):
            files = [f for f in os.listdir(load_model_dir)]
            suffix_len = len(str(tracing_param))
            # get filepath for the  
            kappa_filepath = [path for path in files if re.split('kappa_|.pkl', path)[1] == str(tracing_param)]
            
            if len(kappa_filepath) > 1:
                raise Exception(f"Warning: Multiple models available for kappa = {tracing_param} available.")
            elif kappa_filepath == []: # if no model for current kappa exists
                raise Exception(f"Warning: No pre-saved model for kappa = {tracing_param} available.")
            else:
                preload_path = Path(load_model_dir + "/" + kappa_filepath[0])
        else:
            preload_path = load_model_dir
                        
        #LOAD MODEL and update 'params' with necessary metadata      
        NN_list, params = manage_nn_models.load_model(preload_path, params)
        print("pre-loaded NN: ", NN_list.state_dict())
        
           
    # Load transfer learn model from previous kappa point
    if i == 0:
        last_kappa = 0  # if first kappa point, don't load model
    else:
        last_kappa = tracing_parameters_to_run[i-1]
    model_save_path = params["saved_model_dir"]
    last_model_path = Path(f"{model_save_path}kappa_{last_kappa}.pkl")
    
    if last_model_path.is_file() and transfer_learn:
        
        NN_list, params = manage_nn_models.load_model(last_model_path, params)
        
        print(f"Transfer learning model loaded from kappa_{last_kappa}.")
                   
    
    # SET TRACING PARAMETERS inside params ----------------------------
    if params["obj_fun"] in ["one_sided_quadratic_target_error", "quad_target_error", "huber_loss", "ads"]:
        params["obj_fun_W_target"] = tracing_param  # all of them use Wealth target

    elif params["obj_fun"] in ["mean_cvar_single_level", "meancvarLIKE_constant_wstar"]:
        params["obj_fun_rho"] = tracing_param  # all of them use rho (scalarization parameter)


    # Do ONE-STAGE optimization for some objectives ----------------------------
    if params["obj_fun"] in ["mean_cvar_single_level",
                             "one_sided_quadratic_target_error",
                             "quad_target_error"]:

        params_TRAIN, params_CP_TRAIN, params_TEST, params_CP_TEST = \
            fun_RUN__wrapper.RUN__wrapper_ONE_stage_optimization(
                params=params,  # dictionary as setup in the main code
                NN_list=NN_list,  # list of both torch NNs
                NN_training_options=NN_training_options,
                # dictionary with options to train NN, specifying algorithms and hyperparameters
                output_parameters = output_parameters  # Dictionary with output parameters as setup in main code
            )

    else:
        raise ValueError("PVS error in main code: params['obj_fun'] = " + params['obj_fun'] +
                         " not coded in MAIN code optimization loop.")


    # Give update to on tracing parameter progress
    print("-----------------------------------------------")
    print("Just FINISHED: ")
    print("Asset basket ID: " + params["asset_basket_id"])
    print("Objective function: " + params["obj_fun"])
    print("Tracing param: " + str(tracing_param))
    # print("F value: " + str(params_TRAIN["F_val"]))
    print("-----------------------------------------------")

#END: Loop over tracing_params

# Plot all results
with open(params["results_dir"]+"summary_all_points.json") as in_file:
    results_dict = json.load(in_file)

EF_plotter.plot_from_results_dict(results_dict, params["results_dir"])



