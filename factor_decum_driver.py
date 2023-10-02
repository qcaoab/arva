
# Portfolio optimization problem using NN to model dual control for decumulation/multi factor problem 

run_on_my_computer = False   #True if running in my local desktop, False if running remotely
#-- this flag affects use of matplotlib.use('Agg') below


import pandas as pd
import numpy as np
import math
import os
import gc   #garbage collector
import datetime
import pickle
import sys
import datetime
import codecs, json
from pathlib import Path
import re

#Import files needed (other files are imported within those files as needed)
import fun_Data_timeseries_basket
import fun_Data__bootstrap_wrapper
import fun_Data__MCsim_wrapper
import class_Neural_Network
import fun_train_NN_algorithm_defaults
import fun_RUN__wrapper
import class_NN_Pytorch
import torch

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

# Record start time
now = datetime.datetime.now()
print ("Starting at: ")
start_time = now.strftime("%d-%m-%y_%H:%M")
params["start_time"] = start_time
print(start_time)

#filepath prefixes TODO: clean this up 
abspath = os.path.abspath(__file__)
dname = os.path.dirname(abspath)
os.chdir(dname)
code_title_prefix = "output_heatmaps_constrain/mc_decum_"+start_time+"/"   #used for saving output on local
console_output_prefix = "mc_decum_" +start_time+"/"
params["console_output_prefix"] = console_output_prefix


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
cont_nn = False  #if True, will use weights from previous tracing parameter to initialize NNtheta0. 
cont_nn_start = 3 # tracing param (i.e. kappa) index (starts at 1) to start transfer learning from  
cont_xi = False # uses previous value of optimal xi to initialize xi in next run
cont_xi_start = 3  #tracing param index (starts at 1) to start transfer learning at

# preload saved model TODO: combine these and maybe replace entirely?
preload = False
params["local_path"] = str(os.getcwd())

nn_preload = "/home/marcchen/Documents/constrain_factor/researchcode/saved_models/NN_opt_mc_decum_30-06-23_19:28_kappa_1.0" 
xi_preload = "/home/marcchen/Documents/constrain_factor/researchcode/saved_models/xi_opt_mc_decum_30-06-23_19:28_kappa_1.0.json" 

# Flag for side loading standardization parameters: necessary when you are testing on a distribution different from the 
# training distribution. 
params["sideload_standardization"] = False
    
params["standardization_file_path"] = "/home/marcchen/Documents/constrain_factor/researchcode/saved_models/standardizing_opt_mc_decum_27-06-23_18:47_kappa_1.0.json"

# Options for exporting control: This is for creating a control file that Prof. Forsyth can use for his C++ based simulation.
params["output_control"] = False
params["control_filepath"] = params["local_path"] + "/control_files/feb14_kappa1_add_w1000.txt"
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
params["iter_params"] = "tiny" 

    # parameters for full training loop
if params["iter_params"] == "big":
    N_d_train = int(2.56* (10**5))   # number of random paths simulated or sampled
    itbound = 20000                  # number of training iterations
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

    # small training loop for testing code functionality -- should be able to get reasonable performance from this
if params["iter_params"] == "small":
    N_d_train = int(2.56* (10**4)) 
    itbound = 4000
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

# hold xi (candidate VAR) constant? Only used for debugging purposes with mean-CVAR objective. 
params["xi_constant"] = False

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

#STOCHASTIC BENCHMARK objective functions:
# "ads_stochastic" using asymm. dist. shaping as in Ni, Li, Forsyth (2020)
# "qd_stochastic":  Quadratic deviation from elevated target as in Forsyth (2021)
# "ir_stochastic" info ratio using *STOCHASTIC TARGET* as in Goetzmann et al (2002)
# "te_stochastic": Tracking error as in Forsyth (2021)



#k = 999. for Inf case!
tracing_parameters_to_run =  [1.0]

#[0.05, 0.2, 0.5, 1.0, 1.5, 3.0, 5.0, 50.0]
 #[float(item) for item in sys.argv[1].split(",")]  #[0.1, 0.25, 0.4, 0.6, 0.7, 0.8, 0.9, 1.0] + np.around(np.arange(1.1, 3.1, 0.1),1).tolist() + [10.0]

# print("tracing parameter(s) entered from terminal: ", sys.argv[1])
#[float(item) for item in sys.argv[1].split(",")] #Must be LIST


# TRACING PARAMETERS interpreted as follows:
#   -> STANDARD objectives tracing parameters:
#   params["obj_fun_rho"] if obj_fun is in ["mean_cvar_single_level"], *larger* rho means we are looking for a *higher* mean
#       this "rho" is usually referred to as "\kappa" in Forsyth and Li's papers.
#   params["obj_fun_W_target"] if obj_fun is in ["one_sided_quadratic_target_error", "quad_target_error", "huber_loss", "ads"]


# It could also be interpreted as follows for other objective functions: (none of these are implemented)
#   -> STOCHASTIC BENCHMARK objectives tracing parameters:
# "ads_stochastic": params["obj_fun_ads_beta"]>=0 which is the annual target outperformance rate, in exp(beta*T)
# "qd_stochastic": params["obj_fun_qd_beta"] >= 0, this is beta in exp(beta*T)
# "ir_stochastic": [Information ratio] then this is the embedding parameter gamma >0
# "te_stochastic": [Tracking error] this is "beta" >=0 or "beta_hat" >=1:
#       if params["obj_fun_te_beta_hat_CONSTANT_TrueFalse"] = True, this is beta_hat
#       if params["obj_fun_te_beta_hat_CONSTANT_TrueFalse"] = False, this is beta in exp(beta*t_n)


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
output_parameters["code_title_prefix"] = code_title_prefix # used as prefix for naming files when saving outputs
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
    df_temp.to_excel(code_title_prefix + "bootstrap_source_data.xlsx")
    

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

# Build Withdrawal NN, if needed:
#---------------------------

# nn_options_q = {}              # _q indicates for withdrawals
# nn_options_q["N_layers_h"] = 2   # Nr of hidden layers of NN
#                                # NN will have total layers 1 (input) + N_L (hidden) + 1 (output) = N_L + 2 layers in total
#                                # layer_id list: [0, 1,...,N_L, N_L+1]

# nn_options_q["N_input"] = params["N_phi"]              # number of input nodes
# nn_options_q["N_nodes"] = 8               # number of nodes to add to N_a (number of assets) to set total nodes in each hidden layer    
# nn_options_q["hidden_activation"] = "logistic_sigmoid"       # Type of activation function for hidden layers
# nn_options_q["output_activation"] = "none"                  # Type of activation function for output layer. 
#                                                         # NOTE: When needing any kind of constraint activation function, this should be set to "none" so that the custom activation that encodes constraints can be applied in the fun_invest_NN.py file instead of in the PyTorch NN object itself. 
# nn_options_q["biases"] = True             # add biases
# #store options for record keeping
# params["nn_options_q"] = nn_options_q

# #initialize NN object for withdrawal network
# NN_withdraw = class_NN_Pytorch.pytorch_NN(nn_options_q)


#  TEMPORARY
#---------------------------
params["w_constraint_activation"] = "yy_fix_jan29"
params["N_L_withdraw"] = 2   # Nr of hidden layers of NN
                   # NN will have total layers 1 (input) + N_L (hidden) + 1 (output) = N_L + 2 layers in total
                   # layer_id list: [0, 1,...,N_L, N_L+1]


NN_withdraw_orig = class_Neural_Network.Neural_Network(n_nodes_input = params["N_phi"],
                                         n_nodes_output = 1,
                                         n_layers_hidden = params["N_L_withdraw"])

print("Withdrawal NN:")
NN_withdraw_orig.print_layers_info()  #Check what to update

#Update layers info
nodes_mc = 8
biases_mc = True

for l in range(1, params["N_L_withdraw"]+1):
    NN_withdraw_orig.update_layer_info(layer_id = l , n_nodes = params["N_a"] + nodes_mc , activation = "logistic_sigmoid", add_bias=biases_mc)
    
NN_withdraw_orig.update_layer_info(layer_id = params["N_L_withdraw"]+1, activation = "none", add_bias= False) #output layer

NN_withdraw_orig.print_layers_info() #Check if structure is correct
# ---------------------------------------------------------------------


# Allocation NN: NN_allocate 
#---------------------------
params["N_L_allocate"] = layers_nn # Nr of hidden layers of NN
                   # NN will have total layers 1 (input) + N_L (hidden) + 1 (output) = N_L + 2 layers in total
                   # layer_id list: [0, 1,...,N_L, N_L+1]


NN_allocate_orig = class_Neural_Network.Neural_Network(n_nodes_input = params["N_phi"],
                                         n_nodes_output = params["N_a"],
                                         n_layers_hidden = params["N_L_allocate"])

print("Allocation NN:")
NN_allocate_orig.print_layers_info()  #Check what to update

#Update layers info
for l in range(1, layers_nn+1):
    NN_allocate_orig.update_layer_info(layer_id = l , n_nodes = params["N_a"] + nodes_nn , activation = "logistic_sigmoid", add_bias=biases_nn)

#changed activ to none for constraint function

if params["factor_constraint"]:
    NN_allocate_orig.update_layer_info(layer_id = layers_nn+1, activation = "none", add_bias= False)

else:
    NN_allocate_orig.update_layer_info(layer_id = layers_nn+1, activation = "softmax", add_bias= False)


NN_allocate_orig.print_layers_info() #Check if structure is correct
#---------------------------------------------------------------------


#put original NNs in list:
    
NN_orig_list = [NN_withdraw_orig, NN_allocate_orig]
    
# copy NN structures into pytorch NN
NN_withdraw = class_NN_Pytorch.pytorch_NN(NN_withdraw_orig)
NN_withdraw.to(device)
# NN_withdraw.import_weights(NN_withdraw_orig, params)

NN_allocate = class_NN_Pytorch.pytorch_NN(NN_allocate_orig)
NN_allocate.to(device)
# NN_allocate.import_weights(NN_allocate_orig, params)

NN_list = torch.nn.ModuleList([NN_withdraw, NN_allocate])


# L2 weight regularization (only weights, not biases)
params["lambda_reg"] = 0.0 #1e-08 #1e-07    #Set to zero for no weight regularization


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


#OVERRIDE any of the default values in NN_training_options, for example:
#NN_training_options["methods"] = ["CG", "Adam", "RMSprop"]

#NN_training_options["itbound_SGD_algorithms"] = 12000

#Override some of the NN_training_options set above if needed
NN_training_options["methods"] = [ "Adam"]
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

print(NN_training_options)


# -----------------------------------------------------------------------------
# Loop over tracing parameters [scalarization or wealth targets] and do training, testing and outputs
for i,tracing_param in enumerate(tracing_parameters_to_run): #Loop over tracing_params
    
    if tracing_param == 999.:
        params["kappa_inf"] = True
    else:
        params["kappa_inf"] = False

    # SET INITIAL VALUES ----------------------------
    # - initial NN parameters for both NNs [shuffle for each tracing param]
    NN_theta0_allocate = NN_allocate_orig.initialize_NN_parameters(initialize_scheme="glorot_bengio")
    NN_theta0_withdraw = NN_withdraw_orig.initialize_NN_parameters(initialize_scheme="glorot_bengio")
    
    # RESET PYTORCH NNs------------------------------- 
    # (random initialization independent from orig NNs by default)
    
    #put original NNs in list:
    NN_orig_list = [NN_withdraw_orig, NN_allocate_orig]
        
    # copy NN structures into pytorch NN
    NN_withdraw = class_NN_Pytorch.pytorch_NN(NN_withdraw_orig)
    NN_withdraw.to(device)
    # NN_withdraw.import_weights(NN_withdraw_orig, params)

    NN_allocate = class_NN_Pytorch.pytorch_NN(NN_allocate_orig)
    NN_allocate.to(device)
    # NN_allocate.import_weights(NN_allocate_orig, params)

    NN_list = torch.nn.ModuleList([NN_withdraw, NN_allocate])
  
    
     # - augment NN parameters with additional parameters to be solved
    # if params["obj_fun"] in ["mean_cvar_single_level"]:  # MEAN-CVAR only, augment initial value with initial xi
    #     if tracing_param == 1.0:
    
    xi_0 = 100.  
    params["xi_0"] = xi_0
    params["xi_lr"] = adam_xi_eta
    
    if params["xi_constant"]:
        params["xi_lr"] = 0.0
    
    # manual preload: load at every tracing param, but is overridden by continuation learn if cont is set to true
    if preload:
        
        if os.path.isdir(nn_preload):
            files = [f for f in os.listdir(nn_preload)]
            suffix_len = len(str(tracing_param))
            nn_preload_path = Path(nn_preload + [path for path in files if path[0:15] =='NN_opt_mc_decum' 
                               and re.split('kappa_', path)[-1] == str(tracing_param)][0])
            check = 0
        else:
            nn_preload_path = nn_preload
                        
            
        NN_list.load_state_dict(torch.load(nn_preload_path))
        NN_list.eval()
        print("pre-loaded NN: ", NN_list.state_dict())
        
        if os.path.isdir(nn_preload):
            files = [f for f in os.listdir(nn_preload)]
            suffix_len = len(str(tracing_param)+'.json')
            xi_preload_path = Path(nn_preload + [path for path in files if path[0:15] =='xi_opt_mc_decum' 
                               and re.split('kappa_', path)[-1] == str(tracing_param)+'.json'][0])
        else:
            xi_preload_path = xi_preload
            
        
        obj_text = codecs.open(xi_preload_path,'r', encoding='utf-8').read()
        b_new = json.loads(obj_text)
        print("loaded xi: ", b_new["xi"])
        params["xi_0"] = float(b_new["xi"])
    
    # load continuation learn model
    past_kappa = tracing_parameters_to_run[i-1]
    model_save_path = params["console_output_prefix"]
    nn_saved_model = Path(params["local_path"] + f"/saved_models/NN_opt_{model_save_path}_kappa_{past_kappa}")
    xi_saved = Path(params["local_path"] + f"/saved_models/xi_opt_{model_save_path}_kappa_{past_kappa}.json")
    # nn_saved_model = Path(f"/home/mmkshira/Documents/pytorch_decumulation_mc/researchcode/saved_models/NN_opt_{model_save_path}_kappa_{past_kappa}")
    # xi_saved = Path(f"/home/mmkshira/Documents/pytorch_decumulation_mc/researchcode/saved_models/xi_opt_{model_save_path}_kappa_{past_kappa}.json")

    if nn_saved_model.is_file():
        
        #NN
        if cont_nn and tracing_param not in tracing_parameters_to_run[0:cont_nn_start]:
            NN_list.load_state_dict(torch.load(nn_saved_model))
            NN_list.eval()
            print("loaded continuation NN: ", NN_list.state_dict())
            
        #xi
        if cont_xi and tracing_param not in tracing_parameters_to_run[0:cont_xi_start]:
            obj_text = codecs.open(xi_saved,'r', encoding='utf-8').read()
            b_new = json.loads(obj_text)
            params["xi_0"] = float(b_new["xi"])
            print("loaded xi: ", b_new["xi"])
                   
    
    # SET TRACING PARAMETERS inside params ----------------------------
    if params["obj_fun"] in ["one_sided_quadratic_target_error", "quad_target_error", "huber_loss", "ads"]:
        params["obj_fun_W_target"] = tracing_param  # all of them use Wealth target

    elif params["obj_fun"] in ["mean_cvar_single_level", "meancvarLIKE_constant_wstar"]:
        params["obj_fun_rho"] = tracing_param  # all of them use rho (scalarization parameter)


    #Now set tracing_params for STOCHASTIC BENCHMARK objectives:
    # this is for params["obj_fun"] in ["ads_stochastic", "qd_stochastic", "ir_stochastic", "te_stochastic"]
    elif params["obj_fun"] == "ads_stochastic": #ADS objective
        params["obj_fun_ads_beta"] = tracing_param  # annual target outperformance rate - see Ni, Li, Forsyth (2020)

    elif params["obj_fun"] == "qd_stochastic":
        params["obj_fun_qd_beta"] = tracing_param  #  beta >= 0 in exp(beta*T)

    elif params["obj_fun"] == "ir_stochastic": # INFORMATION RATIO using stochastic target
        params["obj_fun_ir_gamma"] = tracing_param  # objective function gamma (embedding parameter)

    elif params["obj_fun"] == "te_stochastic":  # TRACKING ERROR as in Forsyth (2021)
        if params["obj_fun_te_beta_hat_CONSTANT_TrueFalse"] is False:
            params["obj_fun_te_beta"] = tracing_param  # need to specify beta for beta_hat = exp(beta*t_n)
        else:  # if params["obj_fun_te_beta_hat_CONSTANT_TrueFalse"] is True
            params["obj_fun_te_beta_hat"] = tracing_param  # Need to specify beta_hat CONSTANT value >= 1


    # Do ONE-STAGE optimization for some objectives ----------------------------
    if params["obj_fun"] in ["mean_cvar_single_level",
                             "one_sided_quadratic_target_error",
                             "quad_target_error",
                             "huber_loss",
                             "ads",
                             "ads_stochastic",
                             "qd_stochastic",
                             "ir_stochastic",
                             "te_stochastic"]:

        params_TRAIN, params_CP_TRAIN, params_TEST, params_CP_TEST = \
            fun_RUN__wrapper.RUN__wrapper_ONE_stage_optimization(
                params=params,  # dictionary as setup in the main code
                NN_list=NN_list,  # list of both torch NNs
                NN_orig_list = NN_orig_list, #pieter NNs list 
                theta0=NN_theta0_allocate, #should not be used
                # initial parameter vector (weights and biases) + other parameters for objective function
                NN_training_options=NN_training_options,
                # dictionary with options to train NN, specifying algorithms and hyperparameters
                output_parameters = output_parameters  # Dictionary with output parameters as setup in main code
            )

    # Do TWO-STAGE optimization for some objectives ----------------------------
    # elif params["obj_fun"] in ["meancvarLIKE_constant_wstar"]:

    #     params_TRAIN, params_CP_TRAIN, params_TEST, params_CP_TEST = \
    #         fun_RUN__wrapper.RUN__wrapper_TWO_stage_optimization(
    #             params=params,  # dictionary as setup in the main code
    #             NN=NN,  # object of class_Neural_Network with structure as setup in main code
    #             theta0=theta0,
    #             # initial parameter vector (weights and biases) + other parameters for objective function
    #             NN_training_options=NN_training_options,
    #             # dictionary with options to train NN, specifying algorithms and hyperparameters
    #             output_parameters = output_parameters  # Dictionary with output parameters as setup in main code
    #         )

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



