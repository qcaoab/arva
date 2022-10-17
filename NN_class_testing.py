# NN_portfolio_optim: Code version 08

# Portfolio optimization problem using NN to model control as in Li and Forsyth (2019)

run_on_my_computer = False   #True if running in my local desktop, False if running remotely
#-- this flag affects use of matplotlib.use('Agg') below


import pandas as pd
import numpy as np
import math
import os
import gc   #garbage collector
import seaborn as sns
import datetime
import pickle
import sys
import datetime
import codecs, json

#Import files needed (other files are imported within those files as needed)
import fun_Data_timeseries_basket
import fun_Data__bootstrap_wrapper
import fun_Data__MCsim_wrapper
import class_Neural_Network
import fun_train_NN_algorithm_defaults
import fun_RUN__wrapper


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

now = datetime.datetime.now()
print ("Starting at: ")
print (now.strftime("%Y-%m-%d %H:%M:%S"))
#-----------------------------------------------------------------------------------------------
# Portfolio problem: Main structural parameters
#-----------------------------------------------------------------------------------------------
params = {} #Initialize empty dictionary

abspath = os.path.abspath(__file__)
dname = os.path.dirname(abspath)
os.chdir(dname)
code_title_prefix = "output/output_mc_forsyth_rep"    #used for saving output on local


params["T"] = 5. #Investment time horizon, in years
params["N_rb"] = 20  #Nr of equally-spaced rebalancing events in [0,T]
          #Cash injections AND rebalancing at times t_n = (n-1)*(T/N_rb), for n=1,...,N_rb

params["delta_t"] = params["T"] / params["N_rb"]    # Rebalancing time interval
params["W0"] = 1000.     # Initial wealth W0
params["q"] =  0. * np.ones(params["N_rb"])  # Cash injection schedule (a priori specified)

seed_mc = 1
np.random.seed(seed_mc)
print("\n Random seed: ", seed_mc, " \n")

#Specify TRANSACTION COSTS parameters
params["TransCosts_TrueFalse"] = False #If True, incorporate transaction costs
# - if TransCosts_TrueFalse == True, additional parameters will be used
if params["TransCosts_TrueFalse"] is True:
    params["TransCosts_r_b"] = 5/100 #(Real), cont comp. interest rate on borrowed transaction costs
    params["TransCosts_propcost"] = 0.5/100   #proportional TC in (0,1] of trading in any asset EXCEPT cash account
    params["TransCosts_lambda"] = 1e-6  #lambda>0 parameter for smooth quadratic approx to abs. value function


iter_params = "real_exp"

if iter_params == "real_exp":
    n_d_train_mc = int(2.56* (10**6))
    itbound_mc = 30000
    batchsize_mc = 1000

if iter_params == "test_run":
    n_d_train_mc = int(2.56* (10**4))
    itbound_mc = 1000
    batchsize_mc = 200

if iter_params == "tiny":
    n_d_train_mc = 100
    itbound_mc = 2
    batchsize_mc = 2
    params["N_rb"] = 2
    params["q"] =  0. * np.ones(params["N_rb"]) 


continuation_learn = False  #MC added: if True, will use weights from previous tracing parameter to initialize theta0. 


#Main settings for TRAINING data
params["N_d_train"] = n_d_train_mc #Nr of TRAINING data return sample paths to bootstrap
params["data_source_Train"] = "simulated" #"bootstrap" or "simulated" [data source for TRAINING data]


#Specify if NN has been pre-trained: if FALSE, will TRAIN the NN
params["preTrained_TrueFalse"] = False  #If True, NO TRAINING will occur, instead given F_theta will be used
if params["preTrained_TrueFalse"] is True:
    #Specify the F_theta vector
    # IMPORTANT: Every other input must correspond exactly to the setup used to get this F_theta
    preTrained_F_theta_list = []
    params["preTrained_F_theta"] = np.array(preTrained_F_theta_list)
    #     params["F_theta"] = F_theta #Parameter vector theta of NN at which F_val is obtained
    #                if params["obj_fun"] = "mean_cvar" this is the *VECTOR* [NN_theta, xi, gamma]
    #                if params["obj_fun"] = "mean_cvar_single_level" this is the *VECTOR* [NN_theta, xi]

#Main settings for TESTING data
params["test_TrueFalse"] = False #TRUE if training AND testing, FALSE if *just training*

if params["test_TrueFalse"] is True:
    params["N_d_test"] = int((10**5))  # (only used when test_TrueFalse == True) Nr of TESTING data sample paths to bootstrap
    params["data_source_Test"] = "bootstrap" #"bootstrap" or "simulated" [if test_TrueFalse == True, data source for TESTING data]


#--------------------------------------
# ASSET BASKET: Specify basket of candidate assets, and REAL or NOMINAL data
params["asset_basket_id"] = "basic_T30_VWD"     #"basic_ForsythLi"    #Pre-defined basket of underlying candidate assets - see fun_Data_assets_basket.py
params["add_cash_TrueFalse"] = False     #If True, add "Cash" as an asset to the selected asset basket
    # - We will ALWAYS set add_cash_TrueFalse = True if TransCosts_TrueFalse == True (below)
params["real_or_nominal"] = "real" # "real" or "nominal" for asset data for wealth process: if "real", the asset data will be deflated by CPI
#   Note: real or nominal for TRADING SIGNALS will be set separately below


#--------------------------------------
# CONFIDENCE PENALTY: Entropy-based confidence penalty on the outputs of the NN
# --> Based on the paper by PereyraEtAl 2017
# Specify hyperparameters
params["ConfPenalty_TrueFalse"] = False   #If True, apply confidence penalty
params["ConfPenalty_lambda"] = 0.0 # weight (>0) on confidence penalty term; if == 0, then NO confidence penalty is applied
params["ConfPenalty_n_H"] = 4   # integer in {1,...,N_a}, where N_a is number of *noncash* assets in params["asset_basket_id"]
                                # only large (confident/undiversified) investments in assets {ConfPenalty_n_H,...,N_a}
                                # will be penalized, *NOT* the other assets.
                                # Generates runtime error if ConfPenalty_n_H > N_a

if params["add_cash_TrueFalse"] is True:
    #Add one since cash will be *inserted* as the first asset
    params["ConfPenalty_n_H"] = params["ConfPenalty_n_H"] + 1

#--------------------------------------
# TRADING SIGNALS:
params["use_trading_signals_TrueFalse"] = False    #If TRUE, will use trading signals in feature vector
                                        # if FALSE, will use only default features (e.g. time to go, wealth)
if params["use_trading_signals_TrueFalse"] is True:
    # Trading signal SETTINGS:
    params["trading_signal_basket_id"] = "All_MA_RSTD"    #Pre-defined basket of trading signals
    params["trading_signal_underlying_asset_basket_id"] = "VWD"    #UNDERLYING asset basket for trading signals
    params["trading_signal_real_or_nominal"] = "nominal" # "real" or "nominal": if "real", the asset data will be deflated by CPI

# -----------------------------------------------------------------------------------------------
#  OBJECTIVE FUNCTION:  CHOICE AND PARAMETERS
# -----------------------------------------------------------------------------------------------
params["obj_fun"] = "mean_cvar_single_level"

# STANDARD objective functions of W(T): obj_fun options include:
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

# print("tracing parameter entered from terminal: ", sys.argv[1])
# tracing_parameters_to_run = [0.1, 0.25, 0.4, 0.6, 0.7, 0.8, 0.9, 1.0, 1.2, 1.5, 2.0, 3.0, 10.0]

tracing_parameters_to_run = [0.1, 0.25, 0.4, 0.6, 0.7, 0.8, 0.9, 1.0] + np.around(np.arange(1.1, 3.1, 0.1),1).tolist() + [10.0]

#[float(item) for item in sys.argv[1].split(" ")] #Must be LIST

#[1.0, 1.5, 3.0, 10.0]
# tracing_parameters_to_run = [float(item) for item in sys.argv[1].split(" ")] #Must be LIST



# TRACING PARAMETERS interpreted as follows:
#   -> STANDARD objectives tracing parameters:
#   params["obj_fun_rho"] if obj_fun is in ["mean_cvar_single_level"], *larger* rho means we are looking for a *higher* mean
#   params["obj_fun_W_target"] if obj_fun is in ["one_sided_quadratic_target_error", "quad_target_error", "huber_loss", "ads"]

#   -> STOCHASTIC BENCHMARK objectives tracing parameters:
# "ads_stochastic": params["obj_fun_ads_beta"]>=0 which is the annual target outperformance rate, in exp(beta*T)
# "qd_stochastic": params["obj_fun_qd_beta"] >= 0, this is beta in exp(beta*T)
# "ir_stochastic": [Information ratio] then this is the embedding parameter gamma >0
# "te_stochastic": [Tracking error] this is "beta" >=0 or "beta_hat" >=1:
#       if params["obj_fun_te_beta_hat_CONSTANT_TrueFalse"] = True, this is beta_hat
#       if params["obj_fun_te_beta_hat_CONSTANT_TrueFalse"] = False, this is beta in exp(beta*t_n)


# SGD max iterations and batch size
itbound = itbound_mc #64000    #Mean-CVAR: use at least itbound = 50k
batchsize = batchsize_mc  #100      #Mean-CVAR: use at least batchsize = 1000, other can use = 100

# Set objective function parameters [rarely changed]
# -----------------------------------------------------------------------------
if params["obj_fun"] == "one_sided_quadratic_target_error":
    params["obj_fun_eps"] = 1e-06  # small regularization parameter used for one sided quadratic objective
    params["obj_fun_cashwithdrawal_TrueFalse"] = True  # set to True if we want to report results *after* withdrawal of cash
#-----------------------------------------------------------------------------
elif params["obj_fun"] == "quad_target_error":
    print("No extra obj_fun parameters required.")
# -----------------------------------------------------------------------------
elif params["obj_fun"] == "huber_loss":
    params["obj_fun_huber_delta"] = 100.
# -----------------------------------------------------------------------------
elif params["obj_fun"] == "ads":    #Uses constant target
    params["obj_fun_lambda_smooth"] = 1e-06

# -----------------------------------------------------------------------------
elif params["obj_fun"] == "mean_cvar_single_level":  # SINGLE level formulation, no Lagrangian
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

#STOCHASTIC BENCHMARK objectives:
# -----------------------------------------------------------------------------
elif params["obj_fun"] == "ads_stochastic":  # Uses stochastic target as in Ni, Li, Forsyth (2020)
    params["obj_fun_lambda_smooth"] = 1.00  # 1e-06

# -----------------------------------------------------------------------------
elif params["obj_fun"] == "qd_stochastic":  # QUADRATIC DEVIATION from elevated target as in Forsyth (2021)
    print("No extra obj_fun parameters required.")

# -----------------------------------------------------------------------------
elif params["obj_fun"] == "ir_stochastic":  # INFORMATION RATIO using stochastic target as in Goetzmann et al (2002)
    params["obj_fun_ir_s"] = 0.00  # annual target outperformance rate - see Ni, Li, Forsyth (2020)
    #BUT for standard IR, let's just set this to zero always!!

# -----------------------------------------------------------------------------
elif params["obj_fun"] == "te_stochastic":  # TRACKING ERROR as in Forsyth (2021)
    params["obj_fun_te_beta_hat_CONSTANT_TrueFalse"] = False #See "if statement" below how this is used
    #       if params["obj_fun_te_beta_hat_CONSTANT_TrueFalse"] = True, tracing_param is beta_hat
    #       if params["obj_fun_te_beta_hat_CONSTANT_TrueFalse"] = False, tracing_param is beta in beta_hat = exp(beta*t_n)



#-----------------------------------------------------------------------------------------------
# Output flags
#-----------------------------------------------------------------------------------------------

output_parameters = {}

#Basic output params
output_parameters["code_title_prefix"] = code_title_prefix + "_kappa_" + str(tracing_parameters_to_run) # used as prefix for naming files when saving outputs
output_parameters["output_results_Excel"] = True      #Output results summary to Excel

output_parameters["save_Figures_format"] = "png"  # if saving figs, format to save figures in, e.g. "png", "eps",

# W_T, pdf and cdf
output_parameters["output_W_T_vectors_Excel"] = False       #output the terminal wealth vectors (train, test and benchmark) for histogram construction
output_parameters["output_W_T_histogram_and_cdf"] = True        #output Excel sheet with numerical histogram and CDF of terminal wealth (train, test and benchmark)
output_parameters["output_W_T_histogram_and_cdf_W_max"] = 1000.   # Maximum W_T value for histogram and CDF.
output_parameters["output_W_T_histogram_and_cdf_W_bin_width"] = 5.   # Bin width of wealth for histogram and CDF.

#Output benchmark stats
output_parameters["output_W_T_benchmark_comparisons"] = False #If true, outputs W_T vs Benchmark differences and ratios


# Roling historical test
output_parameters["output_Rolling_Historical_Data_test"] = False  #If true, test NN strategy and benchmark strategy on actual single
                                    # historical data path, starting in each month and investing for the duration
output_parameters["fixed_yyyymm_list"] = [198001, 198912, 199001] #Used for historical rolling test; LIST of yyyymm_start months of particular interest
output_parameters["output_Rolling_Historical_only_for_fixed"] = False  #if True, outputs rolling historical ONLY for output_parameters["fixed_yyyymm_list"]

# NN detail
output_parameters["output_TrainingData_NNweights_test"] = False   #if true, outputs the training paths + features + NN weights required to
                                            # reproduce the top and bottom k results for the terminal wealth and objective function values

# PERCENTILES:
output_parameters["output_Pctiles_Excel"] = True #If True, outputs Excel spreadsheet with NN pctiles of proportions in each asset and wealth over time
output_parameters["output_Pctiles_Plots"] = False #if True, plots the paths of output_Pctiles_Excel over time
output_parameters["output_Pctiles_Plots_W_max"] = 1500. #Maximum y-axis value for WEALTH percentile plots
output_parameters["output_Pctiles_list"] = [5,50,95]  #Only used if output_Pctiles_Excel or output_Pctiles_Plots is True, must be list, e.g.  [20,50,80]
output_parameters["output_Pctiles_on_TEST_data"] = False #Output percentiles for test data as well

#Control heatmap params [used if save_Figures == True]
output_parameters["save_Figures_FunctionHeatmaps"] = True #if True, plot and save the function heatmap figures
output_parameters["output_FunctionHeatmaps_Excel"] = False  # If TRUE, output the heatmap grid data to Excel

output_parameters["save_Figures_DataHeatmaps"] = True   #if True, plot and save the data heatmap figures
output_parameters["output_DataHeatmaps_Excel"] = False   # If TRUE, output the heatmap grid data to Excel

output_parameters["heatmap_y_bin_min"] = 0.  # minimum for the y-axis grid of heatmap
output_parameters["heatmap_y_bin_max"] = 2000.0  # maximum for the y-axis grid of heatmap
output_parameters["heatmap_y_num_pts"] = int(output_parameters["heatmap_y_bin_max"] - output_parameters["heatmap_y_bin_min"])+1  # number of points for y-axis
output_parameters["heatmap_xticklabels"] = 2  # e.g. xticklabels=6 means we are displaying only every 6th xaxis label to avoid overlapping
output_parameters["heatmap_yticklabels"] = 25  # e.g. yticklabels = 10 means we are displaying every 10th label
output_parameters["heatmap_cmap"] = "rainbow"  # e.g. "Reds" or "rainbow" etc colormap for sns.heatmap
output_parameters["heatmap_cbar_limits"] = [0.0, 1.0]  # list in format [vmin, vmax] for heatmap colorbar/scale


#XAI
output_parameters["output_PRPscores"] = False    #If True, outputs PRP score analysis (heatmaps, percentiles)
output_parameters["PRPheatmap_xticklabels"] = 1  # e.g. xticklabels=6 means we are displaying only every 6th xaxis label to avoid overlapping
output_parameters["PRPheatmap_yticklabels"] = 1  # e.g. yticklabels = 500 means we are displaying every 500th label



#-----------------------------------------------------------------------------------------------
# Asset basket and Feature specification (also specify basket of trading signals, if applicable)
#-----------------------------------------------------------------------------------------------

#If modelling TCs, make sure we have incorporated the account:
if params["TransCosts_TrueFalse"] is True:
    params["add_cash_TrueFalse"] = True


#Construct asset basket:
# - this will also give COLUMN NAMES in the historical returns data to use
params["asset_basket"] = fun_Data_timeseries_basket.timeseries_basket_construct(
                            basket_type="asset",
                            basket_id=params["asset_basket_id"],
                            add_cash_TrueFalse=params["add_cash_TrueFalse"],
                            real_or_nominal = params["real_or_nominal"] )

#Assign number of assets based on basket information:
params["N_a"] = len(params["asset_basket"]["basket_columns"])   #Nr of assets = nr of output nodes

#Check application of confidence penalty "ConfPenalty_n_H" is within the number of asset (N_a) bounds
if params["ConfPenalty_TrueFalse"] is True:
    if params["ConfPenalty_n_H"] > params["N_a"]:
        raise ValueError("PVS error: params['ConfPenalty_n_H'] > params['N_a'], need to select new params['ConfPenalty_n_H'].")

#Initialize number of input nodes
params["N_phi"] =  2  #Nr of default features, i.e. the number of input nodes
params["feature_order"] = ["time_to_go", "stdized_wealth"]  #initialize order of the features

if params["obj_fun"] in ["ads_stochastic", "qd_stochastic", "ir_stochastic", "te_stochastic"]:
    params["N_phi"] = 3  # Nr of default features, i.e. the number of input nodes
    params["feature_order"].append("stdized_benchmark_wealth") # initialize order of the features


params["N_phi_standard"] = params["N_phi"] # Set nr of basic features *BEFORE* we add trading signals
if params["use_trading_signals_TrueFalse"] is True:

    #Construct trading signal basket:
    # - this will also give COLUMN NAMES in the historical returns data to use
    params["trading_signal_basket"] = fun_Data_timeseries_basket.timeseries_basket_construct(
                                basket_type="trading_signal",
                                basket_id=params["trading_signal_basket_id"],
                                real_or_nominal = params["trading_signal_real_or_nominal"],
                                underlying_asset_basket_id = params["trading_signal_underlying_asset_basket_id"])

    #Adjust number of features (input nodes to reflect trading signals
    params["N_phi"] = params["N_phi"] + len(params["trading_signal_basket"]["basket_columns"])  #Nr of input nodes
    params["feature_order"].extend(params["trading_signal_basket"]["basket_timeseries_names"])  #Add feature names




#-----------------------------------------------------------------------------------------------
# NEURAL NETWORK (NN) SETUP
#-----------------------------------------------------------------------------------------------
params["N_L"] = 4 # Nr of hidden layers of NN
                   # NN will have total layers 1 (input) + N_L (hidden) + 1 (output) = N_L + 2 layers in total
                   # layer_id list: [0, 1,...,N_L, N_L+1]


NN = class_Neural_Network.Neural_Network(n_nodes_input = params["N_phi"],
                                         n_nodes_output = params["N_a"],
                                         n_layers_hidden = params["N_L"])
NN.print_layers_info()  #Check what to update

#Update layers info
NN.update_layer_info(layer_id = 1 , n_nodes = params["N_a"] + 2 , activation = "logistic_sigmoid", add_bias=False)
NN.update_layer_info(layer_id = 2 , n_nodes = params["N_a"] + 2, activation = "logistic_sigmoid", add_bias=False)
NN.update_layer_info(layer_id = 3 , n_nodes = params["N_a"] + 2, activation = "logistic_sigmoid", add_bias=False)
NN.update_layer_info(layer_id = 4 , n_nodes = params["N_a"] + 2, activation = "logistic_sigmoid", add_bias=False)
#NN.update_layer_info(layer_id = 3 , n_nodes = 8, activation = "ELU", add_bias=False)
NN.update_layer_info(layer_id = 5, activation = "softmax", add_bias= False)

NN.print_layers_info() #Check if structure is correct


# L2 weight regularization (only weights, not biases)
# params["lambda_reg"] = 0.0 #1e-08 #1e-07    #Set to zero for no weight regularization

original_NN = NN

n_nodes_input_orig = original_NN.n_nodes_input  #nr of input nodes = size of feature vector
n_nodes_output_orig = original_NN.n_nodes_output    #nr of output nodes
n_layers_hidden_orig = original_NN.L   # nr of hidden layers
n_layers_total_orig = original_NN.n_layers_total
        
        
for l in np.arange(0, original_NN.n_layers_total, 1):
            orig_dict = {"obj.layers[layer_id]" : "obj.layers[" + str(l) + "]",
                          "layer_id" : original_NN.layers[l].layer_id,
                          "description": original_NN.layers[l].description,
                          "n_nodes" : original_NN.layers[l].n_nodes,
                          "activation":  original_NN.layers[l].activation,
                          "x_l(weights)":  [original_NN.layers[l].x_l_shape],
                          "add_bias" : original_NN.layers[l].add_bias,
                          "b_l(biases)" : original_NN.layers[l].b_l_length}

            if l == 0 :
                nn_orig_df = pd.DataFrame.from_dict(orig_dict)

            else:
                nn_orig_df = pd.concat([nn_orig_df,pd.DataFrame.from_dict(orig_dict)])

nn_orig_df.reset_index()

# node_n_list = []
# activation_list = []
# for index, row in nn_orig_df.iterrows():
#     node_n_list.append(row['n_nodes'])
#     activation_list.append(row["description"])

# create pytorch layers
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from collections import OrderedDict


import class_NN_Pytorch

py_NN = class_NN_Pytorch.pytorch_NN(original_NN)


# init theta

# SET INITIAL VALUES ----------------------------
# - initial NN parameters [shuffle for each tracing param]
NN_theta0 = original_NN.initialize_NN_parameters(initialize_scheme="glorot_bengio")
theta0 = NN_theta0.copy()

if params["obj_fun"] in ["mean_cvar_single_level"]:  # MEAN-CVAR only, augment initial value with initial xi
        xi_0 = 27.744101568234655
        theta0 = np.concatenate([NN_theta0, [xi_0]])

# original_NN.theta = NN_theta0

# layer0 = original_NN.layers[0]

# layer0.x_l()

# original_NN.initialize_NN_parameters()


# original_NN = original_NN.unpack_NN_parameters()

# py_NN.model[1]

# py_NN.model[0].weight.data = torch.Tensor([[ 0.1317, -0.2195],
#         [ 0.2278, -0.7777],
#         [ 0.6114,  0.4792],
#         [ 0.0625,  0.6756]])


# py_NN.model[0].weight.data.numel()
# # print(original_NN.layers[0].x_l)