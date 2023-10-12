

import pandas as pd
import numpy as np
import copy
import fun_Data__assign
from fun_train_NN_SGD_algorithms import run_Gradient_Descent_pytorch    #SGD algorithms
import fun_invest_NN_strategy
import fun_eval_objfun_NN_strategy  #used for final objective function evaluation after training
import fun_invest_NN_strategy
import torch


def train_NN(NN_list,      # object of class_Neural_Network with structure as setup in main code
             params,         # dictionary with investment parameters as set up in main code
             NN_training_options  #dictionary with options to train NN, specifying algorithms and hyperparameters
             ):

    #Objective: find parameters of NN_object (NN_theta) that minimize the objective function as specified in params
    #return params, res_BEST, res_ALL, res_ALL_dataframe

    # OUTPUTS:
    # params = dictionary as set up in the main code, but
    #       all fields associated with terminal wealth, NN parameters, objective function values UPDATED
    #       to reflect the BEST training result, as well as
    #       params["res_BEST"] = res_BEST
    # res_BEST = dictionary of results for the **best-performing** method in NN_training_options["methods"]
    #       achieving the LOWEST objective function value
    # res_ALL = a dictionary of dictionaries with all results,
    #           i.e. a dictionary for every method in NN_training_options["methods"]
    # res_ALL_summary_df = summary of all the results in pandas.DataFrame
    #                       constructed by appending the individual res["summary_df"] in its rows


    #Check that params uses TRAINING data, and make a copy
    params = copy.deepcopy(params)  # Create a copy

    # ---------------------------------------------------------------------------
    # Make sure TRAINING data is being used
    # Set values of params["N_d"], params["Y"] and params["TradSig"] populated with train or test values
    train_test_Flag = "train"  # set train or test
    params = fun_Data__assign.set_training_testing_data(train_test_Flag, params)
    params["train_test_Flag"] = train_test_Flag

    
    #Add theta0 and NN_training_options for reference
    # params["theta0"] = theta0 
    params["NN_training_options"] = NN_training_options

    #Dictionary of dictionaries with results
    res_ALL = {}    #initialize
    res_BEST = {}

    # The only optimization algorithm implemented here is Adaptive Momentum (Adam) with stochastic gradient descent. Additional options for optimization algorithms can be included here. 
   

    if "Adam" in NN_training_options["methods"]:
        
        # print("Running pytorch SGD gradient descent.")
        result_pyt_adam = run_Gradient_Descent_pytorch(NN_list= NN_list,
                                                       params = params, 
                                                       NN_training_options = NN_training_options)
        
        res_ALL["pytorch_adam"] = result_pyt_adam
        params["NN_object"] = NN_list
        
    

    # CONSTRUCT OUTPUTS: ---------------------------------------------------------------------------
    val_min = np.inf  # initialize running minimum
    res_ALL_dataframe = pd.DataFrame() #initialize

    #Loop through res_ALL.keys() [i.e. loop through all methods run] to construct outputs
    # for key in res_ALL.keys():

    
    
    # commented out for temporary output
    #     #Append
    #     res_ALL_dataframe = res_ALL_dataframe.append(res_ALL[key]["summary_df"], ignore_index=True)

    #     # Select result from which achieves lowest overall objective function value
    #     if res_ALL[key]["val"] < val_min:
    #         val_min = res_ALL[key]["val"]  # set new running min for objective function value
    #         res_BEST = res_ALL[key]  # res_BEST contains the results for the new running min


    # #Finally append res_BEST to the bottom of res_ALL_dataframe
    # if params["preTrained_TrueFalse"] is False: #If we actually did training as above
    #     res_BEST_temp = res_BEST #create temp copy to indicate that it has been selected
    #     res_BEST_temp["summary_df"]["method"] = "SELECTED: " + res_BEST_temp["method"]
    #     res_ALL_dataframe = res_ALL_dataframe.append(res_BEST_temp["summary_df"], ignore_index=True)

    # elif params["preTrained_TrueFalse"] is True: #Otherwise just copy provided F_theta across if provided
    #     res_BEST.update({"F_theta": params["preTrained_F_theta"]})


    # # print("------------------------------------------------------")
    # # print("Contents of res_BEST:")
    # # print(res_BEST)
    # #   res_BEST = dictionary of results for the **best-performing** method in NN_training_options["methods"]
    # #               achieving the LOWEST objective function value

    # #---------------------------------------------------------------------------
    # # FINAL RESULT and do LRP/PRP if needed
    # #Implement the res_BEST trading strategy (NN parameters) and update the params dictionary
    # params["res_BEST"] = res_BEST

    # #Also do LRP/PRP if required
    # LRP_for_NN_TrueFalse = params["LRP_for_NN_TrueFalse"]
    # PRP_TrueFalse = params["PRP_TrueFalse"]

    # #   Note: this invests the NN strategy, updates all the terminal wealth and objective function values in params
    # params, _, _, _ = fun_eval_objfun_NN_strategy.eval_obj_NN_strategy(F_theta = res_BEST["F_theta"],
    #                                                                    NN_object = NN_object,
    #                                                                    params = params,
    #                                                                    output_Gradient = True,
    #                                                                    LRP_for_NN_TrueFalse = LRP_for_NN_TrueFalse,
    #                                                                    PRP_TrueFalse = PRP_TrueFalse)

    # # Update  params["res_BEST"]["NN_theta"] for subsequent use
    # if params["preTrained_TrueFalse"] is True:
    #     params["res_BEST"].update({"NN_theta": params["NN_theta"]})
    with torch.no_grad():
        if params["nn_withdraw"]:  #decumulation
            params, _, qsum_T_vector = fun_invest_NN_strategy.withdraw_invest_NN_strategy(NN_list, params)
        else: #NO decumulation
            params, W_T_vector = fun_invest_NN_strategy.invest_NN_strategy_pyt(NN_list, params)
            
        params["NN_object"] = NN_list

    return params, res_ALL['pytorch_adam']