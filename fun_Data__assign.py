import math
import pandas as pd
import numpy as np

#Objective: Wraps all the major TRAINING/TESTING data prep work, so main code is not so long

def set_training_testing_data(train_test_Flag, #"train" or "test"
                              params,    #dictionary as setup in the main code
                              called_from = None):
    # Set generic datasets Y and TradSig (could be training or testing dataset) in params
    # Also set generic "benchmark_W_T_vector" and "benchmark_W_paths" in params if needed
    # RETURNS: params with params["N_d"], params["Y"] and params["TradSig"] populated with train or test values

    if train_test_Flag == "train":

        params["N_d"] = params["N_d_train"]

        params["Y"] = params["Y_train"].copy()
        params["Y_order"] = params["Y_order_train"].copy()

        # Make sure we copy the correct benchmark vector if needed
        if called_from != "invest_ConstProp_strategy":  #Only makes sense for NN strategies
            if params["obj_fun"] in ["ads_stochastic",
                                     "qd_stochastic",
                                     "ir_stochastic",
                                     "te_stochastic"]:
                params["benchmark_W_T_vector"] = params["benchmark_W_T_vector_train"].copy()
                params["benchmark_W_paths"] = params["benchmark_W_paths_train"].copy()


        if params["use_trading_signals_TrueFalse"] == True:
            params["TradSig"] = params["TradSig_train"].copy()
            params["TradSig_order"] = params["TradSig_order_train"].copy()


    elif train_test_Flag == "test":
        params["N_d"] = params["N_d_test"]

        params["Y"] = params["Y_test"].copy()
        params["Y_order"] = params["Y_order_test"].copy()

        # Make sure we copy the correct benchmark vector if needed
        if called_from != "invest_ConstProp_strategy":
            if params["obj_fun"] in ["ads_stochastic", "qd_stochastic",
                                     "ir_stochastic", "te_stochastic"]:
                params["benchmark_W_T_vector"] = params["benchmark_W_T_vector_test"].copy()
                params["benchmark_W_paths"] = params["benchmark_W_paths_test"].copy()


        if params["use_trading_signals_TrueFalse"] == True:
            params["TradSig"] = params["TradSig_test"].copy()
            params["TradSig_order"] = params["TradSig_order_test"].copy()

    return params


