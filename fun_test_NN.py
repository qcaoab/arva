

import pandas as pd
import numpy as np
import copy
import fun_Data__assign
import fun_eval_objfun_NN_strategy  #used for final objective function evaluation after training

def test_NN(F_theta,      # (optimal) parameter vector (weights and biases) + other parameters for objective function
             NN_object,      # object of class_Neural_Network with structure as setup in main code
             params         # dictionary with investment parameters as set up in main code
             ):

    #OUTPUTS
    # return params_TEST
    # - this code gives updated fields for the terminal wealth and objective function values in params_TEST

    #INPUTS:
    #     F_theta = Parameter vector theta of NN at which F_val is obtained. However:
    #                if params["obj_fun"] = "mean_cvar" F_theta is the *VECTOR* [NN_theta, xi, gamma]
    #                if params["obj_fun"] = "mean_cvar_single_level" F_theta is the *VECTOR* [NN_theta, xi]
    # NN_object = object of class_Neural_Network with structure as setup in main code
    # params = dictionary with investment parameters as set up in main code

    # TESTING only when params["test_TrueFalse"] == True
    if params["test_TrueFalse"] == False:
        raise ValueError("PVS error in 'test_NN': to test NN, requires params['test_TrueFalse'] == True.")

    elif params["test_TrueFalse"] == True:


        # Copy
        params_TEST = copy.deepcopy(params) #Create a copy

        #---------------------------------------------------------------------------
        # Make sure TESTING data is being used
        #       Set values of params_TEST["N_d"], params_TEST["Y"] and params_TEST["TradSig"] populated with train or test values
        train_test_Flag = "test"  # set train or test
        params_TEST = fun_Data__assign.set_training_testing_data(train_test_Flag, params_TEST)
        params_TEST["train_test_Flag"] = train_test_Flag

        # NOW done in fun_Data_assign:
        # if params["obj_fun"] in ["ads_stochastic", "qd_stochastic", "ir_stochastic", "te_stochastic"]:  #Make sure we copy the correct benchmark vector
        #     params_TEST["benchmark_W_T_vector"] = params["benchmark_W_T_vector_test"].copy()
        #     params_TEST["benchmark_W_paths"] = params["benchmark_W_paths_test"].copy()

        # ---------------------------------------------------------------------------
        # FINAL RESULT and do LRP if needed
        # Implement the F_theta trading strategy (NN parameters) and update the params_TEST dictionary
        #   Note: this invests the NN strategy, updates all the terminal wealth and objective function values in params_TEST


        # Also do layerwise-relevance propagation if required
        LRP_for_NN_TrueFalse = params_TEST["LRP_for_NN_TrueFalse"]
        PRP_TrueFalse = params_TEST["PRP_TrueFalse"]

        params_TEST, _, _, _ = fun_eval_objfun_NN_strategy.eval_obj_NN_strategy(F_theta= F_theta,
                                                                           NN_object=NN_object,
                                                                           params=params_TEST,
                                                                           output_Gradient=True,
                                                                           LRP_for_NN_TrueFalse = LRP_for_NN_TrueFalse,
                                                                            PRP_TrueFalse = PRP_TrueFalse)


    return params_TEST