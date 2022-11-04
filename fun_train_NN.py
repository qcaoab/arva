

import pandas as pd
import numpy as np
import copy
import fun_Data__assign
from fun_train_NN_scipy_algorithms import run_scipy_minimize    #scipy minimization algorithms
from fun_train_NN_SGD_algorithms import run_Gradient_Descent, run_Gradient_Descent_pytorch    #SGD algorithms
import fun_eval_objfun_NN_strategy  #used for final objective function evaluation after training

def train_NN(theta0,      # initial parameter vector (weights and biases) + other parameters for objective function
             NN_list,      # object of class_Neural_Network with structure as setup in main code
             NN_orig_list,
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

    # Now done in fun_Data_assign:
    # if params["obj_fun"] in ["ads_stochastic", "qd_stochastic", "ir_stochastic", "te_stochastic"]:  # Make sure we copy the correct benchmark vector
    #     params["benchmark_W_T_vector"] = params["benchmark_W_T_vector_train"].copy()
    #     params["benchmark_W_paths"] = params["benchmark_W_paths_train"].copy()

    #Add theta0 and NN_training_options for reference
    params["theta0"] = theta0
    params["NN_training_options"] = NN_training_options

    #Dictionary of dictionaries with results
    res_ALL = {}    #initialize
    res_BEST = {}

    # CG (in scipy algorithms): --------------------------------------------------------------------------------
    # if "CG" in NN_training_options["methods"]:
    #     # method = "CG": uses a nonlinear conjugate gradient algorithm by Polak and Ribiere,
    #     #               a variant of the Fletcher-Reeves method
    #     #               Needs first derivatives only

    #     print("Running CG.")
    #     res_CG = run_scipy_minimize(method="CG",
    #                                 theta0 = theta0,
    #                                 NN_object = NN_object,
    #                                 params = params,
    #                                 itbound = NN_training_options["itbound_scipy_algorithms"],
    #                                 tol =  NN_training_options["tol"],
    #                                 output_progress = NN_training_options["output_progress"]
    #                                 )
    #     res_ALL["res_CG"] = res_CG  #append to res_ALL
    #     print(res_CG)

    # # BFGS (in scipy algorithms): --------------------------------------------------------------------------------
    # if "BFGS" in NN_training_options["methods"]:
    #     # method='BFGS': Broyden-Fletcher-Goldfarb-Shanno algorithm, quasi-Newton method,
    #     #               Needs first derivatives only

    #     print("Running BFGS.")
    #     res_BFGS = run_scipy_minimize(method="BFGS",
    #                                   theta0=theta0,
    #                                   NN_object=NN_object,
    #                                   params=params,
    #                                   itbound=NN_training_options["itbound_scipy_algorithms"],
    #                                   tol=NN_training_options["tol"],
    #                                   output_progress=NN_training_options["output_progress"]
    #                                   )
    #     res_ALL["res_BFGS"] = res_BFGS  # append to res_ALL
    #     print(res_BFGS)

    # # Newton-CG (in scipy algorithms): --------------------------------------------------------------------------------
    # if "Newton-CG" in NN_training_options["methods"]:
    #     # method='Newton-CG': Newton-Conjugate-Gradient algorithm,
    #     #                       uses a CG method to the compute the search direction
    #     #                   Good for big problems, no explicit Hessian inversion/factorization
    #     print("Running Newton-CG.")

    #     res_NewtonCG = run_scipy_minimize(method="Newton-CG",
    #                                       theta0=theta0,
    #                                       NN_object=NN_object,
    #                                       params=params,
    #                                       itbound=NN_training_options["itbound_scipy_algorithms"],
    #                                       tol=NN_training_options["tol"],
    #                                       output_progress=NN_training_options["output_progress"]
    #                                       )
    #     res_ALL["res_NewtonCG"] = res_NewtonCG  # append to res_ALL
    #     print(res_NewtonCG)



    # SGD_constant (in SGD algorithms): ---------------------------------------------------------------------------

    # if "SGD_constant" in NN_training_options["methods"]:
    #     print("Running SGD_constant.")
    #     res_SGD_constant = run_Gradient_Descent(method="SGD_constant",
    #                                    theta0 = theta0,
    #                                    NN_object = NN_object,
    #                                    params = params,
    #                                    itbound = NN_training_options["itbound_SGD_algorithms"],
    #                                    batchsize = NN_training_options["batchsize"],
    #                                    check_exit_criteria = NN_training_options["check_exit_criteria"],
    #                                    output_progress = NN_training_options["output_progress"],
    #                                    nit_running_min = NN_training_options["nit_running_min"],
    #                                    nit_IterateAveragingStart = NN_training_options["nit_IterateAveragingStart"],
    #                                    SGD_learningrate = NN_training_options["SGD_learningrate"]
    #                                    )

    #     res_ALL["res_SGD_constant"] = res_SGD_constant    #append to res_ALL
    #     #print(res_SGD_constant)

    # # Adagrad (in SGD algorithms):: ---------------------------------------------------------------------------
    # if "Adagrad" in NN_training_options["methods"]:
    #     print("Running Adagrad.")
    #     res_Adagrad = run_Gradient_Descent(method="Adagrad",
    #                                        theta0=theta0,
    #                                        NN_object=NN_object,
    #                                        params=params,
    #                                        itbound=NN_training_options["itbound_SGD_algorithms"],
    #                                        batchsize=NN_training_options["batchsize"],
    #                                        check_exit_criteria=NN_training_options["check_exit_criteria"],
    #                                        output_progress=NN_training_options["output_progress"],
    #                                        nit_running_min=NN_training_options["nit_running_min"],
    #                                        nit_IterateAveragingStart=NN_training_options["nit_IterateAveragingStart"],
    #                                        Adagrad_epsilon= NN_training_options["Adagrad_epsilon"],
    #                                        Adagrad_eta= NN_training_options["Adagrad_eta"]
    #                                        )

    #     res_ALL["res_Adagrad"] = res_Adagrad    #append to res_ALL
    #     #print(res_Adagrad)

    # # Adadelta (in SGD algorithms):: ---------------------------------------------------------------------------
    # if "Adadelta" in NN_training_options["methods"]:
    #     print("Running Adadelta.")
    #     res_Adadelta = run_Gradient_Descent(method="Adadelta",
    #                                         theta0=theta0,
    #                                         NN_object=NN_object,
    #                                         params=params,
    #                                         itbound=NN_training_options["itbound_SGD_algorithms"],
    #                                         batchsize=NN_training_options["batchsize"],
    #                                         check_exit_criteria=NN_training_options["check_exit_criteria"],
    #                                         output_progress=NN_training_options["output_progress"],
    #                                         nit_running_min=NN_training_options["nit_running_min"],
    #                                         nit_IterateAveragingStart=NN_training_options["nit_IterateAveragingStart"],
    #                                         Adadelta_epsilon= NN_training_options["Adadelta_epsilon"],
    #                                         Adadelta_ewma= NN_training_options["Adadelta_ewma"]
    #                                         )

    #     res_ALL["res_Adadelta"] = res_Adadelta  # append to res_ALL
    #     #print(res_Adadelta)

    # # RMSprop (in SGD algorithms):: ---------------------------------------------------------------------------
    # if "RMSprop" in NN_training_options["methods"]:
    #     print("Running RMSprop.")
    #     res_RMSprop = run_Gradient_Descent(method="RMSprop",
    #                                        theta0=theta0,
    #                                        NN_object=NN_object,
    #                                        params = params,
    #                                        itbound = NN_training_options["itbound_SGD_algorithms"],
    #                                        batchsize = NN_training_options["batchsize"],
    #                                        check_exit_criteria = NN_training_options["check_exit_criteria"],
    #                                        output_progress = NN_training_options["output_progress"],
    #                                        nit_running_min = NN_training_options["nit_running_min"],
    #                                        nit_IterateAveragingStart=NN_training_options["nit_IterateAveragingStart"],
    #                                        RMSprop_epsilon = NN_training_options["RMSprop_epsilon"],
    #                                        RMSprop_ewma = NN_training_options["RMSprop_ewma"],
    #                                        RMSprop_eta = NN_training_options["RMSprop_eta"]
    #                                        )
    #     res_ALL["res_RMSprop"] = res_RMSprop  # append to res_ALL
    #     #print(res_RMSprop)


    if NN_training_options["pytorch"] and "Adam" in NN_training_options["methods"]:
        
        # print("Running pytorch SGD gradient descent.")
        result_pyt_adam = run_Gradient_Descent_pytorch(NN_list= NN_list,
                                                       NN_orig_list = NN_orig_list, #pieter NNs list
                                                       params = params, 
                                                       NN_training_options = NN_training_options)
        
        res_ALL["pytorch_adam"] = result_pyt_adam
        
    
    # # Adam (in SGD algorithms):: ---------------------------------------------------------------------------
    # # non pytorch adam
    # if not NN_training_options["pytorch"] and "Adam" in NN_training_options["methods"]:
    #     print("Running Adam.")
    #     res_Adam = run_Gradient_Descent(method="Adam",
    #                                    theta0 = theta0,
    #                                    NN_object = NN_object,
    #                                    params = params,
    #                                    itbound = NN_training_options["itbound_SGD_algorithms"],
    #                                    batchsize = NN_training_options["batchsize"],
    #                                    check_exit_criteria = NN_training_options["check_exit_criteria"],
    #                                    output_progress = NN_training_options["output_progress"],
    #                                    nit_running_min = NN_training_options["nit_running_min"],
    #                                    nit_IterateAveragingStart=NN_training_options["nit_IterateAveragingStart"],
    #                                    Adam_ewma_1 = NN_training_options["Adam_ewma_1"],
    #                                    Adam_ewma_2 = NN_training_options["Adam_ewma_2"],
    #                                    Adam_eta = NN_training_options["Adam_eta"],
    #                                    Adam_epsilon = NN_training_options["Adam_epsilon"]
    #                                    )

    #     res_ALL["res_Adam"] = res_Adam  # append to res_ALL
    #     #print(res_Adam)


    # CONSTRUCT OUTPUTS: ---------------------------------------------------------------------------
    val_min = np.inf  # initialize running minimum
    res_ALL_dataframe = pd.DataFrame() #initialize

    #Loop through res_ALL.keys() [i.e. loop through all methods run] to construct outputs
    for key in res_ALL.keys():

        #Append
        res_ALL_dataframe = res_ALL_dataframe.append(res_ALL[key]["summary_df"], ignore_index=True)

        # Select result from which achieves lowest overall objective function value
        if res_ALL[key]["val"] < val_min:
            val_min = res_ALL[key]["val"]  # set new running min for objective function value
            res_BEST = res_ALL[key]  # res_BEST contains the results for the new running min


    #Finally append res_BEST to the bottom of res_ALL_dataframe
    if params["preTrained_TrueFalse"] is False: #If we actually did training as above
        res_BEST_temp = res_BEST #create temp copy to indicate that it has been selected
        res_BEST_temp["summary_df"]["method"] = "SELECTED: " + res_BEST_temp["method"]
        res_ALL_dataframe = res_ALL_dataframe.append(res_BEST_temp["summary_df"], ignore_index=True)

    elif params["preTrained_TrueFalse"] is True: #Otherwise just copy provided F_theta across if provided
        res_BEST.update({"F_theta": params["preTrained_F_theta"]})


    # print("------------------------------------------------------")
    # print("Contents of res_BEST:")
    # print(res_BEST)
    #   res_BEST = dictionary of results for the **best-performing** method in NN_training_options["methods"]
    #               achieving the LOWEST objective function value

    #---------------------------------------------------------------------------
    # FINAL RESULT and do LRP/PRP if needed
    #Implement the res_BEST trading strategy (NN parameters) and update the params dictionary
    params["res_BEST"] = res_BEST

    #Also do LRP/PRP if required
    LRP_for_NN_TrueFalse = params["LRP_for_NN_TrueFalse"]
    PRP_TrueFalse = params["PRP_TrueFalse"]

    #   Note: this invests the NN strategy, updates all the terminal wealth and objective function values in params
    params, _, _, _ = fun_eval_objfun_NN_strategy.eval_obj_NN_strategy(F_theta = res_BEST["F_theta"],
                                                                       NN_object = NN_object,
                                                                       params = params,
                                                                       output_Gradient = True,
                                                                       LRP_for_NN_TrueFalse = LRP_for_NN_TrueFalse,
                                                                       PRP_TrueFalse = PRP_TrueFalse)

    # Update  params["res_BEST"]["NN_theta"] for subsequent use
    if params["preTrained_TrueFalse"] is True:
        params["res_BEST"].update({"NN_theta": params["NN_theta"]})

    return params, res_BEST, res_ALL, res_ALL_dataframe