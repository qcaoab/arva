import pandas as pd
import numpy as np
from fun_invest_NN_strategy import invest_NN_strategy
from fun_train_NN import train_NN
import copy

def trace_eff_frontier_NN_using_param(trace_param_array,  # array of values of trace parameter "p"" to use
                                      params,  # params dictionary as setup in main code, gives objective function
                                      NN_object,  # object of class Neural_Network
                                      NN_training_options,  # training options used for each training of NN
                                      initialize_scheme="glorot_bengio"):  # init scheme for each training of NN

    # OBJECTIVE: Traces out "efficient frontier" by varying a SINGLE  parameter "p" in the objective function
    # - "efficient frontier" is *NOT* necessarily mean-variance optimal, but instead
    #       just points in (Stdev, ExpVal)-plane **ALSO** outputs a bunch of percentiles of terminal wealth
    # - objective function is given in params["obj_fun"], trace parameter "p" identified in IF statement below

    # OUTPUT:
    # return eff_frontier, eff_frontier_df
    # eff_frontier = dictionary of dictionaries with info on each "efficient frontier" point
    # eff_frontier_df = pd.DataFrame giving terminal wealth summary stats: expected val, stdev, percentiles etc.

    # gives output for each p in trace_param_array
    # - for each value of tracing parameter p,  NN_object is trained to find optimal point for given objective function


    # will generate points for values of p in trace_param_array
    trace_param_array = np.array(trace_param_array)

    counter = 0  # Used as dictionary key for output

    eff_frontier = {}  # Dictionary of dictionaries (efficient frontier points)
    eff_frontier_point = {}  # All info associated with particular eff frontier point

    # For convenient summary, create efficient frontier dataframe
    eff_frontier_df = pd.DataFrame()

    for p in trace_param_array:

        #output
        print("Running parameter nr " + str(counter + 1) + " of " + str(len(trace_param_array)) + ".")

        # params dictionary for this value
        fdata_p = copy.deepcopy(params) #Create a copy

        # Assign parameter used to trace out efficient frontier
        if params["obj_fun"] == "one_sided_quadratic_target_error" or params["obj_fun"] == "quad_target_error":
            # In this case it is the terminal wealth target
            fdata_p["obj_fun_W_target"] = p
        else:
            raise ValueError("PVSerror in 'trace_eff_frontier_NN_using_param': parameter tracing of "
                             "efficient frontier should be updated to handle this objective function.")

        # Get initial point for new NN training
        theta0 = NN_object.initialize_NN_parameters(initialize_scheme=initialize_scheme)

        # train NN to get optimal results for this obj function parameter
        _, res, _, _ = train_NN(NN_theta0=theta0,
                             NN_object=NN_object,
                             params=fdata_p,
                             NN_training_options=NN_training_options
                             )
        # Invest the optimal NN to get terminal wealth info
        theta_opt = res["NN_theta"]
        fdata_p = invest_NN_strategy(theta_opt, NN_object, fdata_p, output_Gradient=False)

        # Create dictionary for this efficient frontier point dictionary
        eff_frontier_point = {'counter': counter,
                              'eff_frontier_parameter': p,  # parameter used to trace out efficient frontier
                              'W_T_std': res["W_T_std"],
                              'W_T_mean': res["W_T_mean"],
                              "NN_theta": res["NN_theta"],
                              'obj_func_val': res["val"],
                              'method' : res["method"],     #algorithm to obtain optimal point
                              'results_dict' : res,   #results dictionary from training of NN for this point
                              "params": fdata_p       #params dictionary for this point
                              }

        # Append efficient frontier point  to efficient frontier dictionary
        eff_frontier.update({counter: eff_frontier_point})

        # Create dataframe for all efficient frontier points for convenience
        eff_frontier_df_part1 = pd.DataFrame(data=[[counter, p, res["val"], res["method"], res["runtime_mins"]]],
                                  columns=["counter", "eff_frontier_parameter", "obj_func_val", "method", "runtime_mins"])
        eff_frontier_df_part2 = fdata_p["W_T_summary_stats"].transpose()
        eff_frontier_df_row = pd.concat([eff_frontier_df_part1, eff_frontier_df_part2], ignore_index=False, axis=1)
        eff_frontier_df = eff_frontier_df.append(eff_frontier_df_row)

        counter = counter + 1

    # After loop has run:
    return eff_frontier, eff_frontier_df
