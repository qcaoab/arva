import numpy as np
import fun_Data_MCsim
import copy
#Objective: Wraps the MC simulation of training/testing to reduce the length of main code

def wrap_run_MCsim(train_test_Flag,  # "train" or "test"
                   params,  # params dictionary as in main code
                   model_ID_set_identifier  # (see code)identifier for the collection of models AND correlations to use
                   ):
    # OUTPUT: params dictionary, with following fields appended/modified
    #        Note: Below, xx \in {"train", "test"}

    # params["MCsim_info_xx"]: inputs used to get the bootstrapping results

    # ASSET return data: ( ## NO trading signals with MC!  ##)
    #   params["Y_xx"][j, n, i] = Return, along sample path j, over time period (t_n, t_n+1), for asset i
    #       -- IMPORTANT: params["Y_xx"][j, n, i] entries are basically (1 + return),
    #                           so it is ready for multiplication with start value
    #   params["Y_order_xx"][i] = column name of asset i used for identification

    params = copy.deepcopy(params)

    if train_test_Flag == "train":
        N_d = params["N_d_train"]

    elif train_test_Flag == "test":
        N_d = params["N_d_test"]

    # ----------------------------------------
    # MC simulation info
    MCsim_info = {}
    MCsim_info["N_d"] = N_d  # Number of simulations
    MCsim_info["purpose"] = train_test_Flag  # "train" or "test"

    # Assign simulation model_IDs and correlation matrix
    #   Assign model_IDs: Make sure it is in the CORRECT ORDER!
    # - Order should correspond to order of params["asset_basket"]["basket_columns"], specifically:
    # - MCsim_info["model_ID_XX"] should correspond to model for params["asset_basket"]["basket_columns"][XX]

    if params["asset_basket"]["basket_id"] == "B10_and_VWD":

        if model_ID_set_identifier == "Forsyth_retirementDC_2020" and params["real_or_nominal"] == "real":

            #Order should correspond to order of params["asset_basket"]["basket_columns"]
            MCsim_info["model_ID_0"] = "B10_kou_Forsyth_retirementDC"
            MCsim_info["model_ID_1"] = "VWD_kou_Forsyth_retirementDC"

            # Assign correlations: CHECK ORDER if >= 2 assets!
            rho_sb = 0.04554
            MCsim_info["corr_matrix"] = np.array([[1, rho_sb], [rho_sb, 1]])

        else:
            raise ValueError("Error in wrap_run_MCsim: model_ID_set_identifier not assigned, or real_or_nominal != 'real'.")

    elif params["asset_basket"]["basket_id"] == "basic_T90_VWD":
        if model_ID_set_identifier == "Forsyth_2021_benchmark" and params["real_or_nominal"] == "nominal":
            #Order should correspond to order of params["asset_basket"]["basket_columns"]
            MCsim_info["model_ID_0"] = "T90_Forsyth_2021_benchmark"
            MCsim_info["model_ID_1"] = "VWD_Forsyth_2021_benchmark"

            # Assign correlations: CHECK ORDER if >= 2 assets!
            rho_sb = 0.0
            MCsim_info["corr_matrix"] = np.array([[1, rho_sb], [rho_sb, 1]])





    elif params["asset_basket"]["basket_id"] == "basic_T30_VWD":

        if model_ID_set_identifier == "PVS_196307_to_202012_benchmark_GBM" and params["real_or_nominal"] == "real":
            #Order should correspond to order of params["asset_basket"]["basket_columns"]
            MCsim_info["model_ID_0"] = "PVS_196307_to_202012_benchmark_GBM_T30"
            MCsim_info["model_ID_1"] = "PVS_196307_to_202012_benchmark_GBM_VWD"

            # Assign correlations: CHECK ORDER if >= 2 assets!
            rho_sb = 0.0
            MCsim_info["corr_matrix"] = np.array([[1, rho_sb], [rho_sb, 1]])

        elif model_ID_set_identifier == "PVS_196307_to_202012_benchmark_kou" and params["real_or_nominal"] == "real":
            # Order should correspond to order of params["asset_basket"]["basket_columns"]
            MCsim_info["model_ID_0"] = "PVS_196307_to_202012_benchmark_kou_T30"
            MCsim_info["model_ID_1"] = "PVS_196307_to_202012_benchmark_kou_VWD"

            # Assign correlations: CHECK ORDER if >= 2 assets!
            rho_sb = 0.0
            MCsim_info["corr_matrix"] = np.array([[1, rho_sb], [rho_sb, 1]])





        elif model_ID_set_identifier == "PVS_2020_benchmark_GBM" and params["real_or_nominal"] == "real":
            #Order should correspond to order of params["asset_basket"]["basket_columns"]
            MCsim_info["model_ID_0"] = "PVS_2020_benchmark_GBM_T30"
            MCsim_info["model_ID_1"] = "PVS_2020_benchmark_GBM_VWD"

            # Assign correlations: CHECK ORDER if >= 2 assets!
            rho_sb = 0.0
            MCsim_info["corr_matrix"] = np.array([[1, rho_sb], [rho_sb, 1]])

        elif model_ID_set_identifier == "PVS_2020_benchmark_kou" and params["real_or_nominal"] == "real":
            # Order should correspond to order of params["asset_basket"]["basket_columns"]
            MCsim_info["model_ID_0"] = "PVS_2020_benchmark_kou_T30"
            MCsim_info["model_ID_1"] = "PVS_2020_benchmark_kou_VWD"

            # Assign correlations: CHECK ORDER if >= 2 assets!
            rho_sb = 0.0
            MCsim_info["corr_matrix"] = np.array([[1, rho_sb], [rho_sb, 1]])
        

        elif model_ID_set_identifier == "PVS_2019_benchmark_kou" and params["real_or_nominal"] == "real":
            # Order should correspond to order of params["asset_basket"]["basket_columns"]
            MCsim_info["model_ID_0"] = "PVS_2019_benchmark_kou_T30"
            MCsim_info["model_ID_1"] = "PVS_2019_benchmark_kou_VWD"

            # Assign correlations: CHECK ORDER if >= 2 assets!
            rho_sb = 0.08228
            MCsim_info["corr_matrix"] = np.array([[1, rho_sb], [rho_sb, 1]])


    elif params["asset_basket"]["basket_id"] == "basic_ForsythLi":
        if model_ID_set_identifier == "ForsythLi_2019_basic" and params["real_or_nominal"] == "real":
            #Order should correspond to order of params["asset_basket"]["basket_columns"]
            MCsim_info["model_ID_0"] = "T30_LiForsyth2019"
            MCsim_info["model_ID_1"] = "VWD_kou_LiForsyth2019"

            # Assign correlations: CHECK ORDER if >= 2 assets!
            rho_sb = 0.0
            MCsim_info["corr_matrix"] = np.array([[1, rho_sb], [rho_sb, 1]])

        elif model_ID_set_identifier == "Forsyth_2020_MeanCVAR" and params["real_or_nominal"] == "real":
            #Order should correspond to order of params["asset_basket"]["basket_columns"]
            MCsim_info["model_ID_0"] = "T30_Forsyth_2020"
            MCsim_info["model_ID_1"] = "VWD_kou_Forsyth_MeanCVAR"

            # Assign correlations: CHECK ORDER if >= 2 assets!
            rho_sb = 0.0
            MCsim_info["corr_matrix"] = np.array([[1, rho_sb], [rho_sb, 1]])

        else:
            raise ValueError("Error in wrap_run_MCsim: model_ID_set_identifier not assigned, or real_or_nominal != 'real'.")

    else:
        raise ValueError("Error in wrap_run_MCsim: Asset basket id not assigned params for  MC simulation.")

    # ----------------------------------------
    # APPEND MC sim info to "params" dictionary
    params["MCsim_info_" + train_test_Flag] = MCsim_info.copy()  # copy over

    del MCsim_info

    # ----------------------------------------
    # Append simulated data to "params" dictionary
    params = fun_Data_MCsim.MCsim(MCsim_info=params["MCsim_info_" + train_test_Flag],
                                  params=params)
    # ASSET return data: xx \in {"train", "test"}
    #   params["Y_xx"][j, n, i] = Return, along sample path j, over time period (t_n, t_n+1), for asset i
    #       -- IMPORTANT: params["Y_xx"][j, n, i] entries are basically (1 + return), so it is ready for multiplication with start value
    #   params["Y_order_xx"][i] = column name of asset i used for identification

    return params
