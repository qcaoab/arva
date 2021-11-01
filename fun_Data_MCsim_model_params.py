#Specifies the parameters of the parametric model based on Model ID

import numpy as np

def get_model_params(model_ID, real_or_nominal = "real"):
    import math
    # Objective: returns dictionary with underlying asset parameters
    #           including some basic combinations of parameters

    # INPUTS: model_ID identifies the set of model parameters to use
    #         - see below for options
    #           Prefix "VWD_" refers to the value-weighted index of CRSP stocks
    #           Prefix "B10_" refers to the 10 year US T_bond

    # OUTPUTS:
    # params_dict = {
    #     "r": r,        #  risk-free rate
    #     "mu": mu,      # drift
    #     "sig": sig,    # diffusive vol
    #     "lambda_jump": lambda_jump, #intensity Poisson
    #     "kappa": kappa,
    #     "kappa2": kappa2,
    #     "varparam": varparam,
    #     "A": A
    # }


    if real_or_nominal == "real":

        #--------------------------------------------------------------------------------------------
        # Parameters used in all my papers

        # Risk-free asset
        #r =  0.00623  # risk-free rate
        r = 0.0074  # r = 0.0074: 196307 to 202012 calibration
        # Risky asset parameters

        if model_ID == "VWD_gbm":
            mu = 0.0816
            sig = 0.1863
            lambda_jump = 0.0
            kappa = 0.0
            kappa2 = 0.0

        elif model_ID == "PVS_196307_to_202012_benchmark_GBM_T30":  # Basically use "GBM" but set vol to zero
            r = 0.0  # overwrite (but not used)
            mu = 0.0074  # This is risk-free rate, we set vol to zero
            sig = 0.0
            lambda_jump = 0.0
            kappa = 0.0
            kappa2 = 0.0

        elif model_ID == "PVS_196307_to_202012_benchmark_GBM_VWD":  #GBM
            r = 0.0  # overwrite (but not used)
            mu =  0.0728  # This is risk-free rate, we set vol to zero
            sig =  0.1562
            lambda_jump = 0.0
            kappa = 0.0
            kappa2 = 0.0


        elif model_ID == "PVS_2020_benchmark_GBM_T30":  # Basically use "GBM" but set vol to zero
            r = 0.0  # overwrite (but not used)
            mu = 0.0043  # This is risk-free rate, we set vol to zero
            sig = 0.0
            lambda_jump = 0.0
            kappa = 0.0
            kappa2 = 0.0

        elif model_ID == "PVS_2020_benchmark_GBM_VWD":  #GBM
            r = 0.0  # overwrite (but not used)
            mu =  0.0835  # This is risk-free rate, we set vol to zero
            sig =  0.1854
            lambda_jump = 0.0
            kappa = 0.0
            kappa2 = 0.0

        elif model_ID == "T30_LiForsyth2019":   #Basically use "GBM" but set vol to zero
            r = 0.0  # overwrite (but not used)
            mu = 0.00827    #This is risk-free rate, we set vol to zero
            sig = 0.0
            lambda_jump = 0.0
            kappa = 0.0
            kappa2 = 0.0

        elif model_ID == "T30_Forsyth_2020":   #Basically use "GBM" but set vol to zero
            r = 0. # overwrite (but not used)
            mu = 0.00464    #This is risk-free rate, we set vol to zero
            sig = 0.0
            lambda_jump = 0.0
            kappa = 0.0
            kappa2 = 0.0

        if "merton" in model_ID:  #If merton model

            if model_ID == "VWD_merton2":
                mu = 0.0822
                sig = 0.0972  # diffusive vol
                lambda_jump = 2.3483 # intensity of Poisson
                m_tilde = -0.0192  # mean of log jump mult dist
                gamma_tilde = 0.1058 # stdev of log jump mult dist

            elif model_ID == "VWD_merton3":
                mu = 0.0817
                sig = 0.1453   # diffusive vol
                lambda_jump = 0.3483     # intensity of Poisson
                m_tilde = -0.0700    # mean of log jump mult dist
                gamma_tilde = 0.1924     # stdev of log jump mult dist


            elif model_ID == "VWD_merton4":

                mu = 0.0820
                sig = 0.1584  # diffusive vol
                lambda_jump = 0.1461    # intensity of Poisson
                m_tilde = -0.0521    # mean of log jump mult dist
                gamma_tilde = 0.2659    # stdev of log jump mult dist

            else:
                raise Exception("PVS error: No params for this model_ID for merton model.")

            #   Calculate kappa and kappa2 for "merton" model
            kappa = math.exp(m_tilde + 0.5*(gamma_tilde ** 2) )  - 1
            kappa2 = math.exp( 2*m_tilde + 2*(gamma_tilde**2) ) - 2*math.exp(m_tilde + 0.5*(gamma_tilde**2)) + 1.0


        if "kou" in model_ID:  # If KOU model

            if model_ID == "PVS_196307_to_202012_benchmark_kou_T30":   #Calibrated parameters 196307_to_202012, PVS stochastic BM paper
                r = 0.0 #overwrite (but not used)
                mu = 0.0074 #Basically plays the role of r
                sig = 0.  # diffusive vol
                lambda_jump = 0.  # intensity of Poisson
                nu = 0.  # prob of UP jump
                zeta1 = 0.  # expo param of UP jump
                zeta2 = 0.  # expo param of DOWN jump

            elif model_ID == "PVS_196307_to_202012_benchmark_kou_VWD": #Calibrated parameters 196307_to_202012, PVS stochastic BM paper
                r = 0.0 #overwrite (but not used)
                mu =   0.0749   #drift
                sig =   0.1392    # diffusive vol
                lambda_jump =   0.2090   # intensity of Poisson
                nu =   0.2500    # prob of UP jump
                zeta1 =   7.7830    # expo param of UP jump
                zeta2 =   6.1074   # expo param of DOWN jump



            elif model_ID == "PVS_2020_benchmark_kou_T30":   #Calibrated parameters up to end of 2019, PVS stochastic BM paper
                r = 0.0 #overwrite (but not used)
                mu = 0.0043 #Basically plays the role of r
                sig = 0.  # diffusive vol
                lambda_jump = 0.  # intensity of Poisson
                nu = 0.  # prob of UP jump
                zeta1 = 0.  # expo param of UP jump
                zeta2 = 0.  # expo param of DOWN jump

            elif model_ID == "PVS_2020_benchmark_kou_VWD": #Calibrated parameters up to end of 2019, PVS stochastic BM paper
                r = 0.0 #overwrite (but not used)
                mu =  0.0891  #drift
                sig =  0.1469   # diffusive vol
                lambda_jump =  0.3263  # intensity of Poisson
                nu =  0.2258   # prob of UP jump
                zeta1 =  4.3626   # expo param of UP jump
                zeta2 =  5.5336   # expo param of DOWN jump

            elif model_ID == "PVS_2019_benchmark_kou_T30":   #Calibrated parameters up to end of 2019, PVS stochastic BM paper
                r = 0.0 #overwrite (but not used)
                mu = 0.0044 #Basically plays the role of r
                sig = 0.  # diffusive vol
                lambda_jump = 0.  # intensity of Poisson
                nu = 0.  # prob of UP jump
                zeta1 = 0.  # expo param of UP jump
                zeta2 = 0.  # expo param of DOWN jump

            elif model_ID == "PVS_2019_benchmark_kou_VWD": #Calibrated parameters up to end of 2019, PVS stochastic BM paper
                r = 0.0 #overwrite (but not used)
                mu =  0.0877  #drift
                sig =  0.1459   # diffusive vol
                lambda_jump =  0.3191  # intensity of Poisson
                nu =  0.2333   # prob of UP jump
                zeta1 =  4.3608   # expo param of UP jump
                zeta2 =  5.5040   # expo param of DOWN jump


            elif model_ID == "VWD_kou_Forsyth_retirementDC": #parameters for STOCK in Forsyth retirement decumulation paper
                r = 0.0 #overwrite (but not used)
                mu = 0.0877
                sig = 0.1459  # diffusive vol
                lambda_jump = 0.3191  # intensity of Poisson
                nu = 0.2333  # prob of UP jump
                zeta1 = 4.3608  # expo param of UP jump
                zeta2 = 5.504  # expo param of DOWN jump

            elif model_ID == "B10_kou_Forsyth_retirementDC": #parameters for BOND in Forsyth retirement decumulation paper
                r = 0.0 #overwrite (but not used)
                mu = 0.0239
                sig = 0.0538  # diffusive vol
                lambda_jump = 0.3830  # intensity of Poisson
                nu = 0.6111  # prob of UP jump
                zeta1 = 16.19  # expo param of UP jump
                zeta2 = 17.27  # expo param of DOWN jump

            elif model_ID == "VWD_kou_LiForsyth2019": #parameters for Kou model cap-weighted in Li and Forsyth 2019 paper
                r = 0.0 #overwrite (but not used)
                mu = 0.08889
                sig = 0.14771  # diffusive vol
                lambda_jump = 0.32222  # intensity of Poisson
                nu = 0.27586  # prob of UP jump
                zeta1 = 4.4273  # expo param of UP jump
                zeta2 = 5.2613  # expo param of DOWN jump

            elif model_ID == "VWD_kou_Forsyth_MeanCVAR": #parameters for Kou model in Peter Forsyth SIAM Mean-CVAR paper
                r = 0.00 #overwrite but not used
                mu = 0.0884
                sig = 0.1451  # diffusive vol
                lambda_jump = 0.3370  # intensity of Poisson
                nu = 0.2581  # prob of UP jump
                zeta1 = 4.681  # expo param of UP jump
                zeta2 = 5.600  # expo param of DOWN jump

            elif model_ID == "VWD_kou2":
                mu = 0.0896
                sig = 0.0970  # diffusive vol
                lambda_jump = 2.3483  # intensity of Poisson
                nu = 0.4258  # prob of UP jump
                zeta1 = 11.2321  # expo param of UP jump
                zeta2 = 10.1256  # expo param of DOWN jump

            elif model_ID == "VWD_kou3":
                mu = 0.0874
                sig = 0.1452  # diffusive vol
                lambda_jump = 0.3483  # intensity of Poisson
                nu = 0.2903  # prob of UP jump
                zeta1 = 4.7941  # expo param of UP jump
                zeta2 = 5.4349  # expo param of DOWN jump


            elif model_ID == "VWD_kou4":
                mu = 0.0866
                sig = 0.1584  # diffusive vol
                lambda_jump = 0.1461  # intensity of Poisson
                nu = 0.3846  # prob of UP jump
                zeta1 = 3.7721  # expo param of UP jump
                zeta2 = 3.9943  # expo param of DOWN jump

            else:
                raise Exception("PVS error: No params for this model_ID for kou model.")

            #   Calculate kappa and kappa2 for "kou" model
            kappa = nu * zeta1 / (zeta1 - 1) + (1 - nu) * zeta2 / (zeta2 + 1) - 1
            kappa2 = nu * zeta1 / (zeta1 - 2) + (1 - nu) * zeta2 / (zeta2 + 2) - 2 * kappa - 1

    elif real_or_nominal == "nominal":
        if model_ID == "VWD_Forsyth_2021_benchmark":   #Basically use "GBM"
            #These are just test parameters
            r = 0.0 # overwrite
            mu = 0.08
            sig = 0.15
            lambda_jump = 0.0
            kappa = 0.0
            kappa2 = 0.0

        elif model_ID == "T90_Forsyth_2021_benchmark":   #Risk-free asset
            #These are just test parameters
            r = 0.0 # overwrite
            mu =  0.02
            sig = 0.0
            lambda_jump = 0.0
            kappa = 0.0
            kappa2 = 0.0

    # -------------------------------------------------------------------------
    # Do calcs applicable to all models and output

    #   Calculate the "variance parameter" frequently encountered
    varparam = (sig ** 2) + lambda_jump * kappa2

    #   Calculate ratio of parameters A
    if varparam < 1e-8:
        A = np.inf
    else:
        A   = ((mu - r) ** 2) / varparam

    params_dict = {
        "model_ID": model_ID,
        "r": r,  # risk-free rate
        "mu": mu,  # drift
        "sig": sig,  # diffusive vol
        "lambda_jump": lambda_jump,  # intensity Poisson
        "kappa": kappa,
        "kappa2": kappa2,
        "varparam": varparam,
        "A": A
    }

    # -------------------------------------------------------------------------
    # Add parameters applicable to specific models
    if "kou" in model_ID:
        params_dict["nu"] = nu  # prob of UP jump
        params_dict["zeta1"] = zeta1    # expo param of UP jump
        params_dict["zeta2"] = zeta2  # expo param of DOWN jump

    if "merton" in model_ID:
        params_dict["m_tilde"] = m_tilde     # mean of log jump mult dist
        params_dict["gamma_tilde"] = gamma_tilde  # stdev of log jump mult dist

    return params_dict
