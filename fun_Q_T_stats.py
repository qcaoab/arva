import pandas as pd
import numpy as np

def fun_Q_T_summary_stats(Q_T, prefix_output = "Q_T"):

    #Objective: Calculates summary statistics: mean, stdev,
    #           percentiles etc for terminal wealth vector Q_T

    # INPUTS
    #   Q_T = numpy array with terminal wealth values, one entry for each sample path
    # prefix_output = prefix used in output description

    # OUTPUTS
    #   Q_T_stats_dict = DICTIONARY with Q_T_stats

    #   Contents:
    #       Q_T_summary_stats  = pandas.DataFrame with the summary results
    #       Q_T_mean = mean of Q_T
    #       Q_T_median = median of Q_T
    #       Q_T_std = standard deviation of Q_T
    #       Q_T_pctile_1st = 1st percentile of Q_T
    #       Q_T_pctile_5th = 5th percentile of Q_T
    #       Q_T_CVAR_1_pct = 1% CVAR
    #       Q_T_CVAR_5_pct = 5% CVAR


    Q_T_stats_dict = {} #initialize output

    #Convert to numpy if it isn't already
    Q_T = np.array(Q_T)

    if Q_T.ndim > 2:
        raise ValueError("PVS error in 'fun_Q_T_summary_stats': Q_T dimensions too large. ")
    if Q_T.ndim == 2:
        a,b = Q_T.shape
        if a == 1 or b == 1:
            Q_T = Q_T.flatten() #Flatten into a vector
        else:
            raise ValueError("PVS error in 'fun_Q_T_summary_stats': Q_T dimensions too large. ")


    #Create pandas dataframe for percentiles
    Q_T_df = pd.DataFrame(Q_T)  #Convert to pandas dataframe

    Q_T_summary_stats = Q_T_df.describe(percentiles=[0.01, 0.05, 0.10, .25, .5, .75])

    #Calculate other quantities (some may be in the Q_T_summary_stats dataframe)
    Q_T_mean = np.mean(Q_T)
    Q_T_median = np.median(Q_T)

    if np.std(Q_T) > 0.0:    #To avoid errors with sample stdev ddof = 1
        Q_T_std = np.std(Q_T, ddof=1)
    else:
        Q_T_std = np.std(Q_T)


    #Calculate percentiles of interest
    pctiles = [1,2,3,4,5,6,7,8,9,10,15,20,25,30,35, 40, 45, 50,60,70,75,80,90,95,99]
    dict_percentiles = {}
    for pct in pctiles:
        key = prefix_output + "_pctile_" + str(pct)
        val = np.percentile(Q_T, pct)
        dict_percentiles.update({key : val})


    #CVAR for various percentiles
    pctiles_CVAR = [1,5,10,25,50]
    dict_CVARs = {}

    for pct in pctiles_CVAR:
        key = prefix_output + "_CVAR_" + str(pct) + "_pct"
        val = np.mean(Q_T[Q_T <= np.percentile(Q_T, pct)])  #Calculate CVAR at chosen percentile
        dict_CVARs.update({key:val})


    #Create dictionary for output
    if prefix_output == "Q_T":  #Only add summary stats set if dealing with wealth
        Q_T_stats_dict.update({prefix_output + "_summary_stats" : Q_T_summary_stats})

    Q_T_stats_dict.update({prefix_output + "_mean": Q_T_mean})
    Q_T_stats_dict.update({prefix_output + "_median": Q_T_median})
    Q_T_stats_dict.update({prefix_output + "_std": Q_T_std})

    #Add CVARs to Q_T_stats_dict
    for key in dict_CVARs.keys():
        val = dict_CVARs[key]
        Q_T_stats_dict.update({key: val})

    #Add percentiles to Q_T_stats_dict
    for key in dict_percentiles.keys():
        val = dict_percentiles[key]
        Q_T_stats_dict.update({key: val})


    return Q_T_stats_dict
