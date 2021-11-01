import pandas as pd
import numpy as np

def fun_W_T_summary_stats(W_T, prefix_output = "W_T"):

    #Objective: Calculates summary statistics: mean, stdev,
    #           percentiles etc for terminal wealth vector W_T

    # INPUTS
    #   W_T = numpy array with terminal wealth values, one entry for each sample path
    # prefix_output = prefix used in output description

    # OUTPUTS
    #   W_T_stats_dict = DICTIONARY with W_T_stats

    #   Contents:
    #       W_T_summary_stats  = pandas.DataFrame with the summary results
    #       W_T_mean = mean of W_T
    #       W_T_median = median of W_T
    #       W_T_std = standard deviation of W_T
    #       W_T_pctile_1st = 1st percentile of W_T
    #       W_T_pctile_5th = 5th percentile of W_T
    #       W_T_CVAR_1_pct = 1% CVAR
    #       W_T_CVAR_5_pct = 5% CVAR


    W_T_stats_dict = {} #initialize output

    #Convert to numpy if it isn't already
    W_T = np.array(W_T)

    if W_T.ndim > 2:
        raise ValueError("PVS error in 'fun_W_T_summary_stats': W_T dimensions too large. ")
    if W_T.ndim == 2:
        a,b = W_T.shape
        if a == 1 or b == 1:
            W_T = W_T.flatten() #Flatten into a vector
        else:
            raise ValueError("PVS error in 'fun_W_T_summary_stats': W_T dimensions too large. ")


    #Create pandas dataframe for percentiles
    W_T_df = pd.DataFrame(W_T)  #Convert to pandas dataframe

    W_T_summary_stats = W_T_df.describe(percentiles=[0.01, 0.05, 0.10, .25, .5, .75])

    #Calculate other quantities (some may be in the W_T_summary_stats dataframe)
    W_T_mean = np.mean(W_T)
    W_T_median = np.median(W_T)

    if np.std(W_T) > 0.0:    #To avoid errors with sample stdev ddof = 1
        W_T_std = np.std(W_T, ddof=1)
    else:
        W_T_std = np.std(W_T)


    #Calculate percentiles of interest
    pctiles = [1,2,3,4,5,6,7,8,9,10,15,20,25,30,35, 40, 45, 50,60,70,75,80,90,95,99]
    dict_percentiles = {}
    for pct in pctiles:
        key = prefix_output + "_pctile_" + str(pct)
        val = np.percentile(W_T, pct)
        dict_percentiles.update({key : val})


    #CVAR for various percentiles
    pctiles_CVAR = [1,5,10,25,50]
    dict_CVARs = {}

    for pct in pctiles_CVAR:
        key = prefix_output + "_CVAR_" + str(pct) + "_pct"
        val = np.mean(W_T[W_T <= np.percentile(W_T, pct)])  #Calculate CVAR at chosen percentile
        dict_CVARs.update({key:val})


    #Create dictionary for output
    if prefix_output == "W_T":  #Only add summary stats set if dealing with wealth
        W_T_stats_dict.update({prefix_output + "_summary_stats" : W_T_summary_stats})

    W_T_stats_dict.update({prefix_output + "_mean": W_T_mean})
    W_T_stats_dict.update({prefix_output + "_median": W_T_median})
    W_T_stats_dict.update({prefix_output + "_std": W_T_std})

    #Add CVARs to W_T_stats_dict
    for key in dict_CVARs.keys():
        val = dict_CVARs[key]
        W_T_stats_dict.update({key: val})

    #Add percentiles to W_T_stats_dict
    for key in dict_percentiles.keys():
        val = dict_percentiles[key]
        W_T_stats_dict.update({key: val})


    return W_T_stats_dict
