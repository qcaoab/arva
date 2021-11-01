#Objective: Compares the terminal wealth of the Benchmark strategy with the NN strategy

import pandas as pd
import numpy as np
import datetime
import fun_W_T_stats

def fun_W_T_comparison_BM_vs_NN(
        params_TRAIN,           # dictionary with parameters and results from NN TRAINING
        params_BM_TRAIN,        #dictionary with benchmark strategy results and info on the TRAINING dataset
        params_TEST = None,     # dictionary with parameters and results from NN TESTING
        params_BM_TEST = None,   #dictionary with benchmark strategy results and info on the TESTING dataset
        output_Excel=False,  # write the result to Excel
        filename_prefix="z_"  # used if output_Excel is True
):
    #Fix empty inputs
    if params_BM_TEST == {}:
        params_BM_TEST = None

    if params_TEST == {}:
        params_TEST = None

    #---------------------------------------
    #Training dataset
    W_T_BM_train = params_BM_TRAIN["W_T"].copy()
    W_T_NN_train = params_TRAIN["W_T"].copy()

    Y_T_train = np.divide(W_T_NN_train, W_T_BM_train)  # Ratio
    D_T_train = W_T_NN_train - W_T_BM_train  # Difference

    # ---------------------------------------
    # TESTING data
    if (params_TEST is not None) and (params_BM_TEST is not None):

        W_T_BM_test = params_BM_TEST["W_T"].copy()
        W_T_NN_test = params_TEST["W_T"].copy()

        Y_T_test = np.divide(W_T_NN_test, W_T_BM_test)  # Ratio
        D_T_test = W_T_NN_test - W_T_BM_test  # Difference

    else: #Set these to None to make the subsequent code work
        Y_T_test = None
        D_T_test = None


    # ---------------------------------------
    # Get summary stats
    df_stats =  BM_vs_NN_summary_stats(Y_T_train,
                   Y_T_test,
                   D_T_train,
                   D_T_test
                   )

    if output_Excel is True:
        timestamp = datetime.datetime.now().strftime('%Y-%m-%d_%H_%M')
        filename = filename_prefix + "timestamp_" + timestamp + "_W_T_NN_vs_Benchmark_stats" \
                   + ".xlsx"
        df_stats.to_excel(filename)

    # ---------------------------------------
    # Get histogram and CDF
    df_hist_Y_T, df_cdf_Y_T, df_hist_D_T, df_cdf_D_T = \
        BM_vs_NN_hist_and_CDF(Y_T_train,
                               Y_T_test,
                               D_T_train,
                               D_T_test
                               )
    if output_Excel is True:
        timestamp = datetime.datetime.now().strftime('%Y-%m-%d_%H_%M')

        filename = filename_prefix + "timestamp_" + timestamp + "_W_T_NN_vs_Benchmark_hist_and_cdf" \
                       + ".xlsx"

        with pd.ExcelWriter(filename) as writer:
            #Ratio details
            df_hist_Y_T.to_excel(writer, sheet_name="Ratio_hist_perc", header=True, index=True)
            df_cdf_Y_T.to_excel(writer, sheet_name="Ratio_CDF", header=True, index=True)

            #Difference details
            df_hist_D_T.to_excel(writer, sheet_name="Diff_hist_perc", header=True, index=True)
            df_cdf_D_T.to_excel(writer, sheet_name="Diff_CDF", header=True, index=True)


    return df_stats, df_hist_Y_T, df_cdf_Y_T, df_hist_D_T, df_cdf_D_T


def BM_vs_NN_hist_and_CDF(Y_T_train,
                   Y_T_test,
                   D_T_train,
                   D_T_test
                   ):

    #Bins
    bins_Y_T = np.linspace(start=0.0, stop=2., num=201) #Ratio bins
    bins_D_T = np.linspace(start=-500., stop=500., num=201)  # Difference bins

    #Bins right edges
    bins_Y_T_right_edges = bins_Y_T[1:]
    bins_D_T_right_edges = bins_D_T[1:]

    #Initialize dataframes
    df_hist_Y_T = pd.DataFrame(data=bins_Y_T_right_edges, columns=["bins_Y_T_right_edges"])
    df_cdf_Y_T = pd.DataFrame(data=bins_Y_T_right_edges, columns=["bins_Y_T_right_edges"])

    df_hist_D_T = pd.DataFrame(data=bins_D_T_right_edges, columns=["bins_D_T_right_edges"])
    df_cdf_D_T = pd.DataFrame(data=bins_D_T_right_edges, columns=["bins_D_T_right_edges"])

    #------------------------------------------------
    #Y_T_train
    nobs = Y_T_train.shape[0]
    counts_hist, _ = np.histogram(Y_T_train, bins=bins_Y_T)
    hist_perc = counts_hist/nobs
    df_hist_Y_T["Y_T_train"] = hist_perc
    df_cdf_Y_T["Y_T_train"] = df_hist_Y_T["Y_T_train"].cumsum()

    #------------------------------------------------
    #Y_T_test
    if Y_T_test is not None:
        nobs = Y_T_test.shape[0]
        counts_hist, _ = np.histogram(Y_T_test, bins=bins_Y_T)
        hist_perc = counts_hist/nobs
        df_hist_Y_T["Y_T_test"] = hist_perc
        df_cdf_Y_T["Y_T_test"] = df_hist_Y_T["Y_T_test"].cumsum()

    #------------------------------------------------
    #D_T_train
    nobs = D_T_train.shape[0]
    counts_hist, _ = np.histogram(D_T_train, bins=bins_D_T)
    hist_perc = counts_hist/nobs
    df_hist_D_T["D_T_train"] = hist_perc
    df_cdf_D_T["D_T_train"] = df_hist_D_T["D_T_train"].cumsum()


    #------------------------------------------------
    #D_T_test
    if D_T_test is not None:
        nobs = D_T_test.shape[0]
        counts_hist, _ = np.histogram(D_T_test, bins=bins_D_T)
        hist_perc = counts_hist/nobs
        df_hist_D_T["D_T_test"] = hist_perc
        df_cdf_D_T["D_T_test"] = df_hist_D_T["D_T_test"].cumsum()

    return df_hist_Y_T, df_cdf_Y_T, df_hist_D_T, df_cdf_D_T


def BM_vs_NN_summary_stats(Y_T_train,
                   Y_T_test,
                   D_T_train,
                   D_T_test
                   ):

    #------------------------------------------------
    #Y_T_train stats
    dict_temp = fun_W_T_stats.fun_W_T_summary_stats(Y_T_train, prefix_output="")
    df = pd.DataFrame.from_dict(dict_temp, orient="index", columns=["Y_T_train"])

    df_stats = df.copy()    #initialize output

    #------------------------------------------------
    #Y_T_test stats
    if Y_T_test is not None:
        dict_temp = fun_W_T_stats.fun_W_T_summary_stats(Y_T_test, prefix_output="")
        df = pd.DataFrame.from_dict(dict_temp, orient="index", columns=["Y_T_test"])

        #Merge onto output
        df_stats = df_stats.merge(df, how="inner", left_index=True, right_index=True)

    #------------------------------------------------
    # D_T_train stats
    dict_temp = fun_W_T_stats.fun_W_T_summary_stats(D_T_train, prefix_output="")
    df = pd.DataFrame.from_dict(dict_temp, orient="index", columns=["D_T_train"])

    # Merge onto output
    df_stats = df_stats.merge(df, how="inner", left_index=True, right_index=True)


    #------------------------------------------------
    # D_T_test stats
    if D_T_test is not None:
        dict_temp = fun_W_T_stats.fun_W_T_summary_stats(D_T_test, prefix_output="")
        df = pd.DataFrame.from_dict(dict_temp, orient="index", columns=["D_T_test"])

        # Merge onto output
        df_stats = df_stats.merge(df, how="inner", left_index=True, right_index=True)


    return df_stats




