import pandas as pd
import numpy as np


def fix_signal_lags(df_output, #df with calculated trading signals
                  nr_most_recent_skip ): # Number of most recent observations to SKIP in calc

    #OBJECTIVE: Ensures trading signals does NOT "look into the future"
    # - also ensures the "nr_most_recent_skip" is implemented correctly in the calculation

    # ------------------------------------------------------------------------------------------------
    #STEP 1: Fix "nr_most_recent_skip": Number of most recent observations to SKIP in calc
    # Note this is simply a lag; don't need to change the calculation
    df_output = df_output.shift(periods=nr_most_recent_skip, axis=0, fill_value=np.nan).copy()

    # ------------------------------------------------------------------------------------------------
    #STEP 2: Ensure no "looking into the future"
    # SHIFT trading signals FORWARD:
    #  - returns for each month are as at the END of the month
    #  - trading signals for the month should be received at the START of the month
    #  - this shift has the effect of LAGGING the data on which the signals are calculated
    df_output = df_output.shift(periods=1, axis=0, fill_value=np.nan).copy()


    return df_output


def signal_simple_moving_average(df,   #pandas DataFrame
                          columns, #column names of DataFrame to use
                          window,    #size of moving window to use
                          signal_prefix,  # used in column name for identification
                          nr_most_recent_skip    #Number of most recent observations to SKIP in calc
                          ):

    df_output = pd.DataFrame(index=df.index)

    for col in columns:
        # Adds column with simple moving average of rolling window
        df_output[signal_prefix + col] = \
            df[col].rolling(window = window).mean()

    #FIX LAGS:
    df_output = fix_signal_lags(df_output,
                    nr_most_recent_skip)

    return df_output


def signal_rolling_stdev(df,  #pandas DataFrame
                         columns,  #column names of DataFrame to use
                         window,  #size of moving window to use
                         signal_prefix,  # used in column name for identification
                         nr_most_recent_skip  # Number of most recent observations to SKIP in calc
                         ):


    df_output = pd.DataFrame(index=df.index)

    for col in columns:
        #Adds column with standard deviation of rolling window
        df_output[signal_prefix + col] = \
            df[col].rolling(window = window).std(ddof = 1)


    #FIX LAGS:
    df_output = fix_signal_lags(df_output,
                    nr_most_recent_skip)

    return df_output


