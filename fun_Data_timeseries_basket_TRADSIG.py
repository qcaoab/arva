

import pandas as pd
import numpy as np
import fun_Data_timeseries_basket_ASSETS



def trading_signal_basket(basket_type,   #"trading_signal"
                 basket_id, # IDENTIFIER basket_id of pre-defined trading signals
                 underlying_asset_basket_id, #Asset basket_id on which it is based
                 real_or_nominal     #whether nominal or real historical returns should be used.
                ):
    #Constructs pre-defined trading signal basket
    #NOTE: Trading signals based on market data, and is incorporates asset baskets
    # --> Assumes MONTHLY underlying data

    trading_signal_basket = {}

    #------------------------------------------------------------------------------------------------
    #Get underlying asset basket on which trading signals will be based
    underlying_asset_basket = fun_Data_timeseries_basket_ASSETS.asset_basket(basket_type="asset",
                                 basket_id=underlying_asset_basket_id,
                                real_or_nominal=real_or_nominal,
                                 add_cash_TrueFalse = False)    #no cash signal!


    #------------------------------------------------------------------------------------------------
    #Details of individual trading signals of interest
    #- calculation rules can be found in: fun_Data_construct_trading_signals.py
    #- will be combined below in trading signal baskets
    #All quantities assumes MONTHLY data

    SignalSMA_mom_2_12 = {"function": "signal_simple_moving_average",  # function to calculate trading signal, see fun_Data_construct_trading_signals.py
                    "window": 11,  # size of moving window to use for actual calculation, EXCLUDING "nr_most_recent_mths_to_skip" [which is fixed via lags]
                    "signal_prefix": "SignalSMA_mom_2_12_", # prefix appended to column name for identification
                    "nr_most_recent_mths_to_skip" : 1}  #Specifies number of most recent months to SKIP in calc

    SignalSMA_3y_skip_12 = {"function": "signal_simple_moving_average",  # function to calculate trading signal, see
                    "window": 24,   #size of moving window to use for actual calculation, EXCLUDING "nr_most_recent_mths_to_skip" [which fixed via lags]
                    "signal_prefix": "SignalSMA_3y_skip_12_", # prefix appended to column name for identification
                    "nr_most_recent_mths_to_skip": 12 }


    SignalRSTD_12 = {"function": "signal_rolling_stdev",  # function to calculate trading signal
                     "window": 12,  #size of moving window to use for actual calculation, EXCLUDING "nr_most_recent_mths_to_skip" [which fixed via lags]
                     "signal_prefix": "SignalRSTD_12_", # prefix appended to column name for identification
                     "nr_most_recent_mths_to_skip": 0 }

    SignalRSTD_3y = {"function": "signal_rolling_stdev",  # function to calculate trading signal
                     "window": 36,  #size of moving window to use for actual calculation, EXCLUDING "nr_most_recent_mths_to_skip" [which fixed via lags]
                     "signal_prefix": "SignalRSTD_3y_",  # prefix appended to column name for identification
                     "nr_most_recent_mths_to_skip": 0}


    #------------------------------------------------------------------------------------------------
    #Define combinations of trading signals

    if basket_id == "mom_reversal":
        #description
        basket_desc = "mom_reversal"

        #List containing dictionaries of pre-defined trading signals above to include, for EACH asset in underlying_asset_basket
        basket_trading_signal_list = [SignalSMA_mom_2_12, SignalSMA_3y_skip_12]

        #SHORT label for e.g. figures
        basket_label = "mom_reversal"

    elif basket_id == "All_MA_RSTD":
        #description
        basket_desc = "All MAs and RSTDs"

        #List containing dictionaries of pre-defined trading signals above to include, for EACH asset in underlying_asset_basket
        basket_trading_signal_list = [SignalSMA_mom_2_12, SignalSMA_3y_skip_12, SignalRSTD_12, SignalRSTD_3y ]

        #SHORT label for e.g. figures
        basket_label = "All_MA_RSTD"

    else:
        raise ValueError("PVS error in trading_signal_basket: basket_id not coded.")



    # ------------------------------------------------------------------------------------------------
    # Construct trading_signal_basket

    # Loop through underlying asset basket and assign timeseries names and column names
    basket_timeseries_names = []
    for timeseries_name in underlying_asset_basket["basket_timeseries_names"]:
        for trading_signal in basket_trading_signal_list:
            # timeseries names for graphs etc.
            basket_timeseries_names.append(trading_signal["signal_prefix"] + timeseries_name)

    # identifiers of time series of returns to include
    basket_columns = []
    for asset_basket_column_name in underlying_asset_basket["basket_columns"]:
        for trading_signal in basket_trading_signal_list:
            # timeseries names for graphs etc.
            basket_columns.append(trading_signal["signal_prefix"] + asset_basket_column_name)


    trading_signal_basket= {"basket_type": basket_type,
                         "basket_id": basket_id,
                        "underlying_asset_basket_id": underlying_asset_basket_id,
                        "underlying_asset_basket": underlying_asset_basket,
                        "basket_trading_signal_list": basket_trading_signal_list,
                         "basket_desc": basket_desc,
                         "basket_label": basket_label,
                         "basket_columns": basket_columns,
                         "basket_timeseries_names": basket_timeseries_names
                         }

    return trading_signal_basket