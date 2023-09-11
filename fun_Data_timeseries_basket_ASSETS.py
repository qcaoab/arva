
import pandas as pd
import numpy as np



def asset_basket(basket_type,   #"asset"
                 basket_id, #   Basket ID of pre-defined portfolios we want to consider, see below
                 real_or_nominal,     #whether nominal or real historical returns should be used.
                 add_cash_TrueFalse = False  #True if cash needs to be added as an asset
                ):
    #Constructs pre-defined asset baskets (groups of assets to incorporate in the portfolio)
    #RETURNS dictionary "asset_basket"

    asset_basket = {}

    #------------------------------------------------------------------------------------------------

    #Pre-defined portfolios; "basket_id" will be the identifier


    if basket_id == "Paper_FactorInv_Basic":
        #description
        basket_desc = "Basic portfolio for paper: T30, B10 and VWD"

        #SHORT label for e.g. figures
        basket_label = "T30, B10 and VWD"

        #timeseries names, ONE FOR EVERY ASSET to be used as identifiers (e.g. column headings PREFIX) for timeseries
        basket_timeseries_names = ["T30", "B10", "VWD"]

    elif basket_id == "Paper_FactorInv_Factor2":
        #description
        basket_desc = "Factor2 portfolio for paper: Basic, size and value"

        #SHORT label for e.g. figures
        basket_label = "Factor2 portfolio"

        #timeseries names, ONE FOR EVERY ASSET to be used as identifiers (e.g. column headings PREFIX) for timeseries
        basket_timeseries_names = ["T30", "B10", "VWD", "Size_Lo30", "Value_Hi30"]

    elif basket_id == "Paper_FactorInv_Factor3":
        #description
        basket_desc = "Factor3 portfolio for paper: Basic, size, value, vol"

        #SHORT label for e.g. figures
        basket_label = "Factor3 portfolio"

        #timeseries names, ONE FOR EVERY ASSET to be used as identifiers (e.g. column headings PREFIX) for timeseries
        basket_timeseries_names = ["T30", "B10", "VWD", "Size_Lo30", "Value_Hi30", "Vol_Lo20"]

    elif basket_id == "Paper_FactorInv_Factor3_noB10":
        #description
        basket_desc = "Factor3 portfolio for paper: Basic but no B10, size, value, vol"

        #SHORT label for e.g. figures
        basket_label = "Factor3 portfolio no B10"

        #timeseries names, ONE FOR EVERY ASSET to be used as identifiers (e.g. column headings PREFIX) for timeseries
        basket_timeseries_names = ["T30",  "VWD", "Size_Lo30", "Value_Hi30", "Vol_Lo20"]

    elif basket_id == "Paper_FactorInv_Factor4":
        #description
        basket_desc = "Factor4 portfolio for paper: Basic, size, value, vol, mom"

        #SHORT label for e.g. figures
        basket_label = "Factor4 portfolio"

        #timeseries names, ONE FOR EVERY ASSET to be used as identifiers (e.g. column headings PREFIX) for timeseries
        basket_timeseries_names = ["T30", "B10", "VWD", "Size_Lo30", "Value_Hi30", "Vol_Lo20", "Mom_Hi30"]

    
    
    elif basket_id == "4_factor_1927":
        #description
        basket_desc = "4 factors with data since 1927: Basic, size, value, div, mom"

        #SHORT label for e.g. figures
        basket_label = "4factor_1927"

        #timeseries names, ONE FOR EVERY ASSET to be used as identifiers (e.g. column headings PREFIX) for timeseries
        basket_timeseries_names = ["T30", "B10", "VWD", "Size_Lo30", "Value_Hi30", "Div_Hi30", "Mom_Hi30"]
        
        

    elif basket_id == "3factor_mc":
        #description
        basket_desc = "3 factors : Basic, size, value, mom"

        #SHORT label for e.g. figures
        basket_label = "3factors"

        #timeseries names, ONE FOR EVERY ASSET to be used as identifiers (e.g. column headings PREFIX) for timeseries
        basket_timeseries_names = ["T30", "B10", "VWD", "Size_Lo30", "Value_Hi30", "Mom_Hi30"]
        
    elif basket_id == "Paper_FactorInv_Factor4_noB10":
        #description
        basket_desc = "Factor4 portfolio for paper: Basic but no B10, size, value, vol, mom"

        #SHORT label for e.g. figures
        basket_label = "Factor4 portfolio no B10"

        #timeseries names, ONE FOR EVERY ASSET to be used as identifiers (e.g. column headings PREFIX) for timeseries
        basket_timeseries_names = ["T30", "VWD", "Size_Lo30", "Value_Hi30", "Vol_Lo20", "Mom_Hi30"]


    elif basket_id == "basic_T90_VWD":

        #description
        basket_desc = "CRSP data: T90 and VWD"

        #SHORT label for e.g. figures
        basket_label = "T90 and VWD"

        #timeseries names, ONE FOR EVERY ASSET to be used as identifiers (e.g. column headings PREFIX) for timeseries
        basket_timeseries_names = ["T90", "VWD"]

    elif basket_id == "basic_T90_B10_VWD":

        #description
        basket_desc = "CRSP data: T90, B10 and VWD"

        #SHORT label for e.g. figures
        basket_label = "T90, B10 and VWD"

        #timeseries names, ONE FOR EVERY ASSET to be used as identifiers (e.g. column headings PREFIX) for timeseries
        basket_timeseries_names = ["T90", "B10", "VWD"]

    elif basket_id == "basic_VWD_T90_Chendi":

        #description
        basket_desc = "CRSP data: VWD and T90"

        #SHORT label for e.g. figures
        basket_label = "VWD and T90"

        #timeseries names, ONE FOR EVERY ASSET to be used as identifiers (e.g. column headings PREFIX) for timeseries
        basket_timeseries_names = ["VWD", "T90"]


    elif basket_id == "basic_ForsythLi":

        #description
        basket_desc = "CRSP data: T30 and VWD"

        #SHORT label for e.g. figures
        basket_label = "T30 and VWD"

        #timeseries names, ONE FOR EVERY ASSET to be used as identifiers (e.g. column headings PREFIX) for timeseries
        basket_timeseries_names = ["T30", "VWD"]


    elif basket_id == "basic_T30_VWD":

        #description
        basket_desc = "CRSP data: T30 and VWD"

        #SHORT label for e.g. figures
        basket_label = "T30 and VWD"

        #timeseries names, ONE FOR EVERY ASSET to be used as identifiers (e.g. column headings PREFIX) for timeseries
        basket_timeseries_names = ["T30", "VWD"]
    
    #MC added:
    elif basket_id == "B10_and_VWD":

        #description
        basket_desc = "CRSP data: B10 and VWD"

        #SHORT label for e.g. figures
        basket_label = "B10 and VWD"

        #timeseries names, ONE FOR EVERY ASSET to be used as identifiers (e.g. column headings PREFIX) for timeseries
        basket_timeseries_names = ["B10", "VWD"]


    elif basket_id == "fake_assets":
        #description
        basket_desc = "Fake assets for testing"

        #SHORT label for e.g. figures
        basket_label = "Fake assets for testing"

        #timeseries names, ONE FOR EVERY ASSET to be used as identifiers (e.g. column headings PREFIX) for timeseries
        basket_timeseries_names = ["Asset_00", "Asset_01", "Asset_02", "Asset_03", "Asset_04", "Asset_05", "Asset_06"]


    elif basket_id == "basic_ForsythLi_plus_B10":

        #description
        basket_desc = "CRSP data: T30, VWD and B10"

        #SHORT label for e.g. figures
        basket_label = "T30, VWD and B10"

        #timeseries names, ONE FOR EVERY ASSET to be used as identifiers (e.g. column headings PREFIX) for timeseries
        basket_timeseries_names = ["T30", "VWD", "B10"]


    elif basket_id == "longshort_FF3":

        # description
        basket_desc = "FF 3 factor long-short"

        # SHORT label for e.g. figures
        basket_label = "FF 3 factor long-short"

        # timeseries names, ONE FOR EVERY ASSET to be used as identifiers (e.g. column headings PREFIX) for timeseries
        basket_timeseries_names = ["T30", "VWD", "FF3_SMB", "FF3_HML"]


    elif basket_id == "longshort_FF3_plus_MOM":

        # description
        basket_desc = "FF 3 factor long-short plus momentum"

        # SHORT label for e.g. figures
        basket_label = "FF 3 factor long-short plus momentum"

        # timeseries names, ONE FOR EVERY ASSET to be used as identifiers (e.g. column headings PREFIX) for timeseries
        basket_timeseries_names = ["T30", "VWD", "FF3_SMB", "FF3_HML", "FF_MOM"]


    elif basket_id == "longshort_FF5":

        # description
        basket_desc = "FF 5 factor long-short"

        # SHORT label for e.g. figures
        basket_label = "FF 5 factor long-short"

        # timeseries names, ONE FOR EVERY ASSET to be used as identifiers (e.g. column headings PREFIX) for timeseries
        basket_timeseries_names = ["T30", "VWD", "FF5_SMB", "FF5_HML", "FF5_RMW",	"FF5_CMA"]


    elif basket_id == "longshort_FF5_plus_MOM":

        # description
        basket_desc = "FF 5 factor long-short plus momentum"

        # SHORT label for e.g. figures
        basket_label = "FF 5 factor long-short plus momentum"

        # timeseries names, ONE FOR EVERY ASSET to be used as identifiers (e.g. column headings PREFIX) for timeseries
        basket_timeseries_names = ["T30", "VWD", "FF5_SMB", "FF5_HML", "FF5_RMW",	"FF5_CMA", "FF_MOM"]


    elif basket_id == "factortilt_FF3":

        # description
        basket_desc = "FF 3 factor tilt"

        # SHORT label for e.g. figures
        basket_label = "FF 3 factor tilt"

        # timeseries names, ONE FOR EVERY ASSET to be used as identifiers (e.g. column headings PREFIX) for timeseries
        basket_timeseries_names = ["T30",	"VWD",	"Size_Lo30",	"Value_Hi30"]

    elif basket_id == "factortilt_FF3_plus_LowVol":

        # description
        basket_desc = "FF 3 factor tilt plus LowVol"

        # SHORT label for e.g. figures
        basket_label = "FF 3 factor tilt plus LowVol"

        # timeseries names, ONE FOR EVERY ASSET to be used as identifiers (e.g. column headings PREFIX) for timeseries
        basket_timeseries_names = ["T30",	"VWD",	"Size_Lo30",	"Value_Hi30", "Vol_Lo20"]

    elif basket_id == "factortilt_FF3_plus_MOM":

        # description
        basket_desc = "FF 3 factor tilt plus momentum"

        # SHORT label for e.g. figures
        basket_label = "FF 3 factor tilt plus momentum"

        # timeseries names, ONE FOR EVERY ASSET to be used as identifiers (e.g. column headings PREFIX) for timeseries
        basket_timeseries_names = ["T30",	"VWD",	"Size_Lo30",	"Value_Hi30", "Mom_Hi30"]


    elif basket_id == "factortilt_FF5":

        # description
        basket_desc = "FF 5 factor tilt"

        # SHORT label for e.g. figures
        basket_label = "FF 5 factor tilt"

        # timeseries names, ONE FOR EVERY ASSET to be used as identifiers (e.g. column headings PREFIX) for timeseries
        basket_timeseries_names = ["T30",	"VWD",	"Size_Lo30",	"Value_Hi30", "Oprof_Hi30",	"Inv_Lo30"]


    elif basket_id == "factortilt_FF5_plus_MOM":

        # description
        basket_desc = "FF 5 factor tilt plus momentum"

        # SHORT label for e.g. figures
        basket_label = "FF 5 factor tilt plus momentum"

        # timeseries names, ONE FOR EVERY ASSET to be used as identifiers (e.g. column headings PREFIX) for timeseries
        basket_timeseries_names = ["T30",	"VWD",	"Size_Lo30",	"Value_Hi30", "Oprof_Hi30",	"Inv_Lo30", "Mom_Hi30"]


    elif basket_id == "factortilt_single_equalW_factor":

        # description
        basket_desc = "Basic ForsythLi plus EQWFact"

        # SHORT label for e.g. figures
        basket_label = "Basic ForsythLi plus EQWFact"

        # timeseries names, ONE FOR EVERY ASSET to be used as identifiers (e.g. column headings PREFIX) for timeseries
        basket_timeseries_names = ["T30",	"VWD",	"EQWFact"]


    elif basket_id == "factortilt_ALL":

        # description
        basket_desc = "Factor tilt using all factors"

        # SHORT label for e.g. figures
        basket_label = "Factor tilt using all factors"

        # timeseries names, ONE FOR EVERY ASSET to be used as identifiers (e.g. column headings PREFIX) for timeseries
        basket_timeseries_names = ["T30",	"VWD",	"Size_Lo30",	"Value_Hi30",	"Oprof_Hi30",
                                   "Inv_Lo30",	"Mom_Hi30",	"EP_Hi30",	"Vol_Lo20",	"Div_Hi30"]


    elif basket_id == "factortilt_ALL_plus_equalW_factor":

        # description
        basket_desc = "Factor tilt using all factors plus EQWFact"

        # SHORT label for e.g. figures
        basket_label = "Factor tilt using all factors plus EQWFact"

        # timeseries names, ONE FOR EVERY ASSET to be used as identifiers (e.g. column headings PREFIX) for timeseries
        basket_timeseries_names = ["T30",	"VWD",	"Size_Lo30",	"Value_Hi30",	"Oprof_Hi30",
                                   "Inv_Lo30",	"Mom_Hi30",	"EP_Hi30",	"Vol_Lo20",	"Div_Hi30", "EQWFact"]


    elif basket_id == "factortilt_ALL_plus_B10":

        # description
        basket_desc = "Factor tilt using all factors plus B10"

        # SHORT label for e.g. figures
        basket_label = "Factor tilt using all factors plus B10"

        # timeseries names, ONE FOR EVERY ASSET to be used as identifiers (e.g. column headings PREFIX) for timeseries
        basket_timeseries_names = ["T30", "VWD","B10",	"Size_Lo30",	"Value_Hi30",	"Oprof_Hi30",
                                   "Inv_Lo30",	"Mom_Hi30",	"EP_Hi30",	"Vol_Lo20",	"Div_Hi30"]

    elif basket_id == "factortilt_ALL_plus_B10_and_EQWFact":

        # description
        basket_desc = "Factor tilt using all factors plus B10"

        # SHORT label for e.g. figures
        basket_label = "Factor tilt using all factors plus B10"

        # timeseries names, ONE FOR EVERY ASSET to be used as identifiers (e.g. column headings PREFIX) for timeseries
        basket_timeseries_names = ["T30",	"VWD","B10",	"Size_Lo30",	"Value_Hi30",	"Oprof_Hi30",
                                   "Inv_Lo30",	"Mom_Hi30",	"EP_Hi30",	"Vol_Lo20",	"Div_Hi30", "EQWFact"]

    elif basket_id == "basic_FF":

        #description
        basket_desc = "FF data: RF and Market"

        #SHORT label for e.g. figures
        basket_label = "RF and Mkt"

        #timeseries names, ONE FOR EVERY ASSET to be used as identifiers (e.g. column headings PREFIX for timeseries
        basket_timeseries_names = ["FF_RF", "FF_Mkt"]

    elif basket_id == "VWD":
        #description
        basket_desc = "CRSP data: VWD"

        #SHORT label for e.g. figures
        basket_label = "VWD"

        #timeseries names, ONE FOR EVERY ASSET to be used as identifiers (e.g. column headings PREFIX) for timeseries
        basket_timeseries_names = ["VWD"]

    elif basket_id == "FF_Mkt":
        #description
        basket_desc = "FF data: Market"

        #SHORT label for e.g. figures
        basket_label = "FF Mkt"

        #timeseries names, ONE FOR EVERY ASSET to be used as identifiers (e.g. column headings PREFIX) for timeseries
        basket_timeseries_names = ["FF_Mkt"]

    elif basket_id == "MC_everything":
        
        #description
        basket_desc = "marc_test1_all_assets_longfactors"
        
        #SHORT label for e.g. figures
        basket_label = "marc test1 basket"
                
        #timeseries names, ONE FOR EVERY ASSET to be used as identifiers (e.g. column headings PREFIX) for timeseries
        basket_timeseries_names = ["Size_Lo30", "Value_Hi30","Oprof_Hi30", "Inv_Lo30", "Mom_Hi30", "EP_Hi30", "Vol_Lo20", "Div_Hi30", "EQWFact", "T30",	"T90", "B10", "VWD", "EWD"]
        
    elif basket_id == "MC_conservative":
        
        #description
        basket_desc = "marc_conservative_basket"
        
        #SHORT label for e.g. figures
        basket_label = "marc conservative asset basket"
                
        #timeseries names, ONE FOR EVERY ASSET to be used as identifiers (e.g. column headings PREFIX) for timeseries
        basket_timeseries_names = ["Vol_Lo20", "T90", "B10", "EWD"]
    
    elif basket_id == "7_Factor_plusEWD":
        
        #description
        basket_desc = "7_Factor_totalmax"
        
        #SHORT label for e.g. figures
        basket_label = "7_Factor_totalmax"
                
        #timeseries names, ONE FOR EVERY ASSET to be used as identifiers (e.g. column headings PREFIX) for timeseries
        basket_timeseries_names = ["Size_Lo30", "Value_Hi30","Oprof_Hi30", "Inv_Lo30", "Mom_Hi30", "EP_Hi30", "Vol_Lo20", "Div_Hi30",  "EQWFact","T30", "B10", "VWD", "EWD"]
    
    elif basket_id == "5_Factor_plusEWD":
        
        #description
        basket_desc = "5_Factor_plusEWD"
        
        #SHORT label for e.g. figures
        basket_label = "5_Factor_plusEWD"
                
        #timeseries names, ONE FOR EVERY ASSET to be used as identifiers (e.g. column headings PREFIX) for timeseries
        basket_timeseries_names = ["Size_Lo30", "Value_Hi30", "Mom_Hi30", "Vol_Lo20", "Div_Hi30",  "T30", "B10", "VWD", "EWD"]
    

        

    # ------------------------------------------------------------------------------------------------
    # Add CASH as an asset if needed
    if add_cash_TrueFalse:  #if add_cash_TrueFalse == True, append "Cash" to timeseries names
        basket_timeseries_names.insert(0,"Cash")    #Cash will always be the FIRST asset (asset [0] if it has to be included)

    #Get column names from historical data
    basket_columns =  asset_data_column_names(basket_type, basket_timeseries_names, real_or_nominal,
                                                   returns_or_indices = "returns")

    # ------------------------------------------------------------------------------------------------
    # Construct asset basket
    asset_basket = {"basket_type": basket_type,
                         "basket_id": basket_id,
                         "basket_desc": basket_desc,
                         "basket_label": basket_label,
                         "basket_columns": basket_columns,
                         "basket_timeseries_names": basket_timeseries_names
                         }

    return asset_basket


def asset_data_column_names( basket_type, #"asset"
                                basket_timeseries_names,
                               real_or_nominal,
                               returns_or_indices   #"returns" or "indices"
                                 ):

    #OBJECTIVE: returns data column names associated with timeseries in basket_timeseries_names according to the format:
    # where timeseries is the timeseries name
    # timeseries_nom_ret: timeseries nominal returns
    # timeseries_real_ret: timeseries real returns
    # timeseries_nom_ret_ind: index formed based on timeseries nominal returns
    # timeseries_real_ret_ind: index formed based on timeseries real returns

    #RETURNS: column_names = LIST
    if basket_type == "asset":
        if real_or_nominal == "nominal":
            if returns_or_indices == "returns":
                column_names = [i+"_nom_ret" for i in basket_timeseries_names]
            elif returns_or_indices == "indices":
                column_names = [i + "_nom_ret_ind" for i in basket_timeseries_names]

        elif real_or_nominal == "real":
            if returns_or_indices == "returns":
                column_names = [i+"_real_ret" for i in basket_timeseries_names]
            elif returns_or_indices == "indices":
                column_names = [i + "_real_ret_ind" for i in basket_timeseries_names]

    else:
        raise ValueError("PVS error in asset_data_column_names: only gives conventions for asset return columns.")


    return column_names
