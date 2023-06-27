
import pandas as pd
import numpy as np
import math
import matplotlib.pyplot as plt
import seaborn as sns
import datetime

#Import files needed (other files are imported within those files as needed)
import fun_Data__assign
import fun_Data_timeseries_basket
import fun_Data_read_and_process_market_data
import fun_Data_bootstrap
import os
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import Normalizer
from sklearn.cluster import KMeans
import math

import plotly.graph_objects as go


os.chdir("/home/marcchen/Documents/factor_decumulation/researchcode")

#Initialize empty dictionaries
params = {}
data_settings = {}

save_Figures = True             #If true, save figures in format save_Figures_format
save_Figures_format = "png"     #format to save figures in

output_Excel = True #If True, will save the underlying data to Excel spreadsheets

periods_yyyymm_start_end = {0 : [196307, 202212],
                            1 : [196307,200912],
                            2 : [201001,202212]}


#-----------------------------------------------------------------------------------------------
# Specify market data we want to analyze
#-----------------------------------------------------------------------------------------------
params["asset_basket_id"] = "MC_everything"    #"Paper_FactorInv_Factor2" #Pre-defined basket of underlying candidate assets - see fun_Data_assets_basket.py
params["add_cash_TrueFalse"] = True     #If True, add "Cash" as an asset to the selected asset basket
params["real_or_nominal"] = "real" # "real" or "nominal" for asset data for wealth process: if "real", the asset data will be deflated by CPI
#real or nominal for TRADING SIGNALS will be set below



#-----------------------------------------------------------------------------------------------
# Underlying assets: Read and process market data
#-----------------------------------------------------------------------------------------------

#Construct asset basket:
# - this will also give COLUMN NAMES in the historical returns data to use
params["asset_basket"] = fun_Data_timeseries_basket.timeseries_basket_construct(
                            basket_type="asset",
                            basket_id=params["asset_basket_id"],
                            add_cash_TrueFalse=params["add_cash_TrueFalse"],
                            real_or_nominal = params["real_or_nominal"] )

#Assign number of assets based on basket information:
params["N_a"] = len(params["asset_basket"]["basket_columns"])   #Nr of assets = nr of output nodes




#IMPORTANT:
#   Market data provided always assumed to be NOMINAL returns
#   - if params["real_or_nominal"] = "real", the inflation-adjusted returns time series will be constructed here
#   Training and testing datasets (constructed subsequently) will use SUBSETS of the yyyymm_start and yyyymm_end selected here


for period in periods_yyyymm_start_end.keys():

    yyyymm_start = periods_yyyymm_start_end[period][0] #Start date to use for historical market data, set to None for data set start
    yyyymm_end = periods_yyyymm_start_end[period][1] #End date to use for historical market data, set to None for data set end

    # Market data file import and process settings
    data_settings["data_read_yyyymm_start"] = yyyymm_start  #Start date to use for historical market data, set to None for data set start
    data_settings["data_read_yyyymm_end"] = yyyymm_end    #End date to use for historical market data, set to None for data set end
    data_settings["data_read_input_folder"] = 'Market_data'
    data_settings["data_read_input_file"] = "_PVS_ALLfactors_CRSP_FF_data_202304_MCfactors"
    data_settings["data_read_input_file_type"] = ".xlsx"  #suffix
    data_settings["data_read_delta_t"] = 1 / 12 #time interval for returns data (monthly returns means data_delta_t=1/12)
    data_settings["data_read_returns_format"] = "percentages"  #'percentages' = already multiplied by 100 but without added % sign
                                                        #'decimals' is percentages in decimal form
    data_settings["data_read_skiprows"] = 15   #nr of rows of file to skip before start reading
    data_settings["data_read_index_col"] = 0 #Column INDEX of file with yyyymm to use as index
    data_settings["data_read_header"] = 0 #INDEX of row AFTER "skiprows" to use as column names
    data_settings["data_read_na_values"] = "nan" #how missing values are identified in the data

    data_settings["real_or_nominal"] = params["real_or_nominal"]    #if "real", will process the (nominal) market data
                                                                     # to obtain inflation-adjusted  returns

    # Read and process data
    data_returns = fun_Data_read_and_process_market_data.read_and_process_market_data(data_settings = data_settings,
                                                                                      timeseries_basket=params["asset_basket"])


    #Append historical data and associated key stats (mean, stdev, corr matrix) to asset_basket
    params["asset_basket"] = fun_Data_timeseries_basket.timeseries_basket_append_info(data_df = data_returns,
                                                                             timeseries_basket= params["asset_basket"])

    
    #plot densities of assets in groups
    
    def plot_asset_hist(asset_names, x_range):
        for asset_name in asset_names:
            ax = params["asset_basket"]['data_df'][asset_name].plot.hist(bins=100, alpha=0.5, label=asset_name)
            ax.legend()
            ax.set_xlim(x_range)
            fig = ax.get_figure()
            fig.savefig(f'/home/marcchen/Documents/factor_decumulation/researchcode/market_analysis_output/{asset_names}.png')
        fig.clf()
        ax.clear()
    
    # plot_asset_hist(['T30_real_ret', 'T90_real_ret', 'B10_real_ret'], [-0.05, 0.05])
    # plot_asset_hist(['Div_Hi30_real_ret', 'EQWFact_real_ret'], [-0.3, 0.3])
    
    #-----------------------------------------------------------------------------------------------
    # Get key results from market data
    #-----------------------------------------------------------------------------------------------
    asset_names = params["asset_basket"]["basket_timeseries_names"]
    asset_column_names =  params["asset_basket"]["basket_columns"]

    data_df = params["asset_basket"]["data_df"]
    data_df_mean = params["asset_basket"]["data_df_mean"]
    data_df_stdev = params["asset_basket"]["data_df_stdev"]
    data_df_corr = params["asset_basket"]["data_df_corr"]
    data_df_sharpe = data_df_mean/data_df_stdev


    #Saving underlying data to Excel
    if output_Excel:
        prefix = "market_analysis_output/z_Returns_set_" + str(period) + "_"
        suffix = "_" + str(yyyymm_start) + "_to_" + str(yyyymm_end)

        data_df.to_excel(prefix + "data_df" + suffix + ".xlsx")
        data_df_mean.to_excel(prefix + "data_df_mean" + suffix + ".xlsx")
        data_df_stdev.to_excel(prefix + "data_df_stdev" + suffix + ".xlsx")
        data_df_corr.to_excel(prefix + "data_df_corr" + suffix + ".xlsx")

    # Create Kmeans model
    kmeans = KMeans(n_clusters = 7,max_iter = 1000)# Make a pipeline chaining normalizer and kmeans

    df_t = data_df.transpose()
    kmeans.fit(df_t)
    labels = kmeans.predict(df_t)
    df1 = pd.DataFrame({'labels':labels,'companies':list(data_df.columns)}).sort_values(by=['labels'],axis = 0)
    
    #plot?
    from sklearn.decomposition import PCA
    # 
    # # Reduce the data
    reduced_data = PCA(n_components = 2)# Create Kmeans model
    kmeans = KMeans(n_clusters = 4,max_iter = 1000)# Make a pipeline chaining normalizer, pca and kmeans
    pipeline = make_pipeline(reduced_data,kmeans)# Fit pipeline to daily stock movements
    pipeline.fit(df_t)# Prediction
    labels = pipeline.predict(df_t)# Create dataframe to store companies and predicted labels
    df2 = pd.DataFrame({'labels':labels,'companies':list(data_df.columns)}).sort_values(by=['labels'],axis = 0)

   
    
    # Reduce the data
    reduced_data = PCA(n_components = 2).fit_transform(df_t)# Define step size of mesh
    h = 0.01
    
    # Plot the decision boundary
    x_min,x_max = reduced_data[:,0].min()-1, reduced_data[:,0].max() + 1
    y_min,y_max = reduced_data[:,1].min()-1, reduced_data[:,1].max() + 1
    xx,yy = np.meshgrid(np.arange(x_min,x_max,h),np.arange(y_min,y_max,h))
    
    # Obtain labels for each point in the mesh using our trained model
    # kmeans.fit(reduced_data)
    Z = kmeans.predict(np.c_[xx.ravel(),yy.ravel()])
    
    # Put the result into a color plot
    Z = Z.reshape(xx.shape)
    
    # Define color plot
    cmap = plt.cm.Paired
    
    # Plotting figure
    plt.clf()
    plt.figure(figsize=(10,10))
    plt.imshow(Z,interpolation = 'nearest',extent=(xx.min(),xx.max(),yy.min(),yy.max()),cmap = cmap,aspect = 'auto',origin = 'lower')
    plt.plot(reduced_data[:,0],reduced_data[:,1],'k.',markersize = 5)
    
    # Plot the centroid of each cluster as a white X
    centroids = kmeans.cluster_centers_
    plt.scatter(centroids[:,0],centroids[:,1],marker = 'x',s = 169,linewidths = 3,color = 'w',zorder = 10)
    plt.title('K-Means clustering on stock market movements (PCA-Reduced data)')
    plt.xlim(x_min,x_max)
    plt.ylim(y_min,y_max)
    plt.savefig("/home/marcchen/Documents/factor_decumulation/researchcode/market_analysis_output/clustering.png")


    #Scatterplot
    from matplotlib.ticker import PercentFormatter

    fig, ax = plt.subplots()

    for asset_index in np.arange(0,params["N_a"],1):

        #Plot and annotate points individually
        label = asset_names[asset_index]
        column_name = asset_column_names[asset_index]

        if label != "Cash":
            x =data_df_stdev[column_name]*100
            y = data_df_mean[column_name]*100
            ax.scatter(x,y)
            ax.annotate(label, (x, y))
            # ax.annotate(label, (x,y),
            #             xytext=(0,10), # distance from text to points (x,y)
            #             ha='center' ) # horizontal alignment can be left, right or center
            ax.xaxis.set_major_formatter(PercentFormatter(decimals = 2))
            ax.yaxis.set_major_formatter(PercentFormatter(decimals = 2))

            #Add line with Sharpe ratio
            #ax.plot([0.0,x], [0.0,y], "grey", linestyle=":")


    plt.xlabel("Stdev (monthly returns)")
    plt.ylabel("ExpVal (monthly returns)")
    plt.title("[Stdev, Mean] of monthly returns for equity indices: " +  str(yyyymm_start) + " to " + str(yyyymm_end) )


    if save_Figures:  # Save plots
        fig_filename = "z_Fig_Scatter_StdevMean_" + \
                       str(yyyymm_start) + "_to_" + str(yyyymm_end) + "." + save_Figures_format
        plt.savefig("market_analysis_output/" + fig_filename, format=save_Figures_format)

