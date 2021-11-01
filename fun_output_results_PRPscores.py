
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import datetime
import pandas as pd
import copy

def get_df_PRP_Pctile_paths( params # dictionary with parameters and results from NN investment (TRAINING or TESTING)
                        ):
    #Objective: appends the dataframe of percentiles of PRP scores over time to params
    # - NO plotting or excel output here

    #Returns, params: ADDED field: params["df_PRP_pctiles"]


    params = copy.deepcopy(params)


    df_PRP_pctiles = output_results_PRPscores(params = params)   #use defaults to get just dataframe


    #Add extra fields to params
    params["df_PRP_pctiles"] = df_PRP_pctiles

    return params


def output_results_PRPscores(
        params,  # params dictionary with *trained* NN results and setup as in main code
        nr_bins=20, #nr of bins for PRP scores distribution heatmap
        pctiles_list = None,  #E.g. [20,50,80] List of percentiles to output and/or plot
        output_Excel=False,  # write the result to Excel
        filename_prefix_for_Excel="z_PRP_",  # used if output_Excel is True
        save_Figures=False,  # Saves figures in format specified below
        save_Figures_format="png",
        fig_filename_prefix="z_PRP_",
        xticklabels=12,  # e.g. xticklabels=6 means we are displaying only every 6th xaxis label to avoid overlapping
        yticklabels=1,  # e.g. yticklabels = 500 means we are displaying every 500th label
        cmap="Reds"  # e.g. "Reds" or "rainbow" etc colormap for sns.heatmap
        ):

    #OUTPUTS: if save_Figures is True:
    # heatmaps of PRP scores distribution over time
    # Excel spreadsheet with percentiles of PRP scores over time


    if pctiles_list is None:
        pctiles_list = [20,50,80]    #set defaults

    N_phi = params["N_phi"] #nr of features
    N_rb = params["N_rb"]   #nr of rebalancing events
    PRPscores = params["PRPscores"] #calculated PRP scores on the training/testing dataset
    delta_t = params["delta_t"]
    feature_order = params["feature_order"] #get list of feature names, in order

    #Create bin edges for all histograms of PRP scores
    bin_edges = np.linspace(start=0.0, stop=1.0,num= nr_bins + 1, endpoint=True )
    left_edges = bin_edges[:-1] #Just left endpoints, excluding right endpoint of the last bin

    # Get rebalancing times
    t_n_grid = delta_t * (np.arange(start=0, stop=params["N_rb"], step=1))  # rebalancing events index grid
    t_n_grid = t_n_grid.tolist()

    #Output dictionary for Excel spreadsheet
    dict_pctiles = {}  # initialize
    dict_pctiles.update({"t_n_grid": t_n_grid})

    for phi_index in np.arange(0,N_phi,1):  #Loop through features

        feature_name = feature_order[phi_index]

        #Initialize collection of histograms of PRPscores for this feature
        hists_PRPscores_phi = np.zeros([nr_bins, N_rb])

        #Percentiles over time of this PRP score
        pctiles_PRPscores_phi = np.zeros([len(pctiles_list), N_rb])


        for n_index in np.arange(0, N_rb,1):    #loop through rebalancing times

            #Get histogram of PRP scores for feature phi_index and rebal time n_index
            hist, _ = np.histogram(PRPscores[:,n_index,phi_index], bins=bin_edges)

            pctiles_phi = np.percentile(PRPscores[:,n_index,phi_index], pctiles_list)

            #Standardize the histogram
            hist_standardized = hist/np.sum(hist)

            #Append to collection of histograms of PRPscores for this feature
            hists_PRPscores_phi[:, n_index] = hist_standardized

            #Append to pctiles of PRP scores of this feature over time
            pctiles_PRPscores_phi[:, n_index] = pctiles_phi

        #End of loop over rebalancing times
        #- Still working with a specific feature



        if save_Figures:
            # --------------------------- Construct heatmap  ---------------------------
            # Avoids cutting off the xlabel and sides of box
            from matplotlib import rcParams

            rcParams.update({'figure.autolayout': True})


            left_edges_list = np.round(left_edges, 2).tolist()

            plt.figure()
            sns.set(font_scale=0.85)
            sns.set_style("ticks")

            # Set title and colorbar label
            plt.title("PRP scores of feature: " + feature_name)
            cbar_kws_label = "Percentage of PRP scores in bin"

            h = sns.heatmap(data= hists_PRPscores_phi,
                            yticklabels=left_edges_list[::yticklabels],
                            xticklabels=t_n_grid[::xticklabels],  # time_to_go,
                            vmin=0.0, vmax=0.8,
                            cbar_kws={'label': cbar_kws_label},
                            cmap=cmap, cbar=True)
            h.set_xticks(np.arange(0, len(t_n_grid), xticklabels))
            h.set_yticks(np.arange(0, len(left_edges_list), yticklabels))
            h.invert_yaxis()

            plt.ylabel('PRP score bins')
            plt.xlabel('Rebalancing time')


            # Save figure
            timestamp = datetime.datetime.now().strftime('%Y-%m-%d_%H_%M')

            fig_filename = fig_filename_prefix + "timestamp_" + timestamp + "_PRPscores_feature_" \
                           + str(phi_index) + "_" +  feature_name \
                           + "." + save_Figures_format

            plt.savefig(fig_filename, format=save_Figures_format, bbox_inches="tight")

            # plt.show()
            plt.close(fig="all")

        # --------------------------- Percentiles of PRP scores over time  ---------------------------


        for pctile_index in np.arange(0, len(pctiles_list),1) :
            key = feature_name + "_PRPscore_" + str(pctiles_list[pctile_index]) + "th_pctile"
            val = pctiles_PRPscores_phi[pctile_index, :]

            dict_pctiles.update({key : val})


    #End loop over features

    # ------------------------------------------------------------------------------------------------
    # OUTPUT

    df_dict_pctiles = pd.DataFrame.from_dict(dict_pctiles,  orient="index")
    df_dict_pctiles.sort_index(axis=0, ascending=True,inplace=True)

    if output_Excel:

        timestamp = datetime.datetime.now().strftime('%Y-%m-%d_%H_%M')
        filename = filename_prefix_for_Excel  + "timestamp_" + timestamp + "_PRPscores_Pctiles_ALL"

        df_dict_pctiles.to_excel(filename + ".xlsx")


    return df_dict_pctiles