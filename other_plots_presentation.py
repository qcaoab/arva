import seaborn as sns
import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
import os
from os import listdir
from os.path import isfile, join
import re
from matplotlib.ticker import (MultipleLocator, AutoMinorLocator)



#plan: labeled pctile plots for new kappa=1, for VWD, wealth, and withdrawals. 

 
pct_data = pd.read_excel(r"/home/marcchen/Documents/constrain_factor/researchcode/output_heatmaps_constrain/2_factor_constrain_indiv07-07-23_19:07/1.0timestamp_2023-07-07_23_51_Pctiles_ALL.xlsx")
pct_data = pd.melt(pct_data, id_vars='Unnamed: 0')


def plot_pct_format(pctile_list, year_labels, xy_text_list, nice_name, colors, ha,va,type):
    
    plt.clf()
    fig, ax = plt.subplots()
    
    for data_label, year, xy_text, nice_name, color, ha, va in \
        zip(pctile_list, year_labels, xy_text_list, nice_name, colors, ha, va):
        
        #get data
        plot_data = pct_data[pct_data['Unnamed: 0']==data_label]
        plot_data['variable'] = pd.to_numeric(plot_data['variable'])
        
        #plot 
        ax.plot(plot_data['variable'], plot_data['value'], color=color)
        
    #annotate median
        if type != "assets":
            color = "Black"
        ax.plot(year,plot_data[plot_data['variable']==year]['value'])
        ax.annotate(nice_name, 
                    fontweight="bold",
                    fontsize=14,
                    xy=(year,plot_data[plot_data['variable']==year]['value']),
                    xytext=xy_text,    # fraction, fraction
                    textcoords='figure fraction',
                    horizontalalignment=ha,
                    verticalalignment=va,
                    color=color)

        ax.annotate('', 
                    fontweight="bold",
                    fontsize=14,
                    xy=(year,plot_data[plot_data['variable']==year]['value']),
                    xytext=xy_text,    # fraction, fraction
                    textcoords='figure fraction',
                    arrowprops=dict(arrowstyle="->", color=color),
                    bbox=dict(pad=0, facecolor="none", edgecolor="none"))

        if type =="assets":
            ax.set_ylim(0.,1.02) #make slightly higher than 1.0 so we can see if all wealth is invested in an asset
            ax.set_xlim(0.,30)
            ax.set_ylabel("Allocation in Asset at Median Wealth", fontsize=14, fontweight="bold")
            ax.set_xlabel("Time (years)", fontsize=14,fontweight="bold")
            # plt.title("NN: Percentiles proportion in asset VWD", fontsize=11, fontweight="bold")
            
            #tick marks
            ax.xaxis.set_major_locator(MultipleLocator(5))
            ax.xaxis.set_major_formatter('{x:.0f}')
            ax.xaxis.set_minor_locator(MultipleLocator(1))
            
            ax.yaxis.set_major_locator(MultipleLocator(0.1))
            ax.yaxis.set_minor_locator(MultipleLocator(0.02))

        if type =="wealth":
            
            ax.set_xlim(0.,30)
            ax.set_ylim(0.,2000)
            ax.set_ylabel("Wealth (Thousands)", fontsize=11,fontweight="bold")
            ax.set_xlabel("Time (years)", fontsize=11,fontweight="bold")
            
            #tick marks
            ax.xaxis.set_major_locator(MultipleLocator(5))
            ax.xaxis.set_major_formatter('{x:.0f}')
            ax.xaxis.set_minor_locator(MultipleLocator(1))
            
            ax.yaxis.set_major_locator(MultipleLocator(500))
            ax.yaxis.set_minor_locator(MultipleLocator(100))
        
        if type =="withdraw":
            
            ax.set_xlim(0.,30)
            ax.set_ylim(0.,100.)
            ax.set_ylabel("Withdrawals (thousands)", fontsize=11,fontweight="bold")
            ax.set_xlabel("Time (years)", fontsize=11,fontweight="bold")
            
            #tick marks
            ax.xaxis.set_major_locator(MultipleLocator(5))
            ax.xaxis.set_major_formatter('{x:.0f}')
            ax.xaxis.set_minor_locator(MultipleLocator(1))
            
            ax.yaxis.set_major_locator(MultipleLocator(10))
            ax.yaxis.set_minor_locator(MultipleLocator(2))
            

        
    plt.savefig(f"/home/marcchen/Documents/constrain_factor/researchcode/formatted_output/pctile_plots_2fac_indiv/{type}_pctile_plot.pdf")

# generate asset plot--------

#label positions by year, 95, 50, 05
pctile_list = ['T30_pctile_50', 'B10_pctile_50', 'VWD_pctile_50', 'Size_Lo30_pctile_50', 'Value_Hi30_pctile_50']
nice_name = ['T30', 'B10', 'Mkt', 'Size', 'Value']
year_labels = [20, 3, 12,15, 20]
xy_text_list = [(0.6, 0.8), (0.2, 0.7), (0.34, 0.15), (0.49, 0.22), (0.7, 0.4)]
colors = ["Black", "Red", "Blue", "Green", "Purple"]
ha = ["center", "center", "right", "right","center"]
va = ["bottom", "bottom", "center", "bottom", "bottom"]

plot_pct_format(pctile_list, year_labels, xy_text_list, nice_name, colors, ha, va, type="assets")


# generate wealth plot--------

#label positions by year, 95, 50, 05
pctile_list = ['Wealth_pctile_95', 'Wealth_pctile_50', 'Wealth_pctile_5']
nice_name = ['95th percentile', 'Median', '5th percentile']
year_labels = [10, 21, 12]
xy_text_list = [(0.33, 0.8), (0.68, 0.5), (0.4, 0.2)]
colors = ["darkorange", "Red", "Black"]
ha = ["center", "center", "right"]
va = ["bottom", "bottom", "center"]


plot_pct_format(pctile_list, year_labels, xy_text_list, nice_name, colors, ha, va, type="wealth")


# generate withdrawal plot--------

#label positions by year, 95, 50, 05
pctile_list = ['Withdrawals_pctile_95', 'Withdrawals_pctile_50', 'Withdrawals_pctile_5']
nice_name = ['95th percentile', 'Median', '5th percentile']
year_labels = [3, 5, 12]
xy_text_list = [(0.27, 0.7), (0.3, 0.45), (0.4, 0.2)]
colors = ["Blue", "Red", "Green"]
ha = ["center", "left", "right"]
va = ["bottom", "center", "center"]

plot_pct_format(pctile_list, year_labels, xy_text_list, nice_name, colors, ha, va, type="withdraw")
