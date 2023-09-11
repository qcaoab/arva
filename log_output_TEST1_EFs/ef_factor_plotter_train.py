import seaborn as sns
import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
import os
from os import listdir
from os.path import isfile, join
import re





def read_log_file(filepath_str):
    
    with open(filepath_str, 'r') as f:
        text = f.read()

    split = re.split('\n| ', text)

    kappa_list = []
    cvar_05 = []
    expected_wealth = []
    median_wealth = []
    function_value = []
    qsum_avg = []
    p_med_avg = []


    for i, item in enumerate(split):
        if item == 'param:':
            kappa_list.append(float(split[i+1]))
            
        if item == 'NN-strategy-on-TRAINING': 
            expected_wealth.append(float(split[i+5]))
            cvar_05.append(float(split[i+11]))
            median_wealth.append(float(split[i+7]))
            # function_value.append(float(split[i+11+1]))
            qsum_avg.append(float(split[i+16]))
            
        if item == 'p_equity:':
            p_med_avg.append(float(split[i+2]))
        
    df = pd.DataFrame({'kappa': kappa_list, 'cvar_05': cvar_05, 'expected_wealth': expected_wealth, 
    'median_wealth': median_wealth, 'qsum_avg': qsum_avg, 'p_med_avg': p_med_avg}) 
    # 'median': median_wealth, 'f_val': function_value})
    df.sort_values(by=['kappa'], ignore_index=True, inplace=True)

    return df


factor2_df = read_log_file("/home/marcchen/Documents/constrain_factor/researchcode/log_output_TEST1_EFs/2_factor_constrain_indiv_log_08-06-2023 combined.txt")

#simulated test
basic_df = read_log_file("/home/marcchen/Documents/constrain_factor/researchcode/log_output_TEST1_EFs/2basic_NN_log_08-06-2023 combined.txt")
    
# forsyth_df = pd.read_csv("/home/marcchen/Documents/pytorch_decumulation_mc/researchcode/formatted_output/forsyth_a1_corrected.txt")

plt.clf()
fig, ax = plt.subplots()

ax.plot(factor2_df["cvar_05"],factor2_df["qsum_avg"], marker = 'o', markersize=4, mew=3, 
         color = "red",linewidth=2, label = "2 Factor, limit_type=indiv")

# kappa labels
# for i, val in enumerate(factor2_df['kappa']):
#     if val == 999.:
#         plt.annotate(r"$\infty$", (factor2_df["cvar_05"][i]+15, factor2_df['qsum_avg'][i]+0.3), fontsize = 14)
#     elif val != 25.:  
#         plt.annotate(str(val), (factor2_df["cvar_05"][i]+15, factor2_df['qsum_avg'][i]+0.3))

kappa_to_annotate = 1.0
label_idx = factor2_df['kappa'].tolist().index(kappa_to_annotate)

#plot arrow
xytext = (0.7,0.65)
ax.annotate("NN: 2 Factor, \nlimit_type=indiv", 
            color= "red",
            fontweight="semibold",
            fontsize=14,
            xy=(factor2_df["cvar_05"][label_idx]+10,factor2_df["qsum_avg"][label_idx]-0.9),
            xytext=(0.77,0.75),    # fraction, fraction
            textcoords='figure fraction',
            horizontalalignment='center',
            verticalalignment='bottom')

ax.annotate('', 
            fontweight="medium",
            fontsize=14,
            xy=(factor2_df["cvar_05"][label_idx]+5,factor2_df["qsum_avg"][label_idx]+0.5),
            xytext=(0.77,0.75),    # fraction, fraction
            textcoords='figure fraction',
            arrowprops=dict(arrowstyle="->", color="red"),
            bbox=dict(pad=2, facecolor="none", edgecolor="none"))

    
ax.plot(basic_df["cvar_05"],basic_df["qsum_avg"], marker='o', color="black", mew=3, fillstyle='none', markersize=4, linewidth=2, label = "Training Performance on Bootstrap Dataset")

# kappa labels
# for i, val in enumerate(basic_df['kappa']):
#     if val == 999.:
#         plt.annotate(r"$\infty$", (basic_df["cvar_05"][i]+15, basic_df['qsum_avg'][i]+0.3), fontsize = 14)
#     else:  
#         plt.annotate(str(val), (basic_df["cvar_05"][i]+15, basic_df['qsum_avg'][i]+0.3))


kappa_to_annotate = 0.5
label_idx = basic_df['kappa'].tolist().index(kappa_to_annotate)

#plot arrow
xytext = (0.7,0.8)
ax.annotate("NN: Basic 2 Asset", 
                    fontweight="semibold",
                    fontsize=16,
                    xy=(basic_df["cvar_05"][label_idx]+10,basic_df["qsum_avg"][label_idx]+0.9),
                    xytext=(0.4,0.8),    # fraction, fraction
                    textcoords='figure fraction',
                    horizontalalignment='center',
                    verticalalignment='bottom')

ax.annotate('', 
            fontweight="medium",
            fontsize=14,
            xy=(basic_df["cvar_05"][label_idx]+5,basic_df["qsum_avg"][label_idx]+0.5),
            xytext=(0.4,0.8),    # fraction, fraction
            textcoords='figure fraction',
            arrowprops=dict(arrowstyle="->"),
            bbox=dict(pad=2, facecolor="none", edgecolor="none"))

#bengen, basic constant
bengen_x = -379
bengen_y = 40

ax.plot(bengen_x, bengen_y, marker = "*", color="blue", mew=2, fillstyle='none', markersize=10, linewidth=2, label = "Constant: Basic 2 Asset (Bengen, 1994)")

#plot arrow
xytext = (0.7,0.3)
ax.annotate("Constant: \nBasic 2 Asset", 
                    fontweight="semibold",
                    fontsize=14,
                    color="blue",
                    xy=(bengen_x, bengen_y),
                    xytext=(0.25,0.45),    # fraction, fraction
                    textcoords='figure fraction',
                    horizontalalignment='center',
                    verticalalignment='bottom')

ax.annotate('', 
            fontweight="medium",
            fontsize=14,
            color="blue",
            xy=(bengen_x-10, bengen_y),
            xytext=(0.25,0.45),    # fraction, fraction
            textcoords='figure fraction',
            arrowprops=dict(arrowstyle="->",color="blue"),
            bbox=dict(pad=2, facecolor="none", edgecolor="none", color="blue"))

## constant factor

bengen_x = -357
bengen_y = 40

ax.plot(bengen_x, bengen_y, marker = "s", color="darkorange", mew=2, fillstyle='full', markersize=5, linewidth=2, label = "Constant: 2 Factor Asset Basket")

#plot arrow
xytext = (0.7,0.3)
ax.annotate("Constant: \n2 Factor Basket", 
                    fontweight="semibold",
                    fontsize=14,
                    color="darkorange",
                    xy=(bengen_x, bengen_y),
                    xytext=(0.44,0.35),    # fraction, fraction
                    textcoords='figure fraction',
                    horizontalalignment='center',
                    verticalalignment='bottom')

ax.annotate('', 
            fontweight="medium",
            fontsize=14,
            color="darkorange",
            xy=(bengen_x+5, bengen_y+0.2),
            xytext=(0.44,0.35),    # fraction, fraction
            textcoords='figure fraction',
            arrowprops=dict(arrowstyle="->",color="darkorange"),
            bbox=dict(pad=2, facecolor="none", edgecolor="none", color="blue"))


# plt.title("DC Efficient Frontier: NN Control Trained on Bootstrapped Dataset")
ax.set_xlabel("Expected Shortfall", fontweight='bold', fontsize=20)
ax.set_ylabel("E[Average Withdrawal]", fontweight='bold', fontsize=20)
# plt.legend(loc='lower left')
ax.set_xlim([-680, 300])
ax.set_ylim([35, 65])
ax.spines[['right', 'top']].set_visible(False)
ax.tick_params(axis='both', which='major', labelsize=12)
plt.subplots_adjust(bottom=0.15)

plt.savefig('/home/marcchen/Documents/constrain_factor/researchcode/formatted_output/factor_ef_comp_DS1_train.pdf',format="pdf")

# df_cont.to_csv("/home/marcchen/Documents/testing_pyt_decum/researchcode/bootstrap_trained_models_block12month/bootstrap_trained_12month_simulated_test.csv", float_format="%.3f")
# training_performance_df.to_csv("/home/marcchen/Documents/testing_pyt_decum/researchcode/bootstrap_trained_models_block12month/bootstrap_trained_12month_training_performance.csv", float_format="%.3f")


# df_cont.to_excel("/home/marcchen/Documents/pytorch_decumulation_mc/researchcode/formatted_output/jan29_ef_rangetermsimple.xlsx")

# forsyth_df.to_excel("/home/marcchen/Documents/pytorch_decumulation_mc/researchcode/formatted_output/jan29_ef_rangetermsimple.xlsx")

