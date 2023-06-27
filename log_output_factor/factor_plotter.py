import seaborn as sns
import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
import os
from os import listdir
from os.path import isfile, join
import re


#"/home/marcchen/Documents/testing_pyt_decum/researchcode/bootstrap_trained_models_3month/mar9_ef_3month_bootstrap.txt"

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


df_2asset = pd.read_excel("/home/marcchen/Documents/testing_pyt_decum/researchcode/bootstrap_trained_models_block12month/trainingperformance_bootstrapef_mar7.xlsx")

#simulated test
df_multi = read_log_file("/home/marcchen/Documents/factor_decumulation/researchcode/2factor_main_model/jun9_2factor_rerun.txt")

df_all_asset = read_log_file("/home/marcchen/Documents/factor_decumulation/researchcode/log_output_factor/jun11_allasset_rerun.txt")


plt.clf()
fig, ax = plt.subplots()

ax.plot(df_multi["cvar_05"],df_multi["qsum_avg"], marker = 'x', markersize=3, mew=10, 
         color = "red",linewidth=2, label = "5 Asset/Factor, 6mo")

ax.plot(df_all_asset["cvar_05"],df_multi["qsum_avg"], marker = 'x', markersize=3, mew=10, 
         color = "green",linewidth=2, label = "All Asset/Factor, 6mo")
# for i, val in enumerate(df_multi['kappa']):
#     plt.annotate(str(val), (df_multi["cvar_05"][i], df_multi['qsum_avg'][i]))

kappa_to_annotate = 0.5
label_idx = df_multi['kappa'].tolist().index(kappa_to_annotate)

# #plot arrow
# xytext = (0.4,0.55)
# ax.annotate("Multi-asset", 
#             color= "red",
#             fontweight="semibold",
#             fontsize=14,
#             xy=(df_multi["cvar_05"][label_idx]+10,df_multi["qsum_avg"][label_idx]-0.9),
#             xytext=(0.4,0.51),    # fraction, fraction
#             textcoords='figure fraction',
#             horizontalalignment='center',
#             verticalalignment='center')

# ax.annotate('', 
#             fontweight="medium",
#             fontsize=14,
#             xy=(df_multi["cvar_05"][label_idx]-5,df_multi["qsum_avg"][label_idx]-0.5),
#             xytext=xytext,    # fraction, fraction
#             textcoords='figure fraction',
#             arrowprops=dict(arrowstyle="->", color="red"),
#             bbox=dict(pad=2, facecolor="none", edgecolor="none"))

    
ax.plot(df_2asset["cvar_05"],df_2asset["qsum_avg"], marker='o', color="black", mew=2, fillstyle='none', markersize=10, linewidth=2, label = "2 asset, 12mo")
for i, val in enumerate(df_2asset['kappa']):
    plt.annotate(str(val), (df_2asset["cvar_05"][i]+15, df_2asset['qsum_avg'][i]+0.3))

# #plot arrow
# xytext = (0.7,0.8)
# ax.annotate("Original 2 Asset", 
#                     fontweight="semibold",
#                     fontsize=16,
#                     xy=(df_2asset["cvar_05"][label_idx]+10,df_2asset["qsum_avg"][label_idx]+0.9),
#                     xytext=(0.7,0.86),    # fraction, fraction
#                     textcoords='figure fraction',
#                     horizontalalignment='center',
#                     verticalalignment='center')

# ax.annotate('', 
#             fontweight="medium",
#             fontsize=14,
#             xy=(df_2asset["cvar_05"][label_idx]+5,df_2asset["qsum_avg"][label_idx]+0.5),
#             xytext=(0.65,0.80),    # fraction, fraction
#             textcoords='figure fraction',
#             arrowprops=dict(arrowstyle="->"),
#             bbox=dict(pad=2, facecolor="none", edgecolor="none"))

#bengen point

# plt.title("DC Efficient Frontier: NN Control Trained on Bootstrapped Dataset")
ax.set_xlabel("Expected Shortfall", fontweight='bold', fontsize=20)
ax.set_ylabel("E[Average Withdrawal]", fontweight='bold', fontsize=20)
plt.legend(loc='lower left')
# ax.set_xlim([-770, 150])
# ax.set_ylim([35, 65])
ax.spines[['right', 'top']].set_visible(False)
ax.tick_params(axis='both', which='major', labelsize=12)
plt.subplots_adjust(bottom=0.15)

plt.savefig("/home/marcchen/Documents/factor_decumulation/researchcode/formatted_output/multi_2asset_comparison_ef.png", format = "png")


# df_multi.to_csv("/home/marcchen/Documents/testing_pyt_decum/researchcode/bootstrap_trained_models_3month/bootstrap_trained_3month_simulated_test.csv", float_format="%.3f")
# df_2asset.to_csv("/home/marcchen/Documents/testing_pyt_decum/researchcode/bootstrap_trained_models_3month/bootstrap_trained_3month_training_performance.csv", float_format="%.3f")



# forsyth_df.to_excel("/home/marcchen/Documents/pytorch_decumulation_mc/researchcode/formatted_output/jan29_ef_rangetermsimple.xlsx")

