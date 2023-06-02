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


training_performance_df = read_log_file("/home/marcchen/Documents/testing_pyt_decum/researchcode/bootstrap_trained_models_block12month/may29_block12_ef_2020datainitfor2019.txt")

#simulated test
df_cont = read_log_file("/home/marcchen/Documents/testing_pyt_decum/researchcode/bootstrap_trained_models_block12month/may30_testonsynth_12month.txt")
    
# forsyth_df = pd.read_csv("/home/marcchen/Documents/pytorch_decumulation_mc/researchcode/formatted_output/forsyth_a1_corrected.txt")

plt.clf()
fig, ax = plt.subplots()

ax.plot(df_cont["cvar_05"],df_cont["qsum_avg"], marker = 'x', markersize=3, mew=10, 
         color = "red",linewidth=2, label = "Out-of-distribution Test on Simulated Dataset")
# for i, val in enumerate(df_cont['kappa']):
#     plt.annotate(str(val), (df_cont["cvar_05"][i], df_cont['qsum_avg'][i]))

kappa_to_annotate = 0.5
label_idx = df_cont['kappa'].tolist().index(kappa_to_annotate)

#plot arrow
xytext = (0.4,0.56)
ax.annotate("Out-of-distribution \nTest on Simulated Dataset", 
            color= "red",
            fontweight="semibold",
            fontsize=14,
            xy=(df_cont["cvar_05"][label_idx]+10,df_cont["qsum_avg"][label_idx]-0.9),
            xytext=(0.4,0.51),    # fraction, fraction
            textcoords='figure fraction',
            horizontalalignment='center',
            verticalalignment='center')

ax.annotate('', 
            fontweight="medium",
            fontsize=14,
            xy=(df_cont["cvar_05"][label_idx]-5,df_cont["qsum_avg"][label_idx]-0.5),
            xytext=xytext,    # fraction, fraction
            textcoords='figure fraction',
            arrowprops=dict(arrowstyle="->", color="red"),
            bbox=dict(pad=2, facecolor="none", edgecolor="none"))

    
ax.plot(training_performance_df["cvar_05"],training_performance_df["qsum_avg"], marker='o', color="black", mew=3, fillstyle='none', markersize=4, linewidth=2, label = "Training Performance on Bootstrap Dataset")
for i, val in enumerate(training_performance_df['kappa']):
    plt.annotate(str(val), (training_performance_df["cvar_05"][i]+15, training_performance_df['qsum_avg'][i]+0.3))

#plot arrow
xytext = (0.7,0.8)
ax.annotate("Training Performance \n on Bootstrap Dataset \n (blocksize = 12 months)", 
                    fontweight="semibold",
                    fontsize=16,
                    xy=(training_performance_df["cvar_05"][label_idx]+10,training_performance_df["qsum_avg"][label_idx]+0.9),
                    xytext=(0.7,0.86),    # fraction, fraction
                    textcoords='figure fraction',
                    horizontalalignment='center',
                    verticalalignment='center')

ax.annotate('', 
            fontweight="medium",
            fontsize=14,
            xy=(training_performance_df["cvar_05"][label_idx]+5,training_performance_df["qsum_avg"][label_idx]+0.5),
            xytext=(0.65,0.80),    # fraction, fraction
            textcoords='figure fraction',
            arrowprops=dict(arrowstyle="->"),
            bbox=dict(pad=2, facecolor="none", edgecolor="none"))


bengen_x = -286.86
bengen_y = 40

ax.plot(bengen_x, bengen_y, marker = "*", color="blue", mew=2, fillstyle='none', markersize=10, linewidth=2, label = "Bengen (1994)Strategy")

#plot arrow
xytext = (0.7,0.3)
ax.annotate("Bengen (1994) \nStrategy", 
                    fontweight="semibold",
                    fontsize=16,
                    color="blue",
                    xy=(bengen_x, bengen_y),
                    xytext=(0.58,0.3),    # fraction, fraction
                    textcoords='figure fraction',
                    horizontalalignment='left',
                    verticalalignment='center')

ax.annotate('', 
            fontweight="medium",
            fontsize=14,
            color="blue",
            xy=(bengen_x+10, bengen_y),
            xytext=(0.58,0.3),    # fraction, fraction
            textcoords='figure fraction',
            arrowprops=dict(arrowstyle="->",color="blue"),
            bbox=dict(pad=2, facecolor="none", edgecolor="none", color="blue"))


# plt.title("DC Efficient Frontier: NN Control Trained on Bootstrapped Dataset")
ax.set_xlabel("Expected Shortfall", fontweight='bold', fontsize=20)
ax.set_ylabel("E[Average Withdrawal]", fontweight='bold', fontsize=20)
# plt.legend(loc='lower left')
ax.set_xlim([-770, 150])
ax.set_ylim([35, 65])
ax.spines[['right', 'top']].set_visible(False)
ax.tick_params(axis='both', which='major', labelsize=12)
plt.subplots_adjust(bottom=0.15)

plt.savefig('/home/marcchen/Documents/testing_pyt_decum/researchcode/bootstrap_trained_models_block12month/may28_ef_trainedon_block12month_bootandsim.pdf',format="pdf")

df_cont.to_csv("/home/marcchen/Documents/testing_pyt_decum/researchcode/bootstrap_trained_models_block12month/bootstrap_trained_12month_simulated_test.csv", float_format="%.3f")
training_performance_df.to_csv("/home/marcchen/Documents/testing_pyt_decum/researchcode/bootstrap_trained_models_block12month/bootstrap_trained_12month_training_performance.csv", float_format="%.3f")


# df_cont.to_excel("/home/marcchen/Documents/pytorch_decumulation_mc/researchcode/formatted_output/jan29_ef_rangetermsimple.xlsx")

# forsyth_df.to_excel("/home/marcchen/Documents/pytorch_decumulation_mc/researchcode/formatted_output/jan29_ef_rangetermsimple.xlsx")

