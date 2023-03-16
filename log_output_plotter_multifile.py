import seaborn as sns
import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
import os
from os import listdir
from os.path import isfile, join
import re

os.chdir("/home/marcchen/Documents/testing_pyt_decum/researchcode/feb13_log_output_yyfeb3_big_PAPERMODEL")


# os.chdir("/home/ma3chen/Documents/marc_branch2/researchcode/log_output")
files = [f for f in listdir(os.getcwd()) if isfile(join(os.getcwd(), f))]

kappa_list = []
cvar_05 = []
expected_wealth = []
median_wealth = []
function_value = []
qsum_avg = []
p_med_avg = []

for file in files:
    
    with open(file, 'r') as f:
        text = f.read()
    
    split = re.split('\n| ', text)
    
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
    
    
df_cont = pd.DataFrame({'kappa': kappa_list, 'cvar_05': cvar_05, 'expected_wealth': expected_wealth, 
'median_wealth': median_wealth, 'qsum_avg': qsum_avg, 'p_med_avg': p_med_avg}) 
# 'median': median_wealth, 'f_val': function_value})
df_cont.sort_values(by=['kappa'], ignore_index=True, inplace=True)

forsyth_df = pd.read_csv("/home/marcchen/Documents/testing_pyt_decum/researchcode/formatted_output/forsyth_a1_corrected.txt")


plt.clf()

fig, ax = plt.subplots()

ax.plot(forsyth_df["ES"],forsyth_df["Sum q_i/(M+1)"], marker='o', color="black", mew=2, fillstyle='none', 
         markersize=10, label = "PDE Control", linewidth=2)
for i, val in enumerate(forsyth_df['kappa']):
    if val == 999:
        plt.annotate('Inf', (forsyth_df["ES"][i]-55, forsyth_df['Sum q_i/(M+1)'][i]-0.8))
    else:
        plt.annotate(str(val), (forsyth_df["ES"][i]-55, forsyth_df['Sum q_i/(M+1)'][i]-0.8))

ax.plot(df_cont["cvar_05"],df_cont["qsum_avg"],  marker = '+', markersize=3, mew=10, 
         color = "red", label = "NN Control",linewidth=2)
# for i, val in enumerate(df_cont['kappa']):
#     plt.annotate(str(val), (df_cont["cvar_05"][i], df_cont['qsum_avg'][i]))


# plt.title("EW-ES Frontiers: ")
ax.set_xlabel("Expected Shortfall (cvar 0.05)", fontweight='bold',fontsize=14)
ax.set_ylabel("Expected Average Withdrawals", fontweight='bold',fontsize=14)
ax.legend(loc='lower left')
ax.set_xlim([-680, 100])
ax.set_ylim([30, 60])
ax.spines[['right', 'top']].set_visible(False)

plt.savefig('/home/marcchen/Documents/testing_pyt_decum/researchcode/formatted_output/feb13_log_output_yyfeb3_big_PAPERMODEL.pdf',format="pdf")


# df_cont.to_excel("/home/marcchen/Documents/testing_pyt_decum/researchcode/formatted_output/feb13_log_output_yyfeb3_big_PAPERMODEL.xlsx")

# forsyth_df.to_excel("/home/marcchen/Documents/pytorch_decumulation_mc/researchcode/formatted_output/jan29_ef_rangetermsimple.xlsx")

