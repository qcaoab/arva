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
        plt.annotate('Inf', (forsyth_df["ES"][i]+20, forsyth_df['Sum q_i/(M+1)'][i]+0.25))
    else:
        plt.annotate(str(val), (forsyth_df["ES"][i]+20, forsyth_df['Sum q_i/(M+1)'][i]+0.3))

kappa_to_annotate = 0.5
label_idx = forsyth_df['kappa'].tolist().index(kappa_to_annotate)

#plot arrow
xytext = (0.7,0.8)
ax.annotate("PDE Control", 
                    fontweight="semibold",
                    fontsize=20,
                    xy=(forsyth_df["ES"][label_idx]+10,forsyth_df['Sum q_i/(M+1)'][label_idx]+0.9),
                    xytext=(0.72,0.82),    # fraction, fraction
                    textcoords='figure fraction',
                    horizontalalignment='center',
                    verticalalignment='center')

ax.annotate('', 
            fontweight="medium",
            fontsize=14,
            xy=(forsyth_df["ES"][label_idx]+5,forsyth_df['Sum q_i/(M+1)'][label_idx]+0.5),
            xytext=xytext,    # fraction, fraction
            textcoords='figure fraction',
            arrowprops=dict(arrowstyle="->"),
            bbox=dict(pad=2, facecolor="none", edgecolor="none"))


ax.plot(df_cont["cvar_05"],df_cont["qsum_avg"],  marker = 'x', markersize=3, mew=10, 
         color = "red", label = "NN Control",linewidth=2)
# for i, val in enumerate(df_cont['kappa']):
#     plt.annotate(str(val), (df_cont["cvar_05"][i], df_cont['qsum_avg'][i]))

#plot arrow
xytext = (0.5,0.5)
ax.annotate("NN Control", 
                    fontweight="semibold",
                    color="red",
                    fontsize=20,
                    xy=(df_cont["cvar_05"][label_idx]-5,df_cont["qsum_avg"][label_idx]-0.5),
                    xytext=xytext,    # fraction, fraction
                    textcoords='figure fraction',
                    horizontalalignment='center',
                    verticalalignment='top')

ax.annotate('', 
            fontweight="bold",
            fontsize=14,
            color = "red",
            xy=(forsyth_df["ES"][label_idx]-20,forsyth_df['Sum q_i/(M+1)'][label_idx]),
            xytext=xytext,    # fraction, fraction
            textcoords='figure fraction',
            arrowprops=dict(arrowstyle="->", color="red"),
            bbox=dict(pad=2, facecolor="none", edgecolor="none"))


# plt.title("EW-ES Frontiers: ")
ax.set_xlabel("Expected Shortfall", fontweight='semibold',fontsize=20)
ax.set_ylabel("E[Average Withdrawal]", fontweight='semibold',fontsize=20)
# ax.legend(loc='lower left')
ax.set_xlim([-670, 100])
ax.set_ylim([30, 65])
ax.spines[['right', 'top']].set_visible(False)
ax.tick_params(axis='both', which='major', labelsize=12)
plt.subplots_adjust(bottom=0.15)
plt.savefig('/home/marcchen/Documents/testing_pyt_decum/researchcode/feb13_PAPERMODEL_otherfiles/feb13_log_output_yyfeb3_big_PAPERMODEL.pdf',format="pdf")


# df_cont.to_excel("/home/marcchen/Documents/testing_pyt_decum/researchcode/formatted_output/feb13_log_output_yyfeb3_big_PAPERMODEL.xlsx")

# forsyth_df.to_excel("/home/marcchen/Documents/pytorch_decumulation_mc/researchcode/formatted_output/jan29_ef_rangetermsimple.xlsx")

