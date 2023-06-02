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

oos_test_df = read_log_file("/home/marcchen/Documents/testing_pyt_decum/researchcode/out_of_sample_tests/mar16_ef_oos_test.txt")

# training performance, main model 

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
    
    
training_performance_df = pd.DataFrame({'kappa': kappa_list, 'cvar_05': cvar_05, 'expected_wealth': expected_wealth, 
'median_wealth': median_wealth, 'qsum_avg': qsum_avg, 'p_med_avg': p_med_avg}) 
# 'median': median_wealth, 'f_val': function_value})
training_performance_df.sort_values(by=['kappa'], ignore_index=True, inplace=True)

#filter out inf
# training_performance_df = training_performance_df[training_performance_df['kappa']!= 999]


plt.clf()

fig, ax = plt.subplots()

ax.plot(oos_test_df["cvar_05"],oos_test_df["qsum_avg"],  marker = 's', markersize=7, linewidth=2, color = "red", label = "Out-of-Sample Test")


kappa_to_annotate = 0.5
label_idx = oos_test_df['kappa'].tolist().index(kappa_to_annotate)

#plot arrow
xytext = (0.5,0.4)
ax.annotate("Out-of-Sample Test" , 
                    color = "red",
                    fontweight="semibold",
                    fontsize=20,
                    xy=(oos_test_df["cvar_05"][label_idx],oos_test_df["qsum_avg"][label_idx]),
                    xytext=xytext,    # fraction, fraction
                    textcoords='figure fraction',
                    horizontalalignment='center',
                    verticalalignment='top')

ax.annotate('', 
            fontweight="medium",
            fontsize=14,
            color = "red",
            xy=(oos_test_df["cvar_05"][label_idx],oos_test_df["qsum_avg"][label_idx]),
            xytext=xytext,    # fraction, fraction
            textcoords='figure fraction',
            arrowprops=dict(arrowstyle="->",color="red"),
            bbox=dict(pad=2, facecolor="none", edgecolor="none"))


ax.plot(training_performance_df["cvar_05"],training_performance_df["qsum_avg"], marker='o', color="black", mew=1.5, fillstyle='none',  markersize=7, linewidth=2, label = "Training Performance on Bootstrap Dataset")
for i, val in enumerate(training_performance_df['kappa']):
    plt.annotate(str(val), (training_performance_df["cvar_05"][i]+7, training_performance_df['qsum_avg'][i]+0.2))


#plot arrow
xytext = (0.7,0.8)
ax.annotate("Training Performance", 
                    fontweight="semibold",
                    color="black",
                    fontsize=20,
                    xy=(training_performance_df["cvar_05"][label_idx],training_performance_df["qsum_avg"][label_idx]-0.5),
                    xytext=xytext,    # fraction, fraction
                    textcoords='figure fraction',
                    horizontalalignment='center',
                    verticalalignment='bottom')

ax.annotate('', 
            fontweight="bold",
            fontsize=14,
            color = "black",
            xy=(training_performance_df["cvar_05"][label_idx],training_performance_df["qsum_avg"][label_idx]),
            xytext=xytext,    # fraction, fraction
            textcoords='figure fraction',
            arrowprops=dict(arrowstyle="->", color="black"),
            bbox=dict(pad=2, facecolor="none", edgecolor="none"))


# plt.title("DC Efficient Frontier: ")
ax.set_xlabel("Expected Shortfall", fontweight='semibold',fontsize=20)
ax.set_ylabel("E[Average Withdrawal]", fontweight='semibold',fontsize=20)
# ax.legend(loc='lower left')
ax.set_xlim([-670, 100])
ax.set_ylim([35, 65])
ax.spines[['right', 'top']].set_visible(False)
ax.tick_params(axis='both', which='major', labelsize=12)
plt.subplots_adjust(bottom=0.15)

plt.show()

plt.savefig("/home/marcchen/Documents/testing_pyt_decum/researchcode/out_of_sample_tests/june2_oos_comparison.pdf", format="pdf")

oos_test_df.to_csv("/home/marcchen/Documents/testing_pyt_decum/researchcode/out_of_sample_tests/oos_test.csv", float_format="%.3f")
training_performance_df.to_csv("/home/marcchen/Documents/testing_pyt_decum/researchcode/out_of_sample_tests/oos_training.csv", float_format="%.3f")

# df_cont.to_excel("/home/marcchen/Documents/pytorch_decumulation_mc/researchcode/formatted_output/jan29_ef_rangetermsimple.xlsx")

# forsyth_df.to_excel("/home/marcchen/Documents/pytorch_decumulation_mc/researchcode/formatted_output/jan29_ef_rangetermsimple.xlsx")

