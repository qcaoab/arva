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

# forsyth_df = pd.read_csv("/home/marcchen/Documents/pytorch_decumulation_mc/researchcode/formatted_output/forsyth_a1_corrected.txt")




training_performance_df = read_log_file("/home/marcchen/Documents/testing_pyt_decum/researchcode/bootstrap_trained_models_3month/mar9_ef_3month_bootstrap.txt")

simulated_test_df = read_log_file("/home/marcchen/Documents/testing_pyt_decum/researchcode/bootstrap_trained_models_3month/mar14_ef_runon_synthetic.txt")


plt.clf()
plt.plot(simulated_test_df["cvar_05"],simulated_test_df["qsum_avg"],  marker = 's', label = "Out-of-distribution Test on Simulated Dataset")

plt.plot(training_performance_df["cvar_05"],training_performance_df["qsum_avg"],  marker = 'o', label = "Training Performance on Bootstrap Dataset")
for i, val in enumerate(training_performance_df['kappa']):
    plt.annotate(str(val), (training_performance_df["cvar_05"][i]+7, training_performance_df['qsum_avg'][i]+0.2))



# plt.title("DC Efficient Frontier: NN Control Trained on Bootstrapped Dataset")
plt.xlabel("Expected Shortfall (cvar 0.05)", fontweight='bold')
plt.ylabel("Expected Average Withdrawals", fontweight='bold')
plt.legend(loc='lower left')
plt.xlim([-650, 150])
plt.ylim([40, 65])

plt.show()

plt.savefig('/home/marcchen/Documents/testing_pyt_decum/researchcode/bootstrap_trained_models_3month/mar9_ef_trainedon_block3month_bootandsim.png', dpi = 200)



# df_cont.to_excel("/home/marcchen/Documents/pytorch_decumulation_mc/researchcode/formatted_output/jan29_ef_rangetermsimple.xlsx")

# forsyth_df.to_excel("/home/marcchen/Documents/pytorch_decumulation_mc/researchcode/formatted_output/jan29_ef_rangetermsimple.xlsx")

