import seaborn as sns
import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
import os
from os import listdir
from os.path import isfile, join
import re

os.chdir("/home/marcchen/Documents/pytorch_decumulation_mc/researchcode/log_output_yy2_parallel_3pt")

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
    split = [x for x in split if x !='']
    
    for i, item in enumerate(split):
        if item == 'NN-strategy-on-TRAINING':
       
            kappa_list.append(float(split[i+44]))
            expected_wealth.append(float(split[i+5]))
            cvar_05.append(float(split[i+11]))
            median_wealth.append(float(split[i+7]))
            # function_value.append(float(split[i+11+1]))
            qsum_avg.append(float(split[i+15]))
            p_med_avg.append(float(split[i+24])) 

 
    
df_cont = pd.DataFrame({'kappa': kappa_list, 'cvar_05': cvar_05, 'expected_wealth': expected_wealth, 
'median_wealth': median_wealth, 'qsum_avg': qsum_avg, 'p_med_avg': p_med_avg}) 
# 'median': median_wealth, 'f_val': function_value})
df_cont.sort_values(by=['kappa'], ignore_index=True, inplace=True)

forsyth_df = pd.read_csv("/home/marcchen/Documents/pytorch_decumulation_mc/researchcode/formatted_output/forsyth_a1_corrected.txt")


plt.clf()
plt.plot(forsyth_df["ES"],forsyth_df["Sum q_i/(M+1)"], marker='o', label = "Forsyth PDE results")
for i, val in enumerate(forsyth_df['kappa']):
    plt.annotate(str(val), (forsyth_df["ES"][i], forsyth_df['Sum q_i/(M+1)'][i]))

plt.plot(df_cont["cvar_05"],df_cont["qsum_avg"],  marker = 's', label = "MC NN replication")
for i, val in enumerate(df_cont['kappa']):
    plt.annotate(str(val), (df_cont["cvar_05"][i], df_cont['qsum_avg'][i]))


plt.title("DC Efficient frontier results: Forsyth 2021 vs. MC NN approximation")
plt.xlabel("Expected Shortfall (cvar 0.05)")
plt.ylabel("Expected Average Withdrawals")
plt.legend(loc='lower left')

plt.show()

plt.savefig('/home/marcchen/Documents/pytorch_decumulation_mc/researchcode/formatted_output/log_output_yy2_parallel.png', dpi = 200)


# df_cont.to_excel("/home/marcchen/Documents/pytorch_decumulation_mc/researchcode/formatted_output/jan29_ef_rangetermsimple.xlsx")

# forsyth_df.to_excel("/home/marcchen/Documents/pytorch_decumulation_mc/researchcode/formatted_output/jan29_ef_rangetermsimple.xlsx")

