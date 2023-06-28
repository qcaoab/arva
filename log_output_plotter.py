import seaborn as sns
import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
import os
from os import listdir
from os.path import isfile, join
import re

### read kappa logs and plot efficient frontier

#/home/marcchen/Documents/pytorch_1/researchcode/log_output/dec5_ef_fine_cont_test.txt


with open("/home/marcchen/Documents/pytorch_1/researchcode/log_output/dec5_ef_fine_cont_test.txt", 'r') as f:
    text = f.read()

split = re.split('\n| ', text)
split = [x for x in split if x !='']

kappa_list = []
cvar_05 = []
expected_wealth = []
median_wealth = []
function_value = []


for i, item in enumerate(split):
    if item == 'NN-strategy-on-TRAINING':
       
        if split[i+46] != "not":
            kappa_list.append(float(split[i+46]))
        else:
            kappa_list.append(float(split[i+70]))
            
        expected_wealth.append(float(split[i+3]))
        cvar_05.append(float(split[i+9]))
        median_wealth.append(float(split[i+5]))
        # function_value.append(float(split[i+11+1]))
 
    
df_cont = pd.DataFrame({'kappa': kappa_list, 'cvar_05': cvar_05, 'expected_wealth': expected_wealth, 
'median_wealth': median_wealth}) 
# 'median': median_wealth, 'f_val': function_value})
df_cont.sort_values(by=['kappa'], ignore_index=True, inplace=True)


# os.chdir("/home/ma3chen/Documents/marc_branch2/researchcode/log_output")
# files = [f for f in listdir(os.getcwd()) if isfile(join(os.getcwd(), f))]


# kappa_list = []
# cvar_05 = []
# expected_wealth = []
# median_wealth = []
# function_value = []

# for file in files:

#     with open(file, 'r') as f:
#         text = f.read()

#     split = re.split('\n| ', text)

#     split = split[split.index("FINISHED:")-18:]


#     for i, item in enumerate(split):

#         if item == "param:":
#             kappa_list.append(float(split[i+1]))
#         # if item == "value:":
#         #     function_value.append(float(split[i+1]))
#         if item == "W_T_mean:":
#             expected_wealth.append(float(split[i+1]))
#         # if item == "W_T_median:":
#         #     median_wealth.append(float(split[i+1]))
#         if item == "W_T_CVAR_5_pct:":
#             cvar_05.append(float(split[i+1]))


# df = pd.DataFrame({'kappa': kappa_list, 'cvar_05': cvar_05, 'expected_wealth': expected_wealth}) 
# # 'median': median_wealth, 'f_val': function_value})
# df.sort_values(by=['kappa'], ignore_index=True, inplace=True)



# import forsyth data
forsyth_df = pd.read_csv(r'/home/marcchen/Documents/pytorch_1/researchcode/formatted_output/forsyth_cutyourlosses_results_E1.csv')


# plt.show()
plt.plot(forsyth_df["ES(W_T)"],forsyth_df["E(W_T)"], marker='o', label = "Forsyth 2022 results")

plt.plot(df_cont["cvar_05"], df_cont["expected_wealth"], marker = 's', label = "MC NN replication")

plt.title("Cut Your Losses EF: Forsyth, Vetza 2022 vs. MC NN approximation")
plt.ylabel("Expected Wealth")
plt.xlabel("Expected Shortfall (cvar 0.05)")
plt.legend()

plt.show()

plt.savefig('/home/marcchen/Documents/pytorch_1/researchcode/formatted_output/dec7_cutyourlosses_ef.png')

# df.to_excel("/home/marcchen/Documents/pytorch_1/researchcode/formatted_output/mc_efficient_frontier.xlsx")
