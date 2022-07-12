import seaborn as sns
import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
import os
from os import listdir
from os.path import isfile, join
import re

### read kappa logs and plot efficient frontier


os.chdir("/home/ma3chen/Documents/marc_branch2/researchcode/log_output")
files = [f for f in listdir(os.getcwd()) if isfile(join(os.getcwd(), f))]


kappa_list = []
cvar_05 = []
expected_wealth = []
median_wealth = []
function_value = []

for file in files:

    with open(file, 'r') as f:
        text = f.read()

    split = re.split('\n| ', text)

    split = split[split.index("FINISHED:")-18:]


    for i, item in enumerate(split):

        if item == "param:":
            kappa_list.append(float(split[i+1]))
        # if item == "value:":
        #     function_value.append(float(split[i+1]))
        if item == "W_T_mean:":
            expected_wealth.append(float(split[i+1]))
        # if item == "W_T_median:":
        #     median_wealth.append(float(split[i+1]))
        if item == "W_T_CVAR_5_pct:":
            cvar_05.append(float(split[i+1]))


df = pd.DataFrame({'kappa': kappa_list, 'cvar_05': cvar_05, 'expected_wealth': expected_wealth}) 
# 'median': median_wealth, 'f_val': function_value})
df.sort_values(by=['kappa'], ignore_index=True, inplace=True)



# import forsyth data
forsyth_df = pd.read_excel(r'/home/ma3chen/Documents/marc_branch2/researchcode/forsyth_efficient_frontier_data.xlsx')


# plt.show()
plt.plot(forsyth_df["EW"], forsyth_df["ES"],marker='o', label = "Forsyth 2022 results")

plt.plot(df["expected_wealth"], df["cvar_05"], marker = 'o', label = "MC NN replication")

plt.title("Efficient frontier results: Forsyth, Vetza 2022 vs. MC NN approximation")
plt.xlabel("Expected Wealth")
plt.ylabel("Expected Shortfall (cvar 0.05)")
plt.legend()

plt.show()

# plt.savefig('/home/ma3chen/Documents/marc_branch2/researchcode/output/efficient_frontier_plot.png')

df.to_excel("/home/ma3chen/Documents/marc_branch2/researchcode/output/mc_efficient_frontier.xlsx")
