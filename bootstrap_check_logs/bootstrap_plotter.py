import seaborn as sns
import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
import os
from os import listdir
from os.path import isfile, join
import re





os.chdir("/home/marcchen/Documents/testing_pyt_decum/researchcode/bootstrap_check_logs/data")


files = [f for f in listdir(os.getcwd()) if isfile(join(os.getcwd(), f))]

bootstrap_results = {}

for file in files:
    
    with open(file, 'r') as f:
        text = f.read()
    
    split = re.split('\n| ', text)

    kappa_list = []
    cvar_05 = []
    expected_wealth = []
    median_wealth = []
    function_value = []
    qsum_avg = []
    p_med_avg = []
    
    block_size = split[split.index("block_size:")+1]

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
    
    bootstrap_results[block_size] = {'kappa': kappa_list, 
                                     'cvar_05': cvar_05, 
                                     'expected_wealth': expected_wealth, 
                                     'median_wealth': median_wealth,
                                     'qsum_avg': qsum_avg, 
                                     'p_med_avg': p_med_avg}
    
plt.clf()
fig, ax = plt.subplots()

synthetic_df = pd.read_csv("/home/marcchen/Documents/testing_pyt_decum/researchcode/feb13_PAPERMODEL_otherfiles/main_model_ef.csv")


ax.plot(synthetic_df["cvar_05"],synthetic_df["qsum_avg"],  marker='o', color="black", mew=3, fillstyle='none',
        markersize=4, linewidth=2, label = "Simulated training data")

label_idx = synthetic_df['kappa'].tolist().index(0.2)
    
ax.annotate(f"Simulated \n training data", 
            color= "black",
            fontweight="semibold",
            fontsize=12,
            xy=(synthetic_df["cvar_05"][label_idx]+2,synthetic_df["qsum_avg"][label_idx]-0.2),
            xytext=(0.3,0.50),    # fraction, fraction
            textcoords='figure fraction',
            horizontalalignment='center',
            verticalalignment='center')

ax.annotate('', 
            fontweight="medium",
            fontsize=14,
            xy=(synthetic_df["cvar_05"][label_idx]-2,synthetic_df["qsum_avg"][label_idx]-0.2),
            xytext=(0.3,0.55),    # fraction, fraction
            textcoords='figure fraction',
            arrowprops=dict(arrowstyle="->", color="black"),
            bbox=dict(pad=2, facecolor="none", edgecolor="none"))


colors = ["red", "green", "royalblue"]
markers = ["s", "s", "s"]
kappa_ann = [0.5, 3, 5] 
x_text = [0.65, 0.84, 0.6]
y_text = [0.77, 0.65, 0.35]
x_arrow = [0.65, 0.84, 0.6]
y_arrow = [0.78, 0.66, 0.35]
ha = ['center', 'center', 'right']
va = ['bottom', 'bottom', 'center']

for i, block_size in enumerate(['1','3','12']):
           
    df_cont = pd.DataFrame(bootstrap_results[block_size]) 
    # 'median': median_wealth, 'f_val': function_value})
    df_cont.sort_values(by=['kappa'], ignore_index=True, inplace=True)
    
    ax.plot(df_cont["cvar_05"],df_cont["qsum_avg"],  marker=markers[i], color=colors[i], mew=2, 
        markersize=3, linewidth=2, label = f"Bootstrap testing data (blocksize={block_size} months)")
    
    #plot label and arrow
    label_idx = df_cont['kappa'].tolist().index(kappa_ann[i])
    
    ax.annotate(f"Bootstrap testing data \n (blocksize={block_size} months)", 
                color= colors[i],
                fontweight="semibold",
                fontsize=12,
                xy=(df_cont["cvar_05"][label_idx]+10,df_cont["qsum_avg"][label_idx]-0.9),
                xytext=(x_text[i],y_text[i]),    # fraction, fraction
                textcoords='figure fraction',
                horizontalalignment=ha[i],
                verticalalignment=va[i])

    ax.annotate('', 
                fontweight="medium",
                fontsize=12,
                xy=(df_cont["cvar_05"][label_idx],df_cont["qsum_avg"][label_idx]),
                xytext=(x_arrow[i],y_arrow[i]),    # fraction, fraction
                textcoords='figure fraction',
                arrowprops=dict(arrowstyle="->", color=colors[i]),
                bbox=dict(pad=2, facecolor="none", edgecolor="none"))


#This plot is for the NN trained on sythetic data (2.56*10^5 paths, tested on the same dataset vs bootstrapped datasets of different sizes)
# plt.title("DC Efficient Frontier: NN Control Results on Bootstrapped Historical Data")
ax.set_xlabel("Expected Shortfall",fontweight='bold', fontsize=20)
ax.set_ylabel("E[Average Withdrawal]", fontweight='bold', fontsize=20)
# plt.legend(loc='lower left')
ax.set_xlim([-650, 150])
ax.set_ylim([35, 65])
ax.spines[['right', 'top']].set_visible(False)
ax.tick_params(axis='both', which='major', labelsize=12)
plt.subplots_adjust(bottom=0.15)

# plt.show()

plt.savefig('/home/marcchen/Documents/testing_pyt_decum/researchcode/bootstrap_check_logs/bootstrap_comparison_mar8.pdf', format="pdf")

for i, block_size in enumerate(['1','3','12']):
           
    df_cont = pd.DataFrame(bootstrap_results[block_size]) 
    # 'median': median_wealth, 'f_val': function_value})
    df_cont.sort_values(by=['kappa'], ignore_index=True, inplace=True)
    df_cont.to_csv(f"/home/marcchen/Documents/testing_pyt_decum/researchcode/bootstrap_check_logs/bootstrap_test_ef_{block_size}.csv", float_format="%.3f")

# synthetic_df.to_csv(f"/home/marcchen/Documents/testing_pyt_decum/researchcode/bootstrap_check_logs/synthetic_train_ef_mainmodel.csv", float_format="%.3f")

# forsyth_df.to_excel("/home/marcchen/Documents/pytorch_decumulation_mc/researchcode/formatted_output/jan29_ef_rangetermsimple.xlsx")

