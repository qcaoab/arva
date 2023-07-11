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

def read_log_file_const_prop(filepath_str):
    
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
            
        if item == 'ConstProp_strategy': 
            expected_wealth.append(float(split[i+21]))
            cvar_05.append(float(split[i+23]))
            median_wealth.append(float(split[i+19]))
            # function_value.append(float(split[i+11+1]))
            # qsum_avg.append(float(split[i+16]))
            
        
        
    df = pd.DataFrame({'kappa': kappa_list, 'cvar_05': cvar_05, 'expected_wealth': expected_wealth, 
    'median_wealth': median_wealth}) 
    # 'median': median_wealth, 'f_val': function_value})
    df.sort_values(by=['kappa'], ignore_index=True, inplace=True)

    return df


def read_log_file_test(filepath_str):
    
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
        
    df = pd.DataFrame({'cvar_05': cvar_05, 'expected_wealth': expected_wealth, 
    'median_wealth': median_wealth, 'qsum_avg': qsum_avg, 'p_med_avg': p_med_avg}) 
    # 'median': median_wealth, 'f_val': function_value})
    # df.sort_values(by=['kappa'], ignore_index=True, inplace=True)

    return df


# os.chdir("/home/marcchen/Documents/constrain_factor/researchcode/log_output_EXP1")


# # os.chdir("/home/ma3chen/Documents/marc_branch2/researchcode/log_output")
# files = [f for f in listdir(os.getcwd()) if isfile(join(os.getcwd(), f))]

# df_all = read_log_file(files[0])
# df_all['portfolio'] = files[0][0:-19]
# df_all = df_all[0:0]

# for file in files: 
    
#     df = read_log_file(file)
#     df['portfolio'] = file[0:-19]
    
#     df_all = df_all.append(df)

# check = 0


# read_log_file_const_prop(files[0])

# # df_all.to_csv("/home/marcchen/Documents/constrain_factor/researchcode/formatted_output/training_performance.csv")


# # os.chdir("/home/ma3chen/Documents/marc_branch2/researchcode/log_output")
# files = [f for f in listdir(os.getcwd()) if isfile(join(os.getcwd(), f))]

# df_all = read_log_file_const_prop(files[0])
# df_all['portfolio'] = files[0][0:-19]
# df_all = df_all[0:0]

# for file in files: 
    
#     df = read_log_file(file)
#     df['portfolio'] = file[0:-19]
    
#     df_all = df_all.append(df)

# # df_all.to_csv("/home/marcchen/Documents/constrain_factor/researchcode/formatted_output/const_performance.csv")


os.chdir("/home/marcchen/Documents/constrain_factor/researchcode/log_output_batchTEST1")


files = [f for f in listdir(os.getcwd()) if isfile(join(os.getcwd(), f))]

df_all = read_log_file_test(files[0])
df_all['portfolio'] = files[0][0:-19]
df_all = df_all[0:0]

for file in files: 
    
    df = read_log_file_test(file)
    df['portfolio'] = file[0:-19]
    
    df_all = df_all.append(df)
    
df_all.to_csv("/home/marcchen/Documents/constrain_factor/researchcode/formatted_output/TEST1_performance.csv")

