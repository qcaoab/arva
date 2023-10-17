import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import json
import copy

def plot_from_results_dict(results_dict, out_dir):

    exp_details = results_dict['exp_details'] 
    constant_results = results_dict['constant_benchmark_results']

    dict_to_plot = copy.deepcopy(results_dict)

    del dict_to_plot['exp_details']
    del dict_to_plot['constant_benchmark_results']

    nn_results = pd.DataFrame.from_dict(dict_to_plot, orient='index')
    nn_results.sort_values(by=['kappa'], ignore_index=True, inplace=True)

    
    fig, ax = plt.subplots()

    # Plot NN results
    ax.plot(nn_results["W_T_CVAR_5_pct"],nn_results["avg_withdrawal"], marker = 's', label = "NN Result")
    # for i, val in enumerate(nn_results['kappa']):
    #     plt.annotate(str(val), (nn_results["cvar_05"][i], nn_results['qsum_avg'][i]))

    # Plot forsyth results for comparison:
    forsyth_df = pd.read_csv("formatted_output/forsyth_a1_corrected.txt")
    ax.plot(forsyth_df["ES"],forsyth_df["Sum q_i/(M+1)"], marker='o', label = "PDE Result")
    for i, val in enumerate(forsyth_df['kappa']):
        ax.annotate(str(val), (forsyth_df["ES"][i], forsyth_df['Sum q_i/(M+1)'][i]))

    # Plot constant benchmark
    ax.plot(constant_results["W_T_CVAR_5_pct"],constant_results["constant withdrawal"], marker='o', label = "Constant Benchmark")


    ax.set_xlabel("Expected Shortfall", fontweight='bold', fontsize=20)
    ax.set_ylabel("E[Average Withdrawal]", fontweight='bold', fontsize=20)
    ax.legend(loc='lower left')
    ax.set_xlim([-680, 300])
    ax.set_ylim([35, 65])
    ax.spines[['right', 'top']].set_visible(False)
    ax.tick_params(axis='both', which='major', labelsize=12)
    plt.subplots_adjust(bottom=0.15)


    plt.savefig(out_dir + "/" + "summary_ef.pdf", format='pdf')

    return