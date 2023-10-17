# OBJECTIVE: This is a Bash script to facilitate training multiple NN models at once, under different experiment scenarios. 

# To run this file, navigate to the /researchcode/ directory and enter sh configure_TRAIN.sh
#-------------------------------------------

# Select config json file, containing all desired experiment parameters for each experiment to run: 
config_file = "/home/marcchen/Documents/DC_plan_code_sep2023/researchcode/exp_config_json_files/multi_portfolio_exp1.json"

# List of experiment names to run:
experiment_names = ("2basic_NN" "3basic_NN" "2_factor_unconstrain")

# Text console log output directory:
log_dir = "text_logs"

# Run all desired experiments in a loop:
for EXPERIMENT in experiment_names
do
    nohup python3 -u argparse_driver.py "$EXPERIMENT" config_file > "$log_dir/$EXPERIMENT_log.txt"
done


