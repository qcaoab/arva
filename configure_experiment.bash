#!/bin/bash

# OBJECTIVE: This is a Bash script to facilitate training or testing multiple NN models at once, under different experiment scenarios. 

# To run this file, navigate to the /researchcode/ directory and enter the following:
# < bash configure_experiment.bash >
#-------------------------------------------

# Select config json file, containing all desired experiment parameters for each experiment to run: 
config_file=/exp_config_json_files/test_config1.json

# List of experiment names to run:
experiment_names=("quick_train1")

# Text console log output directory:
log_dir="text_logs"

#get time
current_date_time="`date +%d-%m-%y_%H:%M:%S`"

# Run all desired experiments in a loop:
for EXPERIMENT in $experiment_names
do
    nohup python3 -u argparse_driver.py "$EXPERIMENT" $config_file > "$log_dir/$EXPERIMENT"_"$current_date_time.txt"
done


