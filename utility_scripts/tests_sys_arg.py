import sys
import json
import pandas as pd

# print("tracing parameter entered from terminal: ", sys.argv[1])

# list_k = [float(item) for item in sys.argv[1].split(" ")]

# [print(list_k)]


exp_name_dict = {"exp_name": "chen_decum", "start_time": "12:20:01"}


row1_dict = {"kappa": 1.0, "mean_wt": 25, "cvar": -30}

summary = {}

summary["exp_details"] = exp_name_dict
tracing_param = 1.0
summary[f"tracing_param_{str(tracing_param)}"] = row1_dict


out_file = open("myfile.json", "w") 
  
json.dump(summary, out_file, indent = 6) 
out_file.close() 

with open("myfile.json") as file:
    loaded_dict = json.load(file)

del loaded_dict["exp_details"]

df = pd.DataFrame.from_dict(loaded_dict, orient='index')