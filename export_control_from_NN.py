import fun_construct_Feature_vector
import torch
import numpy as np
from constraint_activations import w_custom_activation
from constraint_activations import asset_constraint_activation

# goal: create a text file in peter's pde solver format of the explicit allocation and withdrawal control for DC, kappa = 1
# create pytorch NN instance, feed it t 


def export_controls(NN_list, params):
    
    torch.no_grad()
    
    w_min = params["w_grid_min"]
    w_max = params["w_grid_max"]
    base_nx = params["nx"] 
    n_rebal = params["N_rb"]
    
    q_min = torch.tensor(params["q_min"], device= params["device"])
    q_range = torch.tensor(params["q_max"] - params["q_min"], device= params["device"])
    
    w_grid_vector = torch.linspace(w_min, w_max, base_nx, device=params["device"])
    w_additional = torch.tensor( [999.8, 999.9, 999.95, 1000, 1000.05, 1000.1, 1000.2], device = params["device"])
    w_grid_vector = torch.cat((w_grid_vector,w_additional))
    w_grid_vector, indices= torch.sort(w_grid_vector)
    
    rb_times = np.linspace(params["T"], 0, params["N_rb"]+1, dtype=int)
    
    nx = len(w_grid_vector)
    
    
    #withdrawal
    Qplus = torch.zeros((nx, len(rb_times)), device=params["device"])
    
    for n in rb_times:
        
        
        phi_1 = fun_construct_Feature_vector.construct_Feature_vector(params = params,  # params dictionary as per MAIN code
                                n = n+1,  # n is rebalancing event number n = 1,...,N_rb, used to calculate time-to-go
                                wealth_n = w_grid_vector,  # Wealth vector W(t_n^+), 
                                                # but *before* rebalancing at time t_n for (t_n, t_n+1)
                                feature_calc_option= None,  # "None" matches my code.  Set calc_option = "matlab" to match matlab code
                                withdraw= True)
        
        nn_out = torch.squeeze(NN_list[0].forward(phi_1))
        
        
        q_n = w_custom_activation(nn_out, w_grid_vector, params)
    
        # fill in qplus back to front, set to negative for pde solver format
        Qplus[:,n] = -q_n
    
    
    
    #bond amounts
    Bplus = torch.zeros((nx, len(rb_times)), device=params["device"])
    
    for n in rb_times:
        
        #withdrawal
        phi_2 = fun_construct_Feature_vector.construct_Feature_vector(params = params,  # params dictionary as per MAIN code
                                n = n+1,  # n is rebalancing event number n = 1,...,N_rb, used to calculate time-to-go
                                wealth_n = w_grid_vector,  # Wealth vector W(t_n^+), 
                                                # but *before* rebalancing at time t_n for (t_n, t_n+1)
                                feature_calc_option= None,  # "None" matches my code.  Set calc_option = "matlab" to match matlab code
                                withdraw= False)
        
        if params["factor_constraint"]:
            a_t_n_output = asset_constraint_activation(NN_list[1].forward(phi_2), params)
        else:
            a_t_n_output = NN_list[1].forward(phi_2)
        
        a_t_n_output = torch.squeeze(a_t_n_output)
        
        Bplus[:,n] = torch.multiply(a_t_n_output[:,0], w_grid_vector)
    
    #write file
    filepath = params["control_filepath"]
    open(filepath, "w").close()
    
    with open(filepath,"a+") as f:
        f.write(f"n_rebal: {n_rebal+1}\n")
        
    for n in rb_times:
        
        with open(filepath,"a+") as f:
            f.write(f"forward_time:  {n}\n")
            
        with open(filepath,"a+") as f:
            f.write(f"nx: {nx}\n")
            
        with open(filepath,"a+") as f:
            f.write(" W_plus      B_plus   Q_plus \n")
        
        with open(filepath,"a+") as f:
            for i in range(nx):
                f.write(f"{w_grid_vector[i]} {Bplus[i,n]} {Qplus[i,n]} \n")
    
    return
    
