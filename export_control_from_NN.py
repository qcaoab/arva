import fun_construct_Feature_vector
import torch
import numpy as np

# goal: create a text file in peter's pde solver format of the explicit allocation and withdrawal control for DC, kappa = 1
# create pytorch NN instance, feed it t 


def export_controls(NN_list, params):
    
    torch.no_grad()
    
    w_min = params["w_grid_min"]
    w_max = params["w_grid_max"]
    nx = params["nx"] 
    n_rebal = params["N_rb"]
    
    q_min = torch.tensor(params["q_min"], device= params["device"])
    q_range = torch.tensor(params["q_max"] - params["q_min"], device= params["device"])
    
    w_grid_vector = torch.linspace(w_min, w_max, nx, device=params["device"])
    rb_times = np.linspace(params["T"], 0, params["N_rb"]+1, dtype=int)
    
    
    
    #withdrawal
    Qplus = torch.zeros((nx, len(rb_times)), device=params["device"])
    
    for n in rb_times:
        
        
        phi_1 = fun_construct_Feature_vector.construct_Feature_vector(params = params,  # params dictionary as per MAIN code
                                n = n+1,  # n is rebalancing event number n = 1,...,N_rb, used to calculate time-to-go
                                wealth_n = w_grid_vector,  # Wealth vector W(t_n^+), 
                                                # but *before* rebalancing at time t_n for (t_n, t_n+1)
                                feature_calc_option= None,  # "None" matches my code.  Set calc_option = "matlab" to match matlab code
                                withdraw= True)
        
        q_n_proportion = torch.squeeze(NN_list[0].forward(phi_1))
    
        q_n = torch.add(q_min, torch.mul(q_range, q_n_proportion))
        
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
        
        a_t_n_output = torch.squeeze(NN_list[1].forward(phi_2))
        
        Bplus[:,n] = torch.multiply(a_t_n_output[:,0], w_grid_vector)
    
    #write file
    filepath = f"control_files/control_file_dc_kappa_inf_nov30.txt"
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
    
