
import time
import numpy as np
import pandas as pd
import copy
# from fun_eval_objfun_NN_strategy import eval_obj_NN_strategy as objfun  #objective function for NN evaluation
from fun_eval_objfun_NN_strategy import eval_obj_NN_strategy_pyt as objfun_pyt #pytorch objective func
from fun_W_T_stats import fun_W_T_summary_stats
import torch
import fun_invest_NN_strategy
import os
import pickle
from torch.optim.swa_utils import AveragedModel
import json
import manage_nn_models

def run_Gradient_Descent_pytorch(NN_list, params, NN_training_options):
    
    # OBJECTIVE: Executes constant SGD and/or Adam, using pytorch implementations

    #------------------------------------------------------------------------------------------
    # OUTPUTS:
    # returns dictionary "res" with content:
    # res["method"] = method
    # res["nit"] = it     #total nr of iterations executed to get res["F_theta"]
    # res["val"] = val    #objective function value evaluated at res["F_theta"]
    # res["runtime_mins"] = t_runtime  # run time in MINUTES until output is obtained\
    # res["summary_df"] = summary_df, where:
    #           summary_df = pd.DataFrame([[method, it, val, supnorm_grad, t_runtime,
    #                                 and summary stats of terminal wealth

    # if output_progress == True:
    #     res["vval"] = vval    #pandas DataFrame outputted ONLY if  output_progress == True:
    #


    #------------------------------------------------------------------------------------------
    # INPUTS:
    # method = Gradient descent algorithm to follow [see below for full details]
    # params = params dictionary, with key components:
    #       params["N_d"] = total number of scenarios/observations in the training data
    #       params["Y"][j, n, i] = Training data, along sample path j, over time period (t_n, t_n+1), for asset i
                                    #this is modified into tensor format
    #       params["Y"].shape = (N_d, N_rb, N_a) = (Nr of training data pts, Nr of rebal, Nr of assets)

    # theta0 = starting point for SGD
    # tol = tolerance for solving problem (e.grad. 1e-06)
    # itbound = maximum number of iterations for the  algorithm
    # batchsize =  size of minibatch to select from the training data set (in params) for each iteration
    # check_exit_criteria == False: means we run code until itbound, do not check exit criteria on each iteration
    # output_progress == True/False: if True, we calculate objective function value at each iteration using FULL training set
    # nit_running_min =  the  nr of iterations at the end that will be used to get the running minimum

    
    #get NN training options
    itbound = NN_training_options["itbound_SGD_algorithms"]
    batchsize = NN_training_options["batchsize"]
    check_exit_criteria = NN_training_options["check_exit_criteria"]
    output_progress = NN_training_options["output_progress"]
    nit_running_min = NN_training_options["nit_running_min"]
    nit_IterateAveragingStart=NN_training_options["nit_IterateAveragingStart"]
    method = NN_training_options["methods"][0]
    running_min_from_avg = NN_training_options["running_min_from_avg"]
    running_min_from_sgd = NN_training_options["running_min_from_sgd"]
    
    #Adam options
    Adam_ewma_1 = NN_training_options["Adam_ewma_1"]
    Adam_ewma_2 = NN_training_options["Adam_ewma_2"]
    Adam_eta = NN_training_options["Adam_eta"]
    Adam_epsilon = NN_training_options["Adam_epsilon"]
    weight_decay = NN_training_options["Adam_weight_decay"]
    
    
    #Check that batch size does NOT exceed N_d
    if batchsize > params["N_d"]:
        raise ValueError("PVS error: Batch size exceeds size of all available training data.")

    t_start = time.time() #start keeping track of runtime
    res = {} #Initialize output dictionary


    # ----------------------------  INITIALIZE algorithms --------------------------------------------
    N_avg = 0   #running number of points over which iterate averaging is calculated
    if nit_IterateAveragingStart == None:
        nit_IterateAveragingStart = itbound  #To handle the None case in the if statement

    if check_exit_criteria == True:
        output_progress = True  #Set true to calculate function value at every iteration using full training data

    if output_progress == True: #Initialize output dataframe for res["vval"]
        vval = pd.DataFrame(columns=['it_nr', 'objfunc_val'])

    
    #not needed if initial fval not needed
    # #select random batch of data
    # batch_indices = np.random.choice(np.arange(0, params["N_d"], 1),
    #                                  size=(1, batchsize), replace=False)
    # batch_indices = batch_indices.flatten()  # make into a 0-dim array for slicing

    # params_it = copy.deepcopy(params)  # Create a copy of input data for initial value
    # del params_it["Y"]  # REPLACE  data in params_it with the subset for initial value
    # params_it["Y"] = torch.tensor(params["Y"][batch_indices, :, :].copy(), device=params["device"])  # populate with subset of training data
    # params_it["N_d"] = batchsize  # Update size of training data for params_it
    
    
    #initialize xi tensor
    xi = torch.tensor([params["xi_0"]], requires_grad=True, device=params["device"])
    
    
    if "Adam" in NN_training_options["methods"]:
        
        #create optimizer, starting with withdrawal NN params
        optimizer_nn = torch.optim.Adam([{'params': NN_list.parameters(), 
                                        'lr': Adam_eta, 
                                        'betas': (NN_training_options["Adam_ewma_1"], NN_training_options["Adam_ewma_2"] ),
                                        'weight_decay': weight_decay}                                   
                                      ])
        
        optimizer_xi = torch.optim.Adam([{'params': xi,
                                       'lr':params["xi_lr"]}                                    
                                      ])
        
        if NN_training_options["lr_schedule"]:
            schedulerxi = torch.optim.lr_scheduler.MultiStepLR(optimizer_xi, [int(itbound*0.70), int(itbound*0.97)], 
                                                 gamma=0.2, last_epoch=-1, verbose=False)
            
            scheduler2 = torch.optim.lr_scheduler.MultiStepLR(optimizer_nn, [int(itbound*0.70), int(itbound*0.97)], 
                                                 gamma=0.2, last_epoch=-1, verbose=False)
            
    
        
    #init iterate averaging model
    swa_both_nns = AveragedModel(NN_list)
    v_min = np.inf
    
    #parameters for 'manual' averaging of xi:
    N_avg = 0 
    xi_avg = xi
        
    # ---------------------------- MAIN LOOP --------------------------------------------

    for it in np.arange(1, itbound+1, 1):   #will run inclusive of itbound
        
        # Select batch_indices = indices in the batch/subset of training data for SGD
        #   sample WITHOUT replacement from {0,1,...,M-1}
        batch_indices = np.random.choice(np.arange(0, params["N_d"], 1),
                                         size = (1,batchsize), replace = False)
                        #--- numpy.random.choice: generates a random sample from a given 1-D array

        batch_indices = batch_indices.flatten()     #make into a 0-dim array for slicing

        params_it = copy.deepcopy(params)  # Create a copy of input data for this iteration

        #-------------------------------------------------------------------------------------
        # REPLACE training data in params_it with the subset
        # - so that we can use NN function objfun without change
        del params_it["Y"] # delete training data
        # params_it["Y"] = torch.tensor(params["Y"][batch_indices, :, :], device=params["device"]) # populate with subset of training data
        params_it["Y"] = params["Y"][batch_indices, :, :]
        params_it["N_d"] = batchsize
        # params_it["benchmark_W_T_vector"] = params["benchmark_W_T_vector"][batch_indices].copy()  # populate with subset
        # params_it["benchmark_W_paths"] = params["benchmark_W_paths"][batch_indices, :].copy()  # populate with subset

        #--------------------  UPDATE STEP -------------------
        
        #clear gradients from previous steps
        optimizer_nn.zero_grad()
        optimizer_xi.zero_grad()
        
        # eval obj fun with SGD batch, includes forward pass on NN with updated weights from last step
        f_val, _ = objfun_pyt(NN_list, params_it, xi)
      
        #calc gradients
        f_val.backward()
        
        #update parameters
        optimizer_nn.step()
        optimizer_xi.step()
        
        #lr scheduler step
        if NN_training_options["lr_schedule"]:
            # scheduler1.step()
            scheduler2.step()
            schedulerxi.step()
        
        #try to clean up memory, lol
        torch.cuda.empty_cache()
               
        # ----------------
        #ITERATE AVERAGING
        
        if it == nit_IterateAveragingStart: #initialize
            xi_avg = xi.detach().clone()
            xi_min = xi.detach().clone()
        
        
        if (it - 1) >= nit_IterateAveragingStart:
            swa_both_nns.update_parameters(NN_list)
            
            #xi averaging            
            N_avg = N_avg + 1
            xi_avg = (xi_avg*N_avg + xi.detach()) / (N_avg + 1)
            
        # else:
        #     xi_avg = xi.detach()
                          
        # ----------------
        #RUNNING MINIMUM
        
        if (it >= itbound - nit_running_min):
            # OR, for last 'nit_running_min' iterations, calculate the newval for EVERY iteration

            # newval is calculated using ALL data, not just subset
            if running_min_from_avg:
                with torch.no_grad():
                    new_fval, _ = objfun_pyt(swa_both_nns.module, params, xi_avg)
            
                if new_fval < v_min:
                    
                    NN_list_min = copy.deepcopy(swa_both_nns.module)
                    xi_min = xi_avg.detach().clone()
                    
                    v_min = new_fval.detach().clone()      
                    print("new min fval from swa: ", new_fval.detach().cpu().numpy())
        
            if running_min_from_sgd:
                with torch.no_grad():
                    new_fval, _ = objfun_pyt(NN_list, params, xi)
            
                if new_fval < v_min:
                    
                    NN_list_min = copy.deepcopy(NN_list)
                    xi_min = xi.detach().clone()
                    
                    v_min = new_fval.detach().clone()      
                    print("new min fval from sgd: ", v_min.cpu().numpy())
                
        # ----------------
        #Update user on progress every x% of SGD iterations
        if itbound >= 1000:
            if it in np.append(np.arange(0, itbound, int(0.02*itbound)), itbound):
                print( str(it/itbound * 100) + "% of gradient descent iterations done. Method = " + method)                
                with torch.no_grad():
                    new_fval, _ = objfun_pyt(NN_list, params, xi) # uses full tensor version of params 
                    if new_fval < v_min:
                        NN_list_min = copy.deepcopy(NN_list)
                        xi_min = xi.detach().clone()
                        v_min = new_fval.detach().clone()
                        print("new min fval: ", v_min.cpu().numpy())
                # supnorm_grad = np.linalg.norm(grad_theta_new_mc, ord = np.inf)     #max(abs(gradient))
                print("Current xi: ", xi.detach().cpu().numpy())
                print( "objective value function right now is: " + str(float(new_fval)))
                # print( "gradient value of function right now is: " + str(grad_theta_new_mc))
                # print( "supnorm grad right now is: " + str(supnorm_grad))
                # print("Weights right now are: ")
                # print(theta)


    # ---------------------------- End: MAIN LOOP --------------------------------------------

    #--------------- SET OUTPUT VALUES ---------------
    
    optimizer_nn.zero_grad()
    del NN_list
    torch.cuda.empty_cache()
    torch.no_grad()

    #temporary output
    with torch.no_grad():
        
        if params["nn_withdraw"]:  #decumulation
            params, _, qsum_T_vector = fun_invest_NN_strategy.withdraw_invest_NN_strategy(NN_list_min, params)
        
        else: #NO decumulation
            params, W_T_vector = fun_invest_NN_strategy.invest_NN_strategy_pyt(NN_list_min, params)
            
    # print("Median terminal wealth: ", torch.median(g))
        min_fval, _ = objfun_pyt(NN_list_min, params, xi_min)
    
        print("min fval: ", min_fval.detach().cpu().numpy())
    
    #Append terminal wealth stats using this optimal value
    W_T = params["W"][:, -1]
    W_T_stats_dict = fun_W_T_summary_stats(W_T)
    
    
    #convert xi from tensor to np 
    xi_np = xi_min.detach().cpu().numpy()
    
     
    res["temp_w_output_dict"] = W_T_stats_dict
    if params["nn_withdraw"]:  #decumulation
        res["q_avg"] = torch.mean(qsum_T_vector).cpu().detach().numpy()/(params["N_rb"]+1)
    res["optimal_xi"] = xi_np
    res["average_median_p"] = np.mean(np.median(params["NN_asset_prop_paths"], axis = 0)[:,1])
    res["objfun_final"] = min_fval
    

    #---------CREATE SAVED MODEL OBJECT---------------
    # If not testing a model, bundle the trained NN model and all revelant meta data into a dictionary. 
    # Save as serialized pickle object. This can be easily loaded later using pickle.load() 
    # These saved models can be used later for either 1) testing, or 
    # 2) initializing new models for transfer learning. i.e., if you are running multiple kappa points to create an efficient frontier, it can be useful to initialize from the previous kappa point to speed up training or prevent hang ups in training, like getting stuck at a local minimum. 
    
    if params["iter_params"] != "check":
         
        manage_nn_models.save_model(NN_list_min, params, (xi_np))
        
        # print("saved model: ")
        # print(NN_list_min.state_dict())
        # print("xi: ", xi_np)
    
    # Make sure parameter dictionary is updated so that e.g. fun_Objective_functions can work correctly
    params["xi"] = xi_np[0] 
    
    
    # TODO
    # # ---------------------------- SET OUTPUT --------------------------------------------
    # res["method"] = method
    # res["F_theta"] = F_theta        #minimizer or point where algorithm stopped
    # res["NN_theta"] = NN_theta        #minimizer or point where algorithm stopped
    # res["nit"] = int(it)     #total nr of iterations executed to get res["NN_theta"]
    # res["val"] = val    #objective function value evaluated at res["NN_theta"]
    # res["supnorm_grad"] = supnorm_grad  # sup norm of gradient vector at res["NN_theta"], i.e. max(abs(gradient))
    # res["runtime_mins"] = t_runtime  # run time in MINUTES until output is obtained

    # #Append terminal wealth stats using this optimal value
    # W_T = params["W"][:, -1]

    # #Override W_T in one case
    # if params["obj_fun"] == "one_sided_quadratic_target_error":  # only in this case
    #     if params["obj_fun_cashwithdrawal_TrueFalse"] == True:  #Check if we want values *after* cash withdrawal
    #         W_T = params["W_T_cashwithdraw"]



    # W_T_stats_dict = fun_W_T_summary_stats(W_T)

    # #Remove "W_T_summary_stats" dataframe
    # del W_T_stats_dict['W_T_summary_stats']

    # #Add summary stats to res dictionary

    # res.update(W_T_stats_dict)

    # #Put results in pandas.Dataframe for easy comparison with other methods
    # W_T_stats_df = pd.DataFrame(data=W_T_stats_dict, index=[0])


    # summary_df = pd.DataFrame([[method, it, val, supnorm_grad, t_runtime]],
    #                           columns=["method", "nit", "objfunc_val", "supnorm_grad", "runtime_mins"])
    
    # summary_df = pd.concat([summary_df, W_T_stats_df], axis=1, ignore_index = False)


    # res["summary_df"] = summary_df

    # if output_progress == True:
    #     res["vval"] = vval    #pandas DataFrame outputted ONLY if  output_progress == True:


    return res
    