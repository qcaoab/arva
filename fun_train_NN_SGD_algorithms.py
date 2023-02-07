
import time
import numpy as np
import pandas as pd
import copy
from fun_eval_objfun_NN_strategy import eval_obj_NN_strategy as objfun  #objective function for NN evaluation
from fun_eval_objfun_NN_strategy import eval_obj_NN_strategy_pyt as objfun_pyt #pytorch objective func
from fun_W_T_stats import fun_W_T_summary_stats
import torch
import fun_invest_NN_strategy

from torch.optim.swa_utils import AveragedModel

import json

def run_Gradient_Descent_pytorch(NN_list, NN_orig_list, params, NN_training_options):
    
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
                                        'lr': NN_training_options["Adam_eta"], 
                                        'betas': (NN_training_options["Adam_ewma_1"], NN_training_options["Adam_ewma_2"] ),
                                        'weight_decay': weight_decay}                                   
                                      ])
        
        optimizer_xi = torch.optim.Adam([{'params': xi,
                                       'lr': params["xi_lr"]}                                    
                                      ])
        
        if NN_training_options["lr_schedule"]:
            schedulerxi = torch.optim.lr_scheduler.MultiStepLR(optimizer_xi, [int(itbound*0.70), int(itbound*0.97)], 
                                                 gamma=0.2, last_epoch=-1, verbose=False)
            
            scheduler2 = torch.optim.lr_scheduler.MultiStepLR(optimizer_nn, [int(itbound*0.70), int(itbound*0.97)], 
                                                 gamma=0.2, last_epoch=-1, verbose=False)
            
        #append xi to optimization parameters
        # if not params["xi_constant"]: 
        #     optimizer.param_groups[0]['params'].append(xi)
        
    #init iterate averaging model
    swa_both_nns = AveragedModel(NN_list)
    v_min = np.inf
    
    #parameters for 'manual' averaging of xi:
    N_avg = 0 
    xi_avg = xi
        
    #create tensor version of params
    # params_full_tensor = copy.deepcopy(params) 
    # params_full_tensor["Y"] = torch.tensor(params["Y"], device=params["device"])
    
    # params_orig = copy.deepcopy(params)

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
    params, _, qsum_T_vector = fun_invest_NN_strategy.withdraw_invest_NN_strategy(NN_list_min, params)
    # print("Median terminal wealth: ", torch.median(g))
    min_fval, _ = objfun_pyt(NN_list_min, params, xi_min)
    
    print("min fval: ", min_fval.detach().cpu().numpy())
    
    #Append terminal wealth stats using this optimal value
    W_T = params["W"][:, -1]
    W_T_stats_dict = fun_W_T_summary_stats(W_T)
    
    
    #convert xi from tensor to np 
    xi_np = xi_min.detach().cpu().numpy()
    
    
    # print("-----------------------------------------------")
    # print("Selected results: NN-strategy-on-TRAINING dataset (temp implementation")
    # print("W_T_mean: " + str(W_T_stats_dict["W_T_mean"]))
    # print("W_T_median: " + str(W_T_stats_dict["W_T_median"]))
    # print("W_T_pctile_5: " + str(W_T_stats_dict["W_T_pctile_5"]))
    # print("W_T_CVAR_5_pct: " + str(W_T_stats_dict["W_T_CVAR_5_pct"]))
    # print("Average q (qsum/M+1): ", torch.mean(qsum_T_vector).cpu().detach().numpy()/(params["N_rb"]+1))
    # print("Optimal xi: ", xi_np)
    # print("Expected(across Rb) median(across samples) p_equity: ", np.mean(np.median(params["NN_asset_prop_paths"], axis = 0)[:,1]))
    # print("-----------------------------------------------")
     
    res["temp_w_output_dict"] = W_T_stats_dict
    res["q_avg"] = torch.mean(qsum_T_vector).cpu().detach().numpy()/(params["N_rb"]+1)
    res["optimal_xi"] = xi_np
    res["average_median_p"] = np.mean(np.median(params["NN_asset_prop_paths"], axis = 0)[:,1])
    res["objfun_final"] = min_fval
    
    # save model for continuation learning
        # NN_theta = F_theta[0:-1]
        # xi = F_theta[-1]  # Last entry is xi, where (xi**2) is candidate VAR

        #save NN params and xi for continuation learning
    model_save_path = params["console_output_prefix"]
    local_path = params["local_path"]
    kappa = params["obj_fun_rho"]
    torch.save(NN_list_min.state_dict(),f"{local_path}/saved_models/NN_opt_{model_save_path}_kappa_{kappa}")
    
    optimal_xi = {"xi":str(xi_np[0])}
    
    with open(f'{local_path}/saved_models/xi_opt_{model_save_path}_kappa_{kappa}.json', 'w') as outfile:
        json.dump(optimal_xi, outfile)

    print("saved model: ")
    print(NN_list_min.state_dict())
    print("xi: ", xi_np)
    
    # Make sure parameter dictionary is updated so that e.g. fun_Objective_functions can work correctly
    params["xi"] = xi_np[0] #(xi**2) is candidate VAR
    
    
    
    # Export trained NN from pytorch to original implementation, 'NN_object' is original NN object using weights
    # from pytorch averaged model.
    
    #pytorch NN is NN_list_min
    
    # NN_withdraw_orig = NN_list_min.module[0].export_weights(NN_orig_list[0])
    # NN_allocation_orig = NN_list_min.module[1].export_weights(NN_orig_list[1])
    
    # # # # copy weights from layers into theta
    # NN_withdraw_orig.stack_NN_parameters()
    # NN_allocation_orig.stack_NN_parameters()
    
    
    
    
    # TO DO: need to implement pieter NN implementation of this.
    
    #append xi_np to NN theta for f_theta
    # F_theta = np.append(NN_object.theta, xi_np)    
    
    # #calc original objfun
    # (params, val, _, grad) = objfun(F_theta = F_theta,   # record the objective function value
    #                              NN_object = NN_object, params = params, output_Gradient=True)
    # supnorm_grad = np.linalg.norm(grad, ord=np.inf)  # max(abs(gradient))

    # if params["obj_fun"] == "mean_cvar":
    #     NN_theta = F_theta[0:-2]
    #     xi = F_theta[-2]  # Second-last entry is xi, where (xi**2) is candidate VAR
    #     gamma = F_theta[-1]  # Lagrange multiplier

    #     # Make sure parameter dictionary is updated so that e.g. fun_Objective_functions can work correctly
    #     params["xi"] = xi
    #     params["gamma"] = gamma

    # elif params["obj_fun"] == "mean_cvar_single_level":
    #     NN_theta = F_theta[0:-1]
    #     xi = F_theta[-1]  # Last entry is xi, where (xi**2) is candidate VAR

    #     #added theta cache
    #     optimal_params = {"NN":NN_theta.tolist()}
    #     with open('NN_optimal2.json', 'w') as outfile:
    #         json.dump(optimal_params, outfile)
        
    #     print("NN weights: " + str(NN_theta))
    #     print("Minimum obj value:" + str(val))
    #     print("Optimal xi: " + str(xi))


    #     # Make sure parameter dictionary is updated so that e.g. fun_Objective_functions can work correctly
    #     params["xi"] = xi #(xi**2) is candidate VAR

    # else:
    #     NN_theta = F_theta

    #     #added theta cache
    #     optimal_params = {"NN":NN_theta.tolist()}
    #     with open('NN_optimal2.json', 'w') as outfile:
    #         json.dump(optimal_params, outfile)



    # t_end = time.time()
    # t_runtime = (t_end - t_start)/60    #we want to output runtime in MINUTES

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
    


def run_Gradient_Descent(method,
                         theta0,        # initial parameter vector (weights and biases) and other params of form
                                        # [NN_theta, extra_theta] where NN_theta = NN_object.NN_theta
                                        #           extra_theta = ADDITIONAL parameters for objective function used by e.g. mean-cvar
                         NN_object,             #object of class_Neural_Network with structure as setup in main code
                         params,                # dictionary with investment parameters as set up in main code
                         itbound,               #max nr of iterations to run
                         batchsize,            # batch size for SGD/mini-batch gradient descent
                         check_exit_criteria = False,   #if False: means we run code until itbound, do not check exit criteria on each iteration
                         tol = None,            # only if check_exit_criteria = True
                         output_progress = False,   #if True, we calculate objective function value at each iteration using FULL training set
                         nit_running_min = 0,  # nr of iterations at the end that will be used to get the running minimum for output
                         nit_IterateAveragingStart = None, #the point from which ONWARDS we do iterate averaging; = None to switch IA off
                         SGD_learningrate = None,  #50.0 SGD Learning rate constant
                         Adagrad_epsilon = None,  #1e-08 Adagrad small constant in denominator to avoid div by zero
                         Adagrad_eta = None,  #1.0 Adagrad numerator of the adaptive learning rate
                         Adadelta_ewma = None,  #0.9 Adadelta EWMA parameter for gradient squared AND for the parameter vector squared
                         Adadelta_epsilon = None,  #1e-08 Adadelta small constant in denominator to avoid div by zero
                         RMSprop_ewma = None,  # 0.8 RMSprop EWMA parameter for RMSprop (Hinton suggests 0.9)
                         RMSprop_epsilon = None,  # 1e-08 RMSprop small constant in denominator to avoid div by zero
                         RMSprop_eta = None,  # 0.1 RMSprop numerator of the adaptive learning rate
                         Adam_ewma_1 = None,  # 0.9 Adam (beta1) EWMA parameter for gradient (momentum) process
                         Adam_ewma_2 = None,  # 0.9 Adam (beta2) EWMA parameter for gradient SQUARED (speed) process
                         Adam_eta = None,  # 0.1 Adam eta in the numerator of the Adam learning rate
                         Adam_epsilon = None  # 1e-08 Adam small constant in denominator to avoid div by zero
                         ):
    # OBJECTIVE: Executes any of the common gradient descent algorithms specified by "method" (see below)
    #   "SGDconstant" = SGD with constant learning rate
    #   "Adagrad"
    #   "Adadelta"
    #   "RMSprop"
    #   "Adam"

    #------------------------------------------------------------------------------------------
    # OUTPUTS:
    # returns dictionary "res" with content:
    # res["method"] = method
    # res["F_theta"] = objective function parameters, [NN_theta, extra_theta] minimizer of objfun or point where algorithm stopped
    # res["NN_theta"] = NN_theta  # NN parameter vector that is minimizer of NN params of objfun or point where algorithm stopped
    # res["nit"] = it     #total nr of iterations executed to get res["F_theta"]
    # res["val"] = val    #objective function value evaluated at res["F_theta"]
    # res["supnorm_grad"] = supnorm_grad  # sup norm of gradient vector at res["F_theta"], i.e. max(abs(gradient))
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
    #       params["Y"].shape = (N_d, N_rb, N_a) = (Nr of training data pts, Nr of rebal, Nr of assets)

    # theta0 = starting point for SGD
    # tol = tolerance for solving problem (e.grad. 1e-06)
    # itbound = maximum number of iterations for the  algorithm
    # batchsize =  size of minibatch to select from the training data set (in params) for each iteration
    # check_exit_criteria == False: means we run code until itbound, do not check exit criteria on each iteration
    # output_progress == True/False: if True, we calculate objective function value at each iteration using FULL training set
    # nit_running_min =  the  nr of iterations at the end that will be used to get the running minimum

    # ------------------------------------------------------------------------------------------
    #METHODS and method-specific INPUTS:

    if method == "SGD_constant": #Vanilla SGD with constant learning rate
        #   SGD_learningrate = multiplier of (estimate of) gradient in each update
        if SGD_learningrate == None or SGD_learningrate <= 0:
            raise ValueError("PVS error: 'SGD_learningrate' must be specified for method 'SGD_constant'.")


    elif method == "SGD_lindecay":  #Vanilla SGD with linearly decaying learning rate up to point, constant thereafter
        print("temp")


    elif method == "Adagrad":  #Adagrad algorithm
        if Adagrad_epsilon == None or Adagrad_eta == None:
            raise ValueError("PVS error: 'Adagrad_epsilon' and 'Adagrad_eta' must be specified \
                            for method 'Adagrad'.")

    elif method == "Adadelta":  #Adadelta algorithm
        if Adadelta_ewma == None or Adadelta_epsilon == None:
            raise ValueError("PVS error: 'Adadelta_ewma' and 'Adadelta_epsilon' must be specified \
                            for method 'Adadelta'.")

    elif method == "RMSprop":  #RMSprop algorithm
        if RMSprop_ewma == None or RMSprop_eta == None or RMSprop_epsilon == None:
            raise ValueError("PVS error: 'RMSprop_ewma' and 'RMSprop_eta' and 'RMSprop_epsilon' must be specified \
                            for method 'RMSprop'.")

    elif method == "Adam":  #Adam algorithm
        if Adam_ewma_1 == None or Adam_ewma_2 == None or Adam_eta == None or Adam_epsilon == None:
            raise ValueError("PVS error: 'Adam_ewma_1' and 'Adam_ewma_2' and 'Adam_eta' and  'Adam_epsilon' must be specified \
                            for method 'Adam'.")
    else:
        raise ValueError("PVS error: 'method' specified not in the algorithms provided.")


    # ------------------------------------------------------------------------------------------


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


    if method == "Adadelta":    #With Adadelta, there CANNOT be any zeros in theta0 otherwise no learning for that param
        if len(theta0) > len(np.nonzero(theta0)): #if nr of entries of theta0 > nr of nonzero entries
            theta0 = np.random.rand(len(theta0))    #replace theta0


    # Get INITIAL value of gradient using the batch size (NOT whole dataset)
    theta_new = theta0.copy()  # Initialize
    batch_indices = np.random.choice(np.arange(0, params["N_d"], 1),
                                     size=(1, batchsize), replace=False)
    batch_indices = batch_indices.flatten()  # make into a 0-dim array for slicing

    params_it = copy.deepcopy(params)  # Create a copy of input data for initial value
    del params_it["Y"]  # REPLACE  data in params_it with the subset for initial value
    params_it["Y"] = params["Y"][batch_indices, :, :].copy()  # populate with subset of training data
    params_it["N_d"] = batchsize  # Update size of training data for params_it

    # if obj  in ["ads_stochastic", "qd_stochastic", "ir_stochastic", "te_stochastic"],
    #   also replace "benchmark_W_T_vector" with benchmark W_T values on subset
    if params_it["obj_fun"] in ["ads_stochastic", "qd_stochastic", "ir_stochastic", "te_stochastic"]:
        del params_it["benchmark_W_T_vector"]  # delete full vector
        del params_it["benchmark_W_paths"]  # delete full set of paths

        params_it["benchmark_W_T_vector"] = params["benchmark_W_T_vector"][batch_indices].copy()  # populate with subset
        params_it["benchmark_W_paths"] = params["benchmark_W_paths"][batch_indices, :].copy()  # populate with subset

    #If using Trading signals, also replace the trade signal data with the selected indices
    if params_it["use_trading_signals_TrueFalse"] == True:
        del params_it["TradSig"]  # REPLACE data in params_it with the subset
        params_it["TradSig"] = params["TradSig"][batch_indices, :, :].copy()  # populate with subset of training data


    # Get gradient grad using this subset of training data
    (_, newval, _, grad_theta_new) = objfun(F_theta = theta_new,
                                    NN_object = NN_object,
                                    params = params_it, 
                                    output_Gradient=True)


    #Initialize running minimum for final output [we check this over the last nit_running_min iterations]
    theta_min = theta0.copy()
    v_min = np.inf

    #Initialize value to be used as running average
    theta_avg = theta_min.copy()

    #Algorithm-specific initialization:
    if method == "Adagrad":
        vnew_Adagrad = np.power(grad_theta_new, 2)  # Adagrad: initialize speed term

    if method == "Adadelta":
        vnew_Adadelta = (1 - Adadelta_ewma) * np.power(grad_theta_new, 2)  # Adadelta: initialize speed term
        unew_Adadelta = (1 - Adadelta_ewma) * np.power(theta_new, 2)  # Adadelta: initialize parameter squared

    if method == "RMSprop":
        vnew_RMSprop = (1 - RMSprop_ewma) * np.power(grad_theta_new, 2)  # RMSprop: initialize speed term

    if method == "Adam":
        mnew_Adam = (1 - Adam_ewma_1) * grad_theta_new  # Adam: initialize momentum term
        vnew_Adam = (1 - Adam_ewma_2) * np.power(grad_theta_new, 2)  # Adam: initialize speed term



    # ---------------------------- MAIN LOOP --------------------------------------------

    for it in np.arange(1, itbound+1, 1):   #will run inclusive of itbound
        #print(str(it))

        if check_exit_criteria == True: #Check exit criteria

            if tol == None:
                raise ValueError("PVS error: 'tol' needs to be specified if check_exit_criteria = True.")

            (_, _, _, grad_theta_new) = objfun(F_theta = theta_new,
                                                          NN_object = NN_object, params = params, output_Gradient=True)

            supnorm_grad = np.linalg.norm(grad_theta_new, ord = np.inf)     #max(abs(gradient))

            if (newval < tol) or (supnorm_grad < tol):

                print("---- Exiting Early ---- ")
                print("Exiting at objective value = " + str(newval))
                print("Exiting at gradient = " + str(supnorm_grad))
                break   #Exit loop, do not execute rest of the main loop


        #Passed exit criteria (or not applicable), so continue


        if method == "Adadelta": #Record previous value theta for Adadelta
            if it == 1:
                theta_prev = np.zeros(theta_new.shape)  # Needed for Adadelta
            else:
                theta_prev = theta.copy()  # Needed for Adadelta


        theta = theta_new.copy() #Update theta, for all methods

        if method == "Adagrad":
            v_Adagrad = vnew_Adagrad.copy()

        if method == "Adadelta":
            v_Adadelta = vnew_Adadelta.copy()  # Adadelta: EWMA of gradient squared
            u_Adadelta = unew_Adadelta.copy()  # Adadelta: EWMA of param diff squared

        if method == "RMSprop":
            v_RMSprop = vnew_RMSprop.copy()  # RMSprop

        if method == "Adam":
            v_Adam = vnew_Adam.copy()  # Adam speed
            m_Adam = mnew_Adam.copy()  # Adam momentum


        if output_progress == True:
            val = newval.copy()  # record the objective function value as it stands at end of PREVIOUS iteration
            temp_vval = pd.DataFrame([[it-1, val]], columns=['it_nr', 'objfunc_val'])
            vval = vval.append(temp_vval)


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
        params_it["Y"] = params["Y"][batch_indices, :, :].copy() #populate with subset of training data
        params_it["N_d"] = batchsize  # Update size of training data for params_it

        #if obj  in ["ads_stochastic", "qd_stochastic", "ir_stochastic", "te_stochastic"],
        # also replace "benchmark_W_T_vector" with benchmark W_T values on subset
        if params_it["obj_fun"] in ["ads_stochastic", "qd_stochastic", "ir_stochastic", "te_stochastic"]:
            del params_it["benchmark_W_T_vector"]  # delete full vector
            del params_it["benchmark_W_paths"]  # delete full set of paths
            
            params_it["benchmark_W_T_vector"] = params["benchmark_W_T_vector"][batch_indices].copy()  # populate with subset
            params_it["benchmark_W_paths"] = params["benchmark_W_paths"][batch_indices, :].copy()  # populate with subset

        # If using Trading signals, also replace the trade signal data with the selected indices
        if params_it["use_trading_signals_TrueFalse"] == True:
            del params_it["TradSig"]  # REPLACE data in params_it with the subset
            params_it["TradSig"] = params["TradSig"][batch_indices, :, :].copy()  # populate with subset of training data

        # Get gradient grad using this subset of training data
        (_, _, _, grad) = objfun(F_theta = theta,
                              NN_object = NN_object, params = params_it, output_Gradient=True)


        #--------------------  UPDATE STEP -------------------
        if method == "SGD_constant": # Vanilla SGD with constant learning rate
            theta_new = theta - SGD_learningrate * grad

        elif method == "Adagrad":   #Adagrad update step
            vnew_Adagrad = v_Adagrad + np.power(grad, 2)
            learningrate_Adagrad = Adagrad_eta / np.sqrt(vnew_Adagrad + Adagrad_epsilon)
            theta_new = theta - np.multiply(learningrate_Adagrad, grad)

        elif method == "Adadelta": # Adadelta update step
            vnew_Adadelta = (Adadelta_ewma * v_Adadelta) + (1 - Adadelta_ewma) * np.power(grad, 2)
            unew_Adadelta = (Adadelta_ewma * u_Adadelta) + (1 - Adadelta_ewma) * np.power(theta - theta_prev, 2)
            earningrate_Adadelta = np.divide(np.sqrt(unew_Adadelta), np.sqrt(vnew_Adadelta + Adadelta_epsilon))
            theta_new = theta - np.multiply(earningrate_Adadelta, grad)


        elif method == "RMSprop":   # RMSprop update step
            vnew_RMSprop = (RMSprop_ewma * v_RMSprop) + (1 - RMSprop_ewma) * np.power(grad, 2)
            learningrate_RMSprop = RMSprop_eta / np.sqrt(vnew_RMSprop + RMSprop_epsilon)
            theta_new = theta - np.multiply(learningrate_RMSprop, grad)


        elif method == "Adam": # Adam update step
            mnew_Adam = (Adam_ewma_1 * m_Adam) + (1 - Adam_ewma_1) * grad
            mnew_Adam_hat = mnew_Adam / (1 - (Adam_ewma_1 ** it))

            vnew_Adam = (Adam_ewma_2 * v_Adam) + (1 - Adam_ewma_2) * np.power(grad, 2)
            vnew_Adam_hat = vnew_Adam / (1 - (Adam_ewma_2 ** it))

            theta_new = theta - np.multiply((Adam_eta / np.sqrt(vnew_Adam_hat + Adam_epsilon)), mnew_Adam_hat)

        else:
            raise ValueError("PVS error: Method selected not in list of algorithms implemented.")

        # ----------------
        #ITERATE AVERAGING
        if (it -1) >= nit_IterateAveragingStart: #nit_IterateAveragingStart is the point from which ONWARDS we do iterate averaging
            N_avg = N_avg + 1   #the running nr of points over which the average is calculated
            theta_avg = (theta_avg*N_avg + theta_new) / (N_avg + 1) #running average
        else:
            theta_avg = theta_new.copy()

        # ----------------
        #RUNNING MINIMUM
        if (output_progress == True) or (it >= itbound - nit_running_min):
            # Always calc if 'output_progress' == True
            # OR, for last 'nit_running_min' iterations, calculate the newval for EVERY iteration
            # regardless of "output_progress", and keep track of running minimum

            # newval is calculated using ALL data, not just subset
            (_, newval, _, _) = objfun(F_theta = theta_avg,
                                       NN_object = NN_object, params = params, output_Gradient=False)

            #Keep track of running minimum for last 'nit_running_min' nr of iterations
            if newval < v_min:  #If there is improvement
                theta_min = theta_avg.copy() #Set new NN_theta_min
                v_min = newval.copy() #Set new min value

        else: # to take care of the case if we don't do a running minimum, e.g. if nit_running_min = 0
              #so we always have a theta_min to use below
            theta_min = theta_avg.copy()

        # ----------------
        #Update user on progress every x% of SGD iterations
        if itbound >= 1000:
            if it in np.append(np.arange(0, itbound, int(0.02*itbound)), itbound):
                print( str(it/itbound * 100) + "% of gradient descent iterations done. Method = " + method)
                (_, newval_mc, _, grad_theta_new_mc) = objfun(F_theta = theta_avg,
                                           NN_object = NN_object, params = params, output_Gradient=True)
                supnorm_grad = np.linalg.norm(grad_theta_new_mc, ord = np.inf)     #max(abs(gradient))
                # print( "objective value function right now is: " + str(newval_mc))
                # print( "gradient value of function right now is: " + str(grad_theta_new_mc))
                # print( "supnorm grad right now is: " + str(supnorm_grad))
                # print("Weights right now are: ")
                # print(theta)


    # ---------------------------- End: MAIN LOOP --------------------------------------------

    #Append final values (not running min, just final values)
    if output_progress == True:
        temp_vval = pd.DataFrame([[it, newval]], columns=['it_nr', 'objfunc_val'])
        vval = vval.append(temp_vval)


    #--------------- SET OUTPUT VALUES ---------------


    F_theta = theta_min.copy()    #Use running minimum over the last 'nit_running_min' iterations

    (params, val, _, grad) = objfun(F_theta = F_theta,   # record the objective function value
                                 NN_object = NN_object, params = params, output_Gradient=True)
    supnorm_grad = np.linalg.norm(grad, ord=np.inf)  # max(abs(gradient))

    if params["obj_fun"] == "mean_cvar":
        NN_theta = F_theta[0:-2]
        xi = F_theta[-2]  # Second-last entry is xi, where (xi**2) is candidate VAR
        gamma = F_theta[-1]  # Lagrange multiplier

        # Make sure parameter dictionary is updated so that e.g. fun_Objective_functions can work correctly
        params["xi"] = xi
        params["gamma"] = gamma

    elif params["obj_fun"] == "mean_cvar_single_level":
        NN_theta = F_theta[0:-1]
        xi = F_theta[-1]  # Last entry is xi, where (xi**2) is candidate VAR

        #added theta cache
        optimal_params = {"NN":NN_theta.tolist()}
        with open('NN_optimal2.json', 'w') as outfile:
            json.dump(optimal_params, outfile)
        
        print("NN weights: " + str(NN_theta))
        print("Minimum obj value:" + str(val))
        print("Optimal xi: " + str(xi))


        # Make sure parameter dictionary is updated so that e.g. fun_Objective_functions can work correctly
        params["xi"] = xi #(xi**2) is candidate VAR

    else:
        NN_theta = F_theta

        #added theta cache
        optimal_params = {"NN":NN_theta.tolist()}
        with open('NN_optimal2.json', 'w') as outfile:
            json.dump(optimal_params, outfile)



    t_end = time.time()
    t_runtime = (t_end - t_start)/60    #we want to output runtime in MINUTES

    # ---------------------------- SET OUTPUT --------------------------------------------
    res["method"] = method
    res["F_theta"] = F_theta        #minimizer or point where algorithm stopped
    res["NN_theta"] = NN_theta        #minimizer or point where algorithm stopped
    res["nit"] = int(it)     #total nr of iterations executed to get res["NN_theta"]
    res["val"] = val    #objective function value evaluated at res["NN_theta"]
    res["supnorm_grad"] = supnorm_grad  # sup norm of gradient vector at res["NN_theta"], i.e. max(abs(gradient))
    res["runtime_mins"] = t_runtime  # run time in MINUTES until output is obtained

    #Append terminal wealth stats using this optimal value
    W_T = params["W"][:, -1]

    #Override W_T in one case
    if params["obj_fun"] == "one_sided_quadratic_target_error":  # only in this case
        if params["obj_fun_cashwithdrawal_TrueFalse"] == True:  #Check if we want values *after* cash withdrawal
            W_T = params["W_T_cashwithdraw"]



    W_T_stats_dict = fun_W_T_summary_stats(W_T)

    #Remove "W_T_summary_stats" dataframe
    del W_T_stats_dict['W_T_summary_stats']

    #Add summary stats to res dictionary

    res.update(W_T_stats_dict)

    #Put results in pandas.Dataframe for easy comparison with other methods
    W_T_stats_df = pd.DataFrame(data=W_T_stats_dict, index=[0])


    summary_df = pd.DataFrame([[method, it, val, supnorm_grad, t_runtime]],
                              columns=["method", "nit", "objfunc_val", "supnorm_grad", "runtime_mins"])

    summary_df = pd.concat([summary_df, W_T_stats_df], axis=1, ignore_index = False)


    res["summary_df"] = summary_df

    if output_progress == True:
        res["vval"] = vval    #pandas DataFrame outputted ONLY if  output_progress == True:


    return res