import time
import numpy as np
from scipy.optimize import minimize
import pandas as pd

# objective function for NN investment strategy evaluation
from fun_eval_objfun_NN_strategy import eval_obj_NN_strategy_pyt as objfun
from fun_eval_objfun_NN_strategy import fun_val as objfun_val   #Function value ONLY
from fun_eval_objfun_NN_strategy import fun_gradient as objfun_gradient #Gradient  ONLY

from fun_W_T_stats import fun_W_T_summary_stats


def run_scipy_minimize(method,
                       theta0,       # initial parameter vector (weights and biases) of form
                                        # [NN_theta, extra_theta] where NN_theta = NN_object.NN_theta
                                        #  extra_theta = ADDITIONAL parameters for objective function used by e.g. mean-cvar
                                        # and extra_theta[xi, gamma] in the case of mean-CVAR
                       NN_object,       #object of class_Neural_Network with structure as setup in main code
                       params,          # dictionary with investment parameters as set up in main code
                       itbound,         #max nr of iterations to run if exit criteria not yet met
                       tol = 1e-06,
                       output_progress = False
                       ):
    # OBJECTIVE: Executes methods from scipy.optimize.minimize for the purposes of
    #           NN optimization and gives same format output function 'run_Gradient_Descent'

    # ------------------------------------------------------------------------------------------
    # OUTPUTS:
    # returns dictionary "res" with content:
    # res["method"] = method
    # res["F_theta"] = objective function parameters, [NN_theta, extra_theta]
    #                where NN_theta = NN parameter vector that is minimizer of objfun or point where algorithm stopped
    # res["NN_theta"] = NN_theta  # NN parameters forming part of F_theta
    # res["nit"] = it     #total nr of iterations executed to get res["NN_theta"]
    # res["val"] = val    #objective function value evaluated at res["NN_theta"]
    # res["supnorm_grad"] = supnorm_grad  # sup norm of gradient vector at res["NN_theta"], i.e. max(abs(gradient))
    # res["runtime_mins"] = t_runtime  # run time in MINUTES until output is obtained
    # res["summary_df"] = summary_df, where:
    #           summary_df = pd.DataFrame([[method, it, val, supnorm_grad, t_runtime,
    #                             and summary stats of terminal wealth


    #------------------------------------------------------------------------------------------
    # INPUTS:

    # method = "CG": uses a nonlinear conjugate gradient algorithm by Polak and Ribiere,
    #               a variant of the Fletcher-Reeves method
    #               first derivatives only

    # method='BFGS': Broyden-Fletcher-Goldfarb-Shanno algorithm, quasi-Newton method,
    #               first derivatives only

    # method='Newton-CG': Newton-Conjugate-Gradient algorithm,
    #                       uses a CG method to the compute the search direction
    #                   Good for big problems, no explicit Hessian inversion/factorization

    # method='trust-exact':  for smaller problems, solution within fewer iteration by
    #                          solving the trust-region subproblems almost exactly.
    #                       Appears to be closest to Matlab implementation.

    # method='trust-ncg': Trust-Region Newton-Conjugate-Gradient Algorithm
    #                      uses a conjugate gradient algorithm to solve the trust-region subproblem
    #                   Good for big problems, no explicit Hessian inversion/factorization

    # method='trust-krylov': Trust-Region Truncated Generalized Lanczos / Conjugate Gradient
    #                       a method suitable for large-scale problems as it uses the hessian only
    #                       as linear operator by means of matrix-vector products.
    #                      It solves the quadratic subproblem more accurately than the trust-ncg method
    #                   Good for big problems, no explicit Hessian inversion/factorization

    # NOTES: Newton-CG, trust-ncg and trust-krylov are suitable for dealing
    # with large-scale problems (problems with thousands of variables).
    # That is because the conjugate gradient algorithm approximately solve the trust-region subproblem
    # (or invert the Hessian) by iterations without the explicit Hessian factorization.


    t_start = time.time() #start keeping track of runtime
    res = {} #Initialize output dictionary

    #Set up general options dictionary for output
    options_dic = {'maxiter': itbound,
                   'disp': output_progress}



    # --------------- Run algorithms ---------------


    #Methods with only first derivatives or where Hessian is optional:
    if method == "CG" or method == "BFGS":
        options_dic.update([('gtol', tol)])

        res = minimize(fun=objfun_val,
                       x0=theta0,
                       args=(NN_object, params,),
                       method= method,
                       jac=objfun_gradient,
                       tol=tol,
                       options=options_dic)

    elif method == 'Newton-CG':
        options_dic.update([('xtol', tol)])

        res = minimize(fun=objfun_val,
                       x0=theta0,
                       args=(NN_object, params,),
                       method= method,
                       jac=objfun_gradient,
                       tol=tol,
                       options=options_dic)

    else:
        raise ValueError("PVS error in run_scipy_minimize.py: \
                        Method selected not in list of algorithms implemented.")





    #--------------- SET FINAL VALUES ---------------
    it = res.nit
    F_theta = res.x

    if params["obj_fun"] == "mean_cvar":
        NN_theta = F_theta[0:-2]
        xi = F_theta[-2]        # Second-last entry is xi, where (xi**2) is candidate VAR
        gamma = F_theta[-1]     # Lagrange multiplier

        #Make sure parameter dictionary is updated so that e.g. fun_Objective_functions can work correctly
        params["xi"] = xi
        params["gamma"] = gamma

    elif params["obj_fun"] == "mean_cvar_single_level":
        NN_theta = F_theta[0:-1]
        xi = F_theta[-1]  # Last entry is xi, where (xi**2) is candidate VAR

        # Make sure parameter dictionary is updated so that e.g. fun_Objective_functions can work correctly
        params["xi"] = xi

    else:
        NN_theta = F_theta

    (params, val, _, grad) = objfun(F_theta = F_theta,   # record the objective function value
                                 NN_object = NN_object, params = params, output_Gradient=True)
    supnorm_grad = np.linalg.norm(grad, ord=np.inf)  # max(abs(gradient))

    if val == res.fun:
        print("Recalculated val == res.fun")


    t_end = time.time()
    t_runtime = (t_end - t_start)/60    #we want to output runtime in MINUTES

    # ---------------------------- SET OUTPUT --------------------------------------------
    res = {}    #initialize output dictionary (otherwise it returns ALL the fields from the scipy.optimize results)
    res["method"] = method
    res["F_theta"] = F_theta        #minimizer or point where algorithm stopped
    res["NN_theta"] = NN_theta      #NN parameters forming part of F_theta
    res["nit"] = int(it)     #total nr of iterations executed to get res["NN_theta"]
    res["val"] = val    #objective function value evaluated at res["NN_theta"]
    res["supnorm_grad"] = supnorm_grad  # sup norm of gradient vector at res["NN_theta"], i.e. max(abs(gradient))
    res["runtime_mins"] = t_runtime  # run time in MINUTES until output is obtained

    #Append terminal wealth stats using this optimal value
    if params["obj_fun"] == "one_sided_quadratic_target_error":  # only in this case
        W_T = params["W_T_cashwithdraw"]
    else:
        W_T = params["W"][:,-1]

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


    return res