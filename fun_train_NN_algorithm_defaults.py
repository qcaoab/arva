import numpy as np

def train_NN_algorithm_defaults():
    #OBJECTIVE: returns dictionary "NN_training_options" with algorithm hyperparameters and default options
    # can be changed/overridden in main code


    NN_training_options = {}

    # Specify METHODS used to train NN:
    #       *smallest* objective func value returned by ANY of the methods specified is used as the final value
    #        along with its associated parameter vector
    #       -> can specify multiple methods
    NN_training_options["methods"] = ["SGD_constant",
                                       "Adagrad",
                                       "Adadelta",
                                       "RMSprop",
                                       "Adam",
                                       "CG",
                                       "BFGS",
                                       "Newton-CG",
                                       ]
    NN_training_options["output_progress"] = False  # output_progress option
    NN_training_options["tol"] = 1e-06

    #Options specific to scipy algorithms in  training_options["methods"]
    NN_training_options["itbound_scipy_algorithms"] = 1000  # max nr of iterations for scipy.optimize.minimize algorithms

    #Options specific to *all* SGD algorithms in  training_options["methods"]
    NN_training_options["check_exit_criteria"] = False #if False: means we run code until itbound,
                                                        # do not check exit criteria on each iteration
    NN_training_options["nit_running_min"] = 1  #500 the final nr of iterations for running min calc
    NN_training_options["itbound_SGD_algorithms"] = 10000  # max nr of iterations
    NN_training_options["nit_IterateAveragingStart"] = None  #the point from which ONWARDS we do iterate averaging in the SGD calc
                                                        # = None to switch IA off
    NN_training_options["batchsize"] = 10 #batch size to determine gradient in SGD/mini batch descent

    #Options specific to *particular* SGD algorithms in  training_options["methods"]
    NN_training_options["SGD_learningrate"] = 50.0  # 50.0 SGD Learning rate constant
    NN_training_options["Adagrad_epsilon"] = 1e-08  # 1e-08 Adagrad small constant in denominator to avoid div by zero
    NN_training_options["Adagrad_eta"] = 1.0  # 1.0 Adagrad numerator of the adaptive learning rate
    NN_training_options["Adadelta_ewma"] = 0.9  # 0.9 Adadelta EWMA parameter for gradient squared AND for the parameter vector squared
    NN_training_options["Adadelta_epsilon"] = 1e-08  # 1e-08 Adadelta small constant in denominator to avoid div by zero
    NN_training_options["RMSprop_ewma"] = 0.8  # 0.8 RMSprop EWMA parameter for RMSprop (Hinton suggests 0.9)
    NN_training_options["RMSprop_epsilon"] = 1e-08  # 1e-08 RMSprop small constant in denominator to avoid div by zero
    NN_training_options["RMSprop_eta"] = 0.1  # 0.1 RMSprop numerator of the adaptive learning rate
    NN_training_options["Adam_ewma_1"] = 0.9  # 0.9 Adam (beta1) EWMA parameter for gradient (momentum) process
    NN_training_options["Adam_ewma_2"] = 0.999   # 0.999 or 0.9 for some problems: Adam (beta2) EWMA parameter for gradient SQUARED (speed) process
    NN_training_options["Adam_eta"] = 0.1 # 0.1 Adam eta in the numerator of the Adam learning rate
    NN_training_options["Adam_epsilon"] = 1e-08  # 1e-08 Adam small constant in denominator to avoid div by zero

    return NN_training_options

