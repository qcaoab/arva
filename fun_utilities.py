#Code notes:

# #to RELOAD a python file for debugging in the console:
# #Suppose we had
# from lib import NN_optimize_SGD_algorithms
# from lib import NN_optimize_scipy_algorithms
# #and we make changes to the code, and want to rerun it in the Python console, then do
# import importlib
# importlib.reload(NN_optimize_SGD_algorithms)
# importlib.reload(NN_optimize_scipy_algorithms)

import numpy as np
import json

def abs_value_smooth_quad(x,lambda_param, output_Gradient = False):
    #Returns smooth approximation to |x| and its derivative using
    #       quadratic approx in interval [-lambda_param, lambda_param]
    # -- If x is a vector, then output in same shape as x
    # lambda_param >0 small, e.g. lambda_param = 1e-6


    abs_value = np.absolute(x)  #Initialize

    #Initialize gradient output
    grad_abs_value = None   #initialize to avoid output errors
    if output_Gradient is True:
        grad_above = (x >= 0) * 1.0
        grad_below = (x < 0) * (-1.0)
        grad_abs_value = grad_above + grad_below

    #Do SMOOTH APPROXIMATION

    if lambda_param > 0:    #Calculate smooth approximation
        ind_above = (x > lambda_param) * 1.0
        ind_below = (x < -lambda_param) * 1.0

        ind_outside = ind_above + ind_below #outside [-lambda_param, lambda_param]
        ind_middle = 1 - ind_outside #inside [-lambda_param, lambda_param]

        #Calculate values of the quadratic
        x_temp = np.multiply(ind_middle, x) #quadratic values will only be used in the middle anyway
        quad_value = (1/(2*lambda_param))*np.power(x_temp,2) + 0.5*lambda_param

        #Calculate smooth approximation value
        abs_value = np.multiply(ind_outside,abs_value) + np.multiply(ind_middle, quad_value)

        #Output (smooth) gradient
        if output_Gradient is True:
            grad_middle = (1/lambda_param)*x
            grad_abs_value = np.multiply(ind_outside,grad_abs_value) + np.multiply(ind_middle, grad_middle)

    return abs_value, grad_abs_value

def is_jsonable(x):
    try:
        json.dumps(x)
        return True
    except (TypeError, OverflowError):
        return False


def np_array_indices_SMALLEST_k_values_no_sorting(array, k):
    # Returns np.array of indices of the SMALLEST k values in array
    idx_SMALLEST_k = np.argpartition(array, kth=k)  #It does not sort the entire array.!
    idx_SMALLEST_k = idx_SMALLEST_k[:k].copy()

    return idx_SMALLEST_k


def np_array_indices_LARGEST_k_values_no_sorting(array, k):
    #Returns np.array of indices of the LARGEST k values in array

    idx_LARGEST_k = np.argpartition(array, kth=-k)  # It does not sort the entire array.!
    idx_LARGEST_k = idx_LARGEST_k[-k:].copy()   #-k to end

    return idx_LARGEST_k


def np_1darray_TO_string_comma_separated(x):
    #Converts numpy array e.g. x = [1,2,3] to string x_string "1,2,3"

    x = np.array(x)

    x_string = ','.join([str(num) for num in x])

    return x_string


def np_1darray_FROM_string_comma_separated(x_string):
    #Converts string e.g. x_string "1,2,3" to numpy array  x = [1,2,3]

    x = np.fromstring(x_string, sep=",")

    return x




def is_positive_definite(A):
    #Checks if a matrix A is symmetric(Hermitian) posdef using a Cholesky factorization

    #INPUTS:
    # A = matrix, as in a numpy array

    #OUTPUTS:
    #   True = matrix A is symmetric(Hermitian) posdef
    #   False = otherwise

    import numpy as np

    if np.array_equal(A, np.matrix.getH(A)): #Check if A is equal to its (conjugate) transpose
        #np.array_equal() returns True if two arrays have the same shape and elements, False otherwise.
        try:
            L = np.linalg.cholesky(A)   #returns lower-triangular Cholesky factor of A

            #if this works, return TRUE
            return True
        except np.linalg.LinAlgError:        #Generic Python-exception-derived object raised by linalg functions
            return False
    else:   #Matrix no symmetric (Hermitian)
        return False




