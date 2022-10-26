import numpy as np
import math

def fun_Activations(activation, u, output_Gradient = False, axis = 1, params = None):
    #INPUT: 'axis' is used for "softmax"
    # if u a matrix  (u.ndim = 2): 'axis' == 1, then each ROW of u is considered a different input vector for the softmax
    # if u a matrix  (u.ndim = 2): 'axis' == 0, then each COLUMN of u is considered a different input vector for the softmax


    if activation == None: #if no activation function (USED FOR INPUT LAYER)
        sig = u
        grad_sig = np.ones(u.shape)
        return sig, grad_sig

    elif activation == "forsyth_ground_truth": # (Used only for ground truth in benchmark paper)
        sig, grad_sig =  forsyth_ground_truth(u, output_Gradient)
        return sig, grad_sig

    elif activation == "scaled_logistic sigmoid":

        LowerBound = params["ScaledLog_LowerBound"]
        UpperBound = params["ScaledLog_Upperbound"]

        sig, grad_sig = scaled_logistic_sigmoid(u, output_Gradient,
                                                LowerBound = LowerBound,
                                                UpperBound = UpperBound)
        return sig, grad_sig

    elif activation == "linear_identity": #Also no activation, just a linear identity
        sig = u
        grad_sig = np.ones(u.shape)
        return sig, grad_sig

    elif activation == "ReLU":
        sig, grad_sig = ReLU(u, output_Gradient)
        return sig, grad_sig

    elif activation == "ELU":   # (scaled) exponential linear unit
        sig, grad_sig = ELU(u, output_Gradient)
        return sig, grad_sig

    elif activation == "logistic_sigmoid":
        sig, grad_sig = logistic_sigmoid(u, output_Gradient)
        return sig, grad_sig

    elif activation == "softmax":
        sig, grad_sig = softmax(u, output_Gradient, axis)
        return sig, grad_sig

    else:
        raise ValueError("PVS error in 'fun_Activations': selected activation not coded.")



def forsyth_ground_truth(u, output_Gradient = False):
    # Used only for ground truth in benchmark paper
    # We have 2 assets, ["T90", "VWD"].
    # Prop in T90 can be positive or negative, VWD positive only which we will assume \in [LowerBound, UpperBound]
    # See NOTES for how this works

    u = np.array(u)

    #Split inputs, effectively discard inputs into T90
    u_T90 = u[:, 0].copy()
    u_VWD = u[:, 1].copy()

    #Construct activation function
    LowerBound = 1e-6
    UpperBound = 20.


    phi, grad_phi = scaled_logistic_sigmoid(u = u_VWD,
                                            output_Gradient = output_Gradient,
                                            LowerBound = LowerBound,
                                            UpperBound = UpperBound)

    #Fraction in each asset
    sig_T90 = 1. - phi
    sig_VWD = phi.copy()

    sig_T90 = np.expand_dims(sig_T90, axis=1)
    sig_VWD = np.expand_dims(sig_VWD, axis=1)

    sig = np.concatenate((sig_T90, sig_VWD), axis = 1)

    #-----------------------------------------------------------
    #Do gradient
    # grad_sig =  jacobian matrix of output associated with each input vector,
    #             grad_sig[i,j,k] will be (j,k)th entry of Jacobian
    #                       evaluated at i-th input vector

    grad_sig = None  # To avoid output errors

    if output_Gradient is True:
        N_d = u.shape[0]    #Get number of input vectors

        grad_sig = np.zeros((N_d,2,2))  #Initialize


        grad_sig[:,0,1] = -1. * grad_phi    #NorthEast corners
        grad_sig[:, 1, 1] = grad_phi.copy()  # SouthEast corners


    return sig, grad_sig    #mixed_linear_sigmoid

def scaled_logistic_sigmoid(u, output_Gradient = False, LowerBound = None, UpperBound = None):
    #SCALED Logistic sigmoid at u
    # LowerBound and UpperBound gives bounds for the scaled logistic sigmoid

    # Definition:
    # LowerBound + (UpperBound - LowerBound) * [1 / ( 1 + exp(u) )]  for u \in R


    #INPUT: u a number or vector, converted to numpy array if it isn't already

    #RETURNS: sig = value of scaled logistic sigmoid at u
    #         grad_sig = derivative of scaled logistic sigmoid evaluated at u

    # Dimensions: scalar valued function, applied componentwise to u
    # sig.shape = grad_sig.shape =  u.shape

    u = np.array(u)

    # Get STANDARD logistic sigmoid evaluated at u
    f_logsig, grad_f_logsig = logistic_sigmoid(u, output_Gradient=output_Gradient)

    # SCALE logistic sigmoid
    sig = LowerBound + (UpperBound - LowerBound) * f_logsig

    grad_sig = None #Initialize To avoid output errors

    if output_Gradient== True:
        #Get gradient of scaled logistic sigmoid
        grad_sig = (UpperBound - LowerBound) * grad_f_logsig


    return sig, grad_sig    #scaled logistic sigmoid


def ELU(u, output_Gradient = False, alpha_scale = 1.0):
    # (Scaled) Exponential Linear Unit (SELU or ELU):
    #   u                           if u > 0
    #   alpha_scale * ( exp(u) - 1 ) if u <= 0  where alpha_scale > 0

    #NOTE:
    # SELU and ELU (exponential linear unit) names are often used interchangeably in the literature
    # Clevert Et Al (2016) calls the whole thing ELU

    #INPUT: u a number or vector, converted to numpy array if it isn't already

    #RETURNS: sig = value of ELU at u   (componentwise)
    #         grad_sig = derivative of ELU evaluated at u (componentwise)

    # Dimensions: scalar valued function, applied componentwise to u
    # sig.shape = grad_sig.shape =  u.shape


    u = np.array(u)
    sig = np.zeros(u.shape)


    # ELU
    pos_indicators = (u > 0) * 1.0
    neg_indicators = (u <= 0) * 1.0

    pos_u_values = np.multiply(pos_indicators, u)   #for numerical stability calc this first
    neg_u_values = np.multiply(neg_indicators, u)

    sig_pos_u = pos_indicators * pos_u_values
    sig_neg_u = neg_indicators * alpha_scale * ( np.exp(neg_u_values) - 1 )

    # Add together
    sig = sig_pos_u + sig_neg_u


    # GRADIENT ---------------------------------
    grad_sig = None #To avoid output errors

    if output_Gradient== True:
        #Gradient of sig(u)
        #   1                            if u > 0
        #   sig(u) + alpha_scale         otherwise

        grad_sig = pos_indicators *1.0      + \
                   neg_indicators * ( sig + alpha_scale)

    return sig, grad_sig        #ELU



def ReLU(u, output_Gradient = False):
    #ReLU = rectified linear unit: max(0,u)

    #INPUT: u a number or vector, converted to numpy array if it isn't already

    #RETURNS: sig = value of ReLU at u
    #         grad_sig = derivative of ReLU evaluated at u

    # Dimensions: scalar valued function, applied componentwise to u
    # sig.shape = grad_sig.shape =  u.shape

    u = np.array(u)
    sig = np.zeros(u.shape)


    # ReLU
    sig = np.maximum(0,u)


    grad_sig = None #To avoid output errors

    if output_Gradient== True:
        #Gradient of sig(u)
        #   1 if u >= 0
        #   0 otherwise

        grad_sig = (u >= 0) * 1.0

    return sig, grad_sig    #ReLU



def logistic_sigmoid(u, output_Gradient = False):
    #Logistic sigmoid at u
    # we define this as 1 / ( 1 + exp(u) )  for u \in R

    #INPUT: u a number or vector, converted to numpy array if it isn't already

    #RETURNS: sig = value of logistic sigmoid at u
    #         grad_sig = derivative of logistic sigmoid evaluated at u

    # Dimensions: scalar valued function, applied componentwise to u
    # sig.shape = grad_sig.shape =  u.shape

    u = np.array(u)
    sig = np.zeros(u.shape, dtype=np.float64)

    #Causing overflow issues
    # sig = np.multiply((u < 0), 1 / (1 + np.exp(u))) \
    #     + np.multiply((u >= 0), np.divide(np.exp(-u), (1 + np.exp(-u))))

    pos_indicators = (u >= 0) * 1
    neg_indicators = (u < 0) * 1

    pos_u_values = np.multiply(pos_indicators, u)
    neg_u_values = np.multiply(neg_indicators, u)

    # mc edits: change sign of x to make it match standard pytorch logistic
    sig_pos_u = np.divide(1, (1 + np.exp(-pos_u_values)))
    sig_pos_u = np.multiply(pos_indicators, sig_pos_u)  # Correct for zeros

    sig_neg_u = np.divide(np.exp(neg_u_values), (1 + np.exp(neg_u_values)))
    sig_neg_u = np.multiply(neg_indicators, sig_neg_u)  # Correct for zeros

    # Add together
    sig = sig_pos_u + sig_neg_u

    grad_sig = None #To avoid output errors

    if output_Gradient== True:
        grad_sig = np.multiply(sig, sig - 1)    #will subtract one from each element


    return sig, grad_sig    #logistic sigmoid

def softmax(u, output_Gradient = False, axis = 1):

    #INPUT: u a vector or a matrix, converted to numpy array if it isn't already
    # u.ndim == 1 (vector) or u.ndim = 2 (matrix)

    # if u a vector (u.ndim == 1): 'axis' doesn't make a difference
    # if u a matrix  (u.ndim = 2): 'axis' == 1, then each ROW of u is considered a different input vector for the softmax
    #                                       i.e. the vectors are each pointing across the columns (axis = 1 in numpy)
    # if u a matrix  (u.ndim = 2): 'axis' == 0, then each COLUMN of u is considered a different input vector for the softmax

    #RETURNS: sig =  value (vector) of softmax for each input vector (same size as input vector)
    #         grad_sig = Jacobian matrix of softmax evaluated at each input vector

    # Dimensions: vector valued function:
    # sig  =  softmax for each input vector, returned in same shape as u (i.e. the input vectors)
    # grad_sig =  jacobian matrix of softmax associated with each input vector,
    #             grad_sig[i,j,k] will be (j,k)th entry of Jacobian of softmax
    #                       evaluated at i-th input vector


    u = np.array(u)



    if u.ndim == 2 and  axis == 0:  #if u is a matrix and each column is a different input vector
        u = np.transpose(u) #so that all the calcs below can proceed under the assumption that axis == 1
                            # results for sig will be transposed back prior to output

    #each ROW of u is considered a different input vector for the softmax
    if u.ndim == 2: #u a matrix
        K = - np.amax(u, axis=1 , keepdims=True)   #For numerical stability, take max of each row
        u = u + K   #Shift for purposes of stability

        num = np.exp(u)           #applied elementwise
        den = np.sum(np.exp(u), axis = 1, keepdims= True)

    elif u.ndim == 1:  #u a vector
        K = - np.max(u)  # For numerical stability, take max
        u = u + K  # Shift for purposes of stabilitygrad
        num = np.exp(u)  # applied elementwise
        den = np.sum(np.exp(u))

    else:
        raise ValueError("PVS error in 'softmax': input can only be a vector or a matrix.")


    sig = num/den       #softmax output

    grad_sig = None #To avoid output errors

    if output_Gradient== True: #Outputs Jacobian of softmax

        if u.ndim == 2: #u a matrix
            n_rows = sig.shape[0]
            row_dim = sig.shape[1]

            temp = np.expand_dims(sig, axis=1) #Insert a new axis that will appear at the
                                            # 'axis' position in the expanded array shape.
            #temp.shape = (n_rows, 1, row_dim)
            #temp[:,0,:] is the data in sig, but each explicitly as a 1x(row_dim) ROW vector

            diag = temp * np.identity(row_dim)

            #diag.shape = (n_rows, row_dim, row_dim)
            #diag contains n_rows diagonal matrices, each vector in temp on the diagonal

            temp_as_col_vectors = np.transpose(temp, axes=(0,2,1))
            #temp but with last 2 dimensions switched, so now each vector in sig is a column vector

            outer_prod = np.matmul(temp_as_col_vectors,
                                   np.ones(temp.shape))
            #np.matmul: If either argument is N-D, N > 2, it is treated as a stack of matrices
            # residing in the LAST 2 indices and broadcast accordingly.
            # temp_as_col_vectors.shape = (n_rows, row_dim, 1)
            # temp.shape = (n_rows, 1, row_dim)
            # outer_prod = (n_rows, row_dim, row_dim): i.e. we perform outer product on last 2 indices

            outer_prod = np.transpose(outer_prod, axes=(0,2,1))
            # Transpose stack of matrices in last 2 dimensions

            #Create stack of (row_dim, row_dim) identities
            temp_sig_ones = np.ones(temp.shape)
            temp_identities = temp_sig_ones * np.identity(row_dim)


            bracket = temp_identities - outer_prod

            grad_sig = np.matmul(diag, bracket)
            # grad_sig[i,j,k] will be (j,k)th entry of Jacobian of softmax
            #   evaluated at row i of sig

        elif u.ndim == 1:  # u a vector
            diag = np.diag(sig, k=0)
            outer_prod = np.outer(sig, np.ones(sig.shape))
            outer_prod = np.transpose(outer_prod)
            bracket = np.identity(len(sig)) - outer_prod
            grad_sig = np.matmul(diag, bracket)


    if axis == 0: #we transposed it above, so transpose before output
        sig = np.transpose(sig)

        #Note: grad_sig is NOT transposed
        #grad_sig always assumes the data is in ROWS;

        #if axis == 0:
        # grad_sig[i,j,k] will be (j,k)th entry of Jacobian of softmax
        #   evaluated at COLUMN i of sig


    return sig, grad_sig