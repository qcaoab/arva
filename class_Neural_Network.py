
import numpy as np
import pandas as pd

#----------------------------------------------------------------------------------------
class Neural_Network(object):
    def __init__(self, n_nodes_input, n_nodes_output, n_layers_hidden):
        #Basic parameters
        self.n_nodes_input = n_nodes_input  #nr of input nodes = size of feature vector
        self.n_nodes_output = n_nodes_output    #nr of output nodes
        self.L = n_layers_hidden  # nr of hidden layers
        self.n_layers_total = n_layers_hidden + 2   #hidden + input + output; useful for looping

        self.layers = []     #List containing a total of L+2 "Neural_Network_Layer" objects
        self.theta = None   #single parameter vector for NN, containing stacked weights matrices and bias vectors
        self.theta_length = None    #length of single parameter vector 'theta' for NN
        self.x_length = None    #length of vector x containing all the weights, i.e. total number of weights in the NN


        #Can only add layers as part of initialization
        def add_layer(layer_id, n_nodes=None, activation=None, description=""):
            # Constructs a Neural_Network_Layer object
            #   - basically just runs __ilnit__() for class Neural_Network_Layer
            return Neural_Network_Layer(layer_id, n_nodes, activation, description)

        #Construct initial Neural_Network.layers list
        for l in np.arange(0, self.n_layers_total, 1):
            if l == 0: #input layer
                self.layers.append( add_layer(layer_id = l,
                                               n_nodes = n_nodes_input,
                                               activation=None,
                                               description = "input_layer")
                )


            elif l == self.n_layers_total-1: #output layer
                self.layers.append(add_layer(layer_id = l,
                                               n_nodes = n_nodes_output,
                                               activation= None,
                                               description = "output_layer")
                )


            else:   #hidden layers
                 self.layers.append(add_layer(layer_id = l,
                                                n_nodes = None,
                                                activation= None,
                                                description = "hidden_layer")
                 )



    def update_layer_info(self, layer_id, n_nodes = None, activation = "", description = "", add_bias = False):
        #Objective: updates the info in self.layers[layer_id]
        #   Due to IF statements, updates only with selected arguments provided, not all of it
        #   CANNOT update layer_id in this method

        if n_nodes is not None:
            self.layers[layer_id].n_nodes = n_nodes       #nr of nodes for layer


        if activation != "":    #Activation can be set to None!
            if self.layers[layer_id].layer_id == 0:
                raise ValueError("PVS error: No activation function for input layer.")
            else:
                self.layers[layer_id].activation = activation #string giving activation function for layer

        if description != "":
            self.layers[layer_id].description = description  #layer description for ease of reference

        if add_bias is True:    #Add bias vector
            if self.layers[layer_id].layer_id == 0:
                raise ValueError("Bias cannot be added to input layer.")
            else:
                self.layers[layer_id].add_bias = True

        elif add_bias is False:
            self.layers[layer_id].add_bias = False
            self.layers[layer_id].b_l = None

        #After the above, run function self.update_shape_weights_biases
        self.update_shape_weights_biases()

    def update_shape_weights_biases(self):
        #If n_nodes attribute of a layer is updated, this method reshapes
        # the weights matrix (x) and bias vector (b) associated with layer

        for layer_id in np.arange(0, self.n_layers_total, 1):

            if layer_id == 0:
                #No weights matrix or bias vector for input layer
                #   Do this just to confirm
                self.layers[layer_id].x_l = None
                self.layers[layer_id].x_l_shape = None

                self.layers[layer_id].add_bias = False
                self.layers[layer_id].b_l = None
                self.layers[layer_id].b_l_length = None


            else:
                prev_layer_n_nodes = self.layers[layer_id-1].n_nodes
                current_layer_n_nodes = self.layers[layer_id].n_nodes

                if prev_layer_n_nodes is None or current_layer_n_nodes is None:
                    #Set to None to avoid meaningless matrix
                    self.layers[layer_id].x_l = None
                    self.layers[layer_id].x_l_shape = None

                else:
                    #Update shape of weights matrix for this layer
                    self.layers[layer_id].x_l_shape = (prev_layer_n_nodes, current_layer_n_nodes)
                    self.layers[layer_id].x_l = np.zeros( self.layers[layer_id].x_l_shape )

                if current_layer_n_nodes is None:
                    self.layers[layer_id].b_l = None
                    self.layers[layer_id].b_l_length = None
                else:
                    # Update shape of bias vector for this layer
                    if self.layers[layer_id].add_bias is True:
                        self.layers[layer_id].b_l_length = current_layer_n_nodes
                        self.layers[layer_id].b_l = np.zeros(current_layer_n_nodes)

        #After updating shapes of weights, and biases, update single parameter vector 'theta' shape + initialize
        self.stack_NN_parameters()

    def initialize_NN_parameters(self, initialize_scheme = None):
        #return theta0

        #Objective: Initializes the NN parameters according to scheme given by "initialize_scheme"
        # -- bias vectors (if applicable) are always initialized zero

        #OUTPUT: theta0 = single stacked parameter vector of NN containing initial values

        #INPUTS: initialize_scheme:
        #   "zeros": zeros in all weight matrices
        #   "xavier_he": Xavier/He initialization   [DEFAULT!]
        #   "glorot_bengio": Glorot and Bengio (2010) initialization
        #   "matlab": Initialization scheme used in Matlab, ONLY for 2 assets, 2-3-2 NN,  non-random


        #Make sure the right shapes are used (this also intializes all to zero), and restack values
        self.update_shape_weights_biases()


        #Set default
        if initialize_scheme is None:
            initialize_scheme = "glorot_bengio"


        #Do initialization
        if initialize_scheme == "zeros":
            theta0 = self.theta     #first step above 'update_shape_weights_biases' initializes all to zero

        elif initialize_scheme == "xavier_he" or initialize_scheme == "glorot_bengio":
            for layer_id in np.arange(1, self.n_layers_total, 1):

                # get shape of weights matrix to be initialized
                init_shape = self.layers[layer_id].x_l_shape

                # nr of inputs + nr of outputs associated with layer
                init_nrnodes = self.layers[layer_id-1].n_nodes + self.layers[layer_id].n_nodes


                if initialize_scheme == "xavier_he": #Xavier/He initialization

                    # (Xavier/He) location, scale for random normals
                    init_loc = 0.0
                    init_scale = np.sqrt(2 / init_nrnodes)

                    self.layers[layer_id].x_l = np.random.normal(loc = init_loc,
                                                                 scale= init_scale,
                                                                 size= init_shape
                                                                 )

                elif initialize_scheme == "glorot_bengio":

                    # (Glorot/Bengio) [low, high] for uniform randoms
                    init_low = - np.sqrt(6 / init_nrnodes)
                    init_high = np.sqrt(6 / init_nrnodes)

                    self.layers[layer_id].x_l = np.random.uniform(low = init_low,
                                                                  high= init_high,
                                                                  size= init_shape)

            #After looping through layers and initializing according to these schemes
            self.stack_NN_parameters()  #flatten matrices and update attribute self.theta
            theta0 = self.theta #copy into output

        elif initialize_scheme == "matlab": # USE MATLAB intitial value (only for 2 assets, no biases, 2-3-2 NN)
            if self.n_nodes_output != 2:
                raise ValueError("PVS error in 'initialize_NN_parameters': Matlab initialization only for 2 assets.")
            else:
                theta0 = np.zeros(self.theta_length)
                theta0[self.layers[1].n_nodes *2] = 1.0
                theta0[0] = 0.0

                self.theta = theta0.copy()
                self.unpack_NN_parameters() #copy into NN weights and biases



        return theta0


    def stack_NN_parameters(self):
    #Objective: flatten all the weights matrices and bias vectors of the NN
    #           into a single parameter vector 'theta' and update attribute self.theta

    # Weights: Create single vector x as follows:
    #          stack each layer's weights matrix x_l by COLUMN,
    #          with col 1 of the matrix x_l on top of x_l_stacked
    #          then we stack x_l column vectors, such that output layer is on top, and input layer at the bottom

    # Biases:  Create single vector b as follows:
    #           stack b_l column vectors, such that output layer is on top, and input layer at the bottom

    # Theta: Single vector of all parameters, created as follows:
    #        stack single weights vector x on top, then single bias vector b at the bottom

        x = None    # initialize single vector x
        b = None    # initialize single vector b
        theta = None    #intialize local single parameter vector for NN

        #loop backwards from output layer
        for layer_id in np.arange(self.n_layers_total-1, -1, -1):
            #Create copy to make it easier to work with
            x_l = self.layers[layer_id].x_l
            b_l = self.layers[layer_id].b_l

            #Stack weights matrices into vector x
            if x_l is not None:
                x_l_stacked = np.ndarray.flatten(x_l, order="F")
                #       F means to flatten in column-major order, i.e. stack columns, with col 1 on top

                #Append weights matrix to single weights vector
                if x is None: #if there are no weights matrices stacked yet
                    #create
                    x = x_l_stacked.copy()

                else: #otherwise, append
                    x = np.append(x, x_l_stacked) #layer l on top of layer (l-1)

            #Stack bias vectors
            if b_l is not None:
                # Append bias vector to single bias vector
                if b is None:  # if there are no bias vectors stacked yet
                    # create
                    b = b_l.copy()

                else:  # otherwise, append
                    b = np.append(b, b_l)  # layer l on top of layer (l-1)
        #- end: loop through layers

        if x is not None:
            if b is None:
                theta = x.copy()
            else:
                theta = np.append(x,b) #weights vector x at the top, bias vector at the bottom
        elif x is None:
            if b is not None:
                theta = b.copy()

        #copy into attribute or update
        if theta is not None:
            self.theta = theta.copy()
            self.theta_length = len(theta)
            self.x_length = len(x)
        else:
            self.theta = None
            self.theta_length = None
            self.x_length = None


    def unpack_NN_parameters(self):
    # Objective: unpack single parameter vector self.theta into individual weights matrices and bias vectors of the NN

        if self.theta is None:
            raise ValueError("PVS error: in 'unpack_NN_parameters', cannot unpack empty parameter vector.")
        else:
            #create a local copy of theta, so that original is not affected
            theta = self.theta.copy()

        #Weights: loop backwards from output layer, since output layer on top
        for layer_id in np.arange(self.n_layers_total-1, -1, -1):
            x_l_shape = self.layers[layer_id].x_l_shape

            if x_l_shape is not None:
                x_l_stacked_length =x_l_shape[0] * x_l_shape[1]

                #Extract x_l_stacked vector from theta
                x_l_stacked = theta[0:x_l_stacked_length].copy()

                #Reshape x_l_stacked into weights matrix x_l and update attribute
                x_l = np.reshape(x_l_stacked, x_l_shape, order='F')  # Fortran-like index ordering, i.e. do this by column
                self.layers[layer_id].x_l = x_l.copy()

                #Delete x_l_stacked from theta so that new layer is on top
                theta = np.delete(theta, np.arange(0, x_l_stacked_length, 1))

        #Biases: loop backwards from output layer, since output layer on top
        for layer_id in np.arange(self.n_layers_total-1, -1, -1):
            b_l_length = self.layers[layer_id].b_l_length

            if b_l_length is not None:
                #Extract b_l vector from theta and update attribute
                b_l = theta[0:b_l_length].copy()
                self.layers[layer_id].b_l = b_l.copy()

                #Delete b_l from theta so that new layer is on top
                theta = np.delete(theta, np.arange(0, b_l_length, 1))

        #Check that local copy of theta has been allocated fully once loops are done
        if len(theta) > 0:
            raise ValueError("PVS error: in 'unpack_NN_parameters', parameters remain after unpacking.")


    def print_layers_info(self):
        #Prints Neural_Network.layers list in a way that makes it easy to update

        layers_info_df = pd.DataFrame(columns=["obj.layers[layer_id]",
                                             "layer_id",
                                             "description",
                                             "n_nodes",
                                             "activation",
                                             "x_l(weights)",
                                              "add_bias",
                                             "b_l(biases)"
                                             ])

        print("--------------------------------------------------------------------------------------------------")
        print("Neural_Network object 'obj' has instance attribute obj.layers.")
        print("obj.layers is a list of 'Neural_Network_Layer' objects with following attributes:\n")
        for l in np.arange(0, self.n_layers_total, 1):
            layer_dict = {"obj.layers[layer_id]" : "obj.layers[" + str(l) + "]",
                          "layer_id" : self.layers[l].layer_id,
                          "description": self.layers[l].description,
                          "n_nodes" : self.layers[l].n_nodes,
                          "activation":  self.layers[l].activation,
                          "x_l(weights)":  self.layers[l].x_l_shape,
                          "add_bias" : self.layers[l].add_bias,
                          "b_l(biases)" : self.layers[l].b_l_length}

            layers_info_df = layers_info_df.append(layer_dict, ignore_index=True)

        pd.set_option("display.max_rows", None, "display.max_columns", None)
        print(layers_info_df)
        print("\n Run 'update_layer' method of Neural_Network object to change layer attributes.")
        print("--------------------------------------------------------------------------------------------------")

    def PRPscores_back_propagation(self, phi, a_layers_all, PRP_eps_1, PRP_eps_2):

    # OBJECTIVE: Calculate PRP (Portfolio relevance propagation) scores via backpropagation
    # Inputs:
    #   phi = required input/feature vector (if adjusted/standardized, must be done PRIOR to using it here)
    #      i.e. every ROW of phi must correspond to ONE input/feature vector in the training data
    #      if phi is a matrix, ROW (j) corresponds to the feature vector
    #      for the jth data point/sample path in training data
    # a_layers_all = DICTIONARY in format {layer_id : a}, where a is the output of the layer with layer_id
    #   PRP_eps_1, PRP_eps_2 >=0 gives (1) numerical stability for unimportant features, and
    #                                   (2) can result in sparsity of relevance scores if PRP_epsilon is large

    # OUTPUTS:
    # R_score.shape = phi.shape
    #   R_score[j,i] PRP relevance score, along sample path j for feature i

        self.unpack_NN_parameters()     #make sure we have latest weights and biases

        #INITIALIZE matrix of relevance scores R_score
        # R_score.shape = phi.shape
        #   R_score[j,i] relevance score, along sample path j for feature i
        R_score = np.zeros(phi.shape)

        # --------------------- PRP relevance scores backprop loop -------------------------

        for layer_id in np.arange(self.n_layers_total - 1, -1, - 1):
            #   loop starts at output layer: layer_id = self.n_layers_total - 1
            #   loop last execution is at the INPUT layer: layer_id = 0

            if layer_id == self.n_layers_total - 1:  # At the OUTPUT layer

                R_score = a_layers_all[layer_id]    #at output layer, relevance scores = *outputs* of softmax

            elif layer_id < self.n_layers_total - 1:  # if we are NOT at OUTPUT layer

                R_score_l_plus_1 = R_score.copy()   #R_score at next layer

                #Get information from forward propagation
                x_l_plus_1 = self.layers[layer_id + 1].x_l  # weights into *next* layer,  layer_id + 1
                a_l = a_layers_all[layer_id] # outputs out of this layer

                #Get number of nodes in this layer and the next
                n_l_plus_1 = self.layers[layer_id + 1].n_nodes
                n_l = self.layers[layer_id].n_nodes

                #Create stack of matrices
                N_d = a_l.shape[0]
                a_l_matrix = np.zeros([N_d, n_l, n_l_plus_1])

                #Populate  a_l_matrix column-by-column (safer than tile)
                for col in np.arange(0, n_l_plus_1,1):
                    a_l_matrix[:,:,col] = a_l

                #Create omega matrix
                omega = np.multiply(a_l_matrix, x_l_plus_1)


                #Get the positive and negative elements
                M_1 = np.maximum(omega, 0)
                M_2 = np.minimum(omega, 0)


                #Sum the rows of each matrix M
                S_1 = np.sum(M_1, axis=1)
                S_2 = np.sum(M_2, axis=1)

                #Do elementwise division after incorporating PRP_eps_2
                D_1 = np.divide(1, PRP_eps_2 + S_1)
                D_2 = np.divide(1, PRP_eps_2 - S_2)


                #Construct the matrices C_1 and C_2, block by block
                C_1 = np.zeros([N_d, n_l, n_l_plus_1])
                C_2 = np.zeros([N_d, n_l, n_l_plus_1])
                for i in np.arange(0, n_l,1):
                    for j in np.arange(0,n_l_plus_1,1):
                        C_1[:,i,j] = np.multiply(M_1[:,i,j], D_1[:,j])
                        C_2[:, i, j] = np.multiply(M_2[:, i, j], D_2[:, j])


                #Construct big_PHI column by column
                C_diff = C_1 - C_2
                big_PHI =  np.zeros([N_d, n_l])
                for i in np.arange(0, n_l,1):
                    for j in np.arange(0,n_l_plus_1,1):
                        big_PHI[:,i] = big_PHI[:,i] + np.multiply(C_diff[:,i,j], R_score_l_plus_1[:,j])

                #Sum the columns of big_PHI
                S_3 = np.sum(big_PHI, axis=1)

                #Get D_3 by incorporating PRP_eps_1
                D_3 = np.divide(1, PRP_eps_1 + S_3)

                # Calculate relevance score R_score for this layer column by column
                R_score = np.zeros([N_d, n_l])
                for i in np.arange(0,n_l,1):
                    R_score[:,i] = np.multiply(big_PHI[:,i], D_3)

        return R_score


    def LRPscores_back_propagation(self, phi, a_layers_all, z_layers_all, LRP_epsilon):
    # OBJECTIVE: Calculate LRP (layerwise relevance propagation) scores via backpropagation
    # Inputs:
    #   phi = required input/feature vector (if adjusted/standardized, must be done PRIOR to using it here)
    #      i.e. every ROW of phi must correspond to ONE input/feature vector in the training data
    #      if phi is a matrix, ROW (j) corresponds to the feature vector
    #      for the jth data point/sample path in training data
    # a_layers_all = DICTIONARY in format {layer_id : a}, where a is the output of the layer with layer_id
    # z_layers_all = DICTIONARY in format {layer_id : z_l}, where
    #                  z_l is the weighted input into each node in layer layer_id
    #   LRP_epsilon gives (1) numerical stability for unimportant features, and
    #                     (2) can result in sparsity of relevance scores if LRP_epsilon is large


    #OUTPUTS:
    # R_score.shape = phi.shape
    #   R_score[j,i] relevance score, along sample path j for feature i

        self.unpack_NN_parameters()     #make sure we have latest weights and biases

        #INITIALIZE matrix of relevance scores R_score
        # R_score.shape = phi.shape
        #   R_score[j,i] relevance score, along sample path j for feature i
        R_score = np.zeros(phi.shape)

        # --------------------- LRP relevance scores backprop loop -------------------------

        for layer_id in np.arange(self.n_layers_total - 1, -1, - 1):
            #   loop starts at output layer: layer_id = self.n_layers_total - 1
            #   loop last execution is at the INPUT layer: layer_id = 0

            if layer_id == self.n_layers_total - 1:  # At the OUTPUT layer

                R_score = z_layers_all[layer_id]    #at output layer, relevance scores = weighted inputs into node


            if layer_id != self.n_layers_total - 1:  # if we are NOT at OUTPUT layer

                #Get information from forward propagation
                z_l_plus_1 = z_layers_all[layer_id + 1]  # weighted inputs into *next* layer,  layer_id + 1
                x_l_plus_1 = self.layers[layer_id + 1].x_l  # weights into *next* layer,  layer_id + 1
                a_l = a_layers_all[layer_id] # outputs out of this layer

                #Calculate matrix s_l_plus_1 using elementwise division
                s_l_plus_1 = np.divide(R_score, LRP_epsilon + z_l_plus_1)


                #Calculate matrix c_l_plus_1 using matrix multiplication
                c_l_plus_1 = np.matmul(s_l_plus_1, np.transpose(x_l_plus_1))


                #Calculate relevance score R_score for this layer using elementwise multiplication
                R_score = np.multiply(a_l, c_l_plus_1)


        return R_score


    def back_propagation(self, grad_h_a, a_layers_all, z_layers_all):
    # OBJECTIVE: Do backpropagation of NN

    #Inputs:
    # grad_h_a = gradient of function h(a) w.r.t. a, where a is output at terminal layer of NN
    # a_layers_all = DICTIONARY in format {layer_id : a}, where a is the output of the layer with layer_id
    # z_layers_all = DICTIONARY in format {layer_id : z_l}, where
    #                  z_l is the weighted input into each node in layer layer_id

    #OUTPUTS:
    # grad_h_theta = gradient of function h(a), where a is output at terminal layer of NN
    #                         with respect to parameter vector self.theta
    # grad_h_theta.shape = (number of data points in training set, len(self.theta)
    # grad_h_theta[j,i] = for data point j in training set, grad_h w.r.t. self.theta[i]



        import fun_Activations  # import activation functions

        self.unpack_NN_parameters()     #make sure we have latest weights and biases

        #Check shape:
        if grad_h_a.ndim == 2:

            if grad_h_a.shape[1] != self.n_nodes_output:
                raise ValueError("PVS error in 'back_propagation': grad_h_a.shape[1] != self.n_nodes_output.")

            # INSERT DIMENSION to make grad_h_a vector of row vectors
            grad_h_a = grad_h_a[:, np.newaxis, :]


        #--------------------- BACKPROPAGATION LOOP -------------------------


        for layer_id in np.arange(self.n_layers_total - 1, 0, - 1):
            #   loop starts at output layer: layer_id = self.n_layers_total - 1
            #   loop last execution at first hidden layer: layer_id = 1

            if layer_id != self.n_layers_total - 1:  # if we are NOT at OUTPUT layer
                error_l_plus_1 = error_l.copy() # error matrix of (current layer plus one)
                x_l_plus_1 = x_l.copy() #weights into (current layer plus one)

            #Assign values for current layer
            z_l = z_layers_all[layer_id]    #weighted inputs into layer_id
            activation_l = self.layers[layer_id].activation  #activation function layer_id
            x_l = self.layers[layer_id].x_l     #weights into layer l
            b_l = self.layers[layer_id].b_l #biases for layer_id, ONLY used to test there are in fact biases

            #Get derivative of activation function at layer evaluated at z_l
            _, grad_sig_l = fun_Activations.fun_Activations(activation=activation_l,
                                               u=z_l,
                                               output_Gradient=True,
                                               axis=1)  # axis = 1 means each row of input corresponds to an activation input

            if grad_sig_l.ndim == 2: #E.g. for logistic_sigmoid
                #insert dimension to make grad_sig_l a vector of row vectors
                grad_sig_l = grad_sig_l[:, np.newaxis, :]

            # ----------------- Calculate ERROR MATRIX -------------------------------
            if layer_id == self.n_layers_total - 1:  #OUTPUT layer

                if activation_l in ["linear_identity"]: #if activation at output layer is linear identity
                    error_l = np.multiply(grad_h_a, grad_sig_l)

                else: #OTHER output activations, like softmax etc.
                    error_l = np.matmul(grad_h_a, grad_sig_l)
            else:
                error_l = np.multiply(grad_sig_l, np.matmul(error_l_plus_1, np.transpose(x_l_plus_1)) )


            # ----------------- Backprop calcs at every level of NN -----------------
            a_l_min_1 = a_layers_all[layer_id - 1]  #output from (current layer minus one)

            if a_l_min_1.ndim == 2:
                # insert dimension to make a_l_min_1 a vector of row vectors
                a_l_min_1 = a_l_min_1[:, np.newaxis, :]

            #Swap last 2 dimensions of a_l_min_1
            a_l_min_1_tp = np.transpose(a_l_min_1, axes=(0,2,1))

            #J_x_l = stack of matrices, derivatives of h wrt weights x into layer l
            J_x_l = np.matmul(a_l_min_1_tp, error_l)

            #Transpose the matrix in each "row"
            J_x_l_tp = np.transpose(J_x_l, axes=(0, 2, 1))

            #Each matrix in J_x_l is flattened as in my notes into a row in grad_h_x_l
            #   grad_h_x_l[j,:] = derivatives of h wrt weights x into layer l
            grad_h_x_l = J_x_l_tp.reshape(J_x_l_tp.shape[0], -1)
                # np.reshape can take -1 as an argument, meaning "total  size
                # divided by product of all other listed dimensions"

            #Derivatives with respect to biases for this layer
            if b_l is not None:
                grad_h_b_l = np.squeeze(error_l, axis= 1)   #Remove axis=1 to make 2dim
            else:
                grad_h_b_l = None


            #Append to stacks of derivatives, from output layer down to first hidden layer
            if layer_id == self.n_layers_total - 1:  #OUTPUT layer
                grad_h_x_ALL = grad_h_x_l.copy()
                grad_h_b_ALL = None  # initialize, used below

            else: #Append horizontally
                grad_h_x_ALL = np.hstack( (grad_h_x_ALL, grad_h_x_l) )



            if grad_h_b_l is not None:  #There are biases for this layer

                if grad_h_b_ALL is None:    #There were no biases for higher layers
                    grad_h_b_ALL =  grad_h_b_l.copy() #create if not yet created

                else: #append if grad_h_b_ALL already created
                    grad_h_b_ALL = np.hstack((grad_h_b_ALL, grad_h_b_l))

        # End: --------------------- BACKPROPAGATION LOOP -------------------------


        #Set final output:
        if grad_h_b_ALL is None:  # No biases in any layers
            grad_h_theta = grad_h_x_ALL.copy()
        else:
            #Stack gradients
            grad_h_theta = np.hstack((grad_h_x_ALL, grad_h_b_ALL))
        # grad_h_theta.shape = (number of data points in training set, len(self.theta)
        # grad_h_theta[j,i] = for data point j in training set, grad_h w.r.t. self.theta[i]

        return grad_h_theta


    def forward_propagation(self, phi, theta = None):

    #OBJECTIVE: Do forward propagation of NN

    #OUTPUT: a_output, a_layers_all, z_layers_all

    # a_output = outputs of nodes in the outer layer of the NN
    # a_output.shape = (rows of training data, self.n_nodes_output)
    #  - if a_output is a VECTOR (a_output.ndim == 1), then len(a_output) == self.n_nodes_output
    # -  if a_output is a MATRIX (a_output.ndim = 2), then every ROW of a_output
    #    corresponds to the output of the NN (each output node in a column of a_output)
    #    for each row in the input/feature matrix in the training data

    #  Outputs for BACKPROPAGATION:
    # a_layers_all = DICTIONARY in format {layer_id : a}, where a is the output of the layer with layer_id
    # z_layers_all = DICTIONARY in format {layer_id : z_l}, where z_l is the weighted input into each node in layer layer_id

    #INPUTS:
    #   phi = required input/feature vector (if adjusted/standardized, must be done PRIOR to using it here)
    #   - if phi is a VECTOR (phi.ndim == 1), then we must have len(phi) == self.layer[0].n_nodes
    #   - if phi is a MATRIX (phi.ndim == 2), then we must have phi.shape[1] == self.layer[0].n_nodes
    #      i.e. every ROW of phi must correspond to ONE input/feature vector in the training data

    #  if phi is a matrix, row (j) corresponds to the feature vector
    #           for the jth data point/sample path in training data

    #   theta = optional single parameter vector with weights and biases to use for forward propagation
    #           if theta is  None: use values in attribute self.theta

        import fun_Activations  #import activation functions

        self.unpack_NN_parameters()  # make sure we have latest weights and biases

        #Check shape of feature vector(s)

        phi = np.array(phi)  # just make sure it is in np.array form

        if phi.ndim == 1:
            if len(phi) != self.layers[0].n_nodes:
                raise ValueError("PVS error in 'forward_propagation': size of input vector does "
                                 "not match number of nodes in input layer of NN.")
        elif phi.ndim == 2:


            if phi.shape[1] != self.layers[0].n_nodes:
                raise ValueError("PVS error in 'forward_propagation': nr feature vector matrix COLUMNS does "
                                 "not match number of nodes in input layer of NN.")
        else:
            raise ValueError("PVS error in 'forward_propagation': can only handle 1 or 2 dimensions of feature vector inputs.")

        #Unpack provided parameter vector, otherwise use self.theta (and unpack just to make sure)
        if theta is None: #if no theta input has been provided
            if self.theta is None: #if no theta is currently in self
                raise ValueError("PVS error in 'forward_prop': no parameter vector associated with NN.")
            else:  #if self.theta contains values:
                self.unpack_NN_parameters() #unpack just to make sure self.theta has been unpacked

        else: #theta is not None, so copy into NN.theta
            if len(theta) == self.theta_length: #check length of provided input
                theta = np.array(theta) #just make sure it is in np.array form
                self.theta = theta.copy() #copy into NN attribute
                self.unpack_NN_parameters()   #unpack into weights matrices and bias vectors

            else:   #length of provided parameter vector does not match number of NN parameters
                raise ValueError("PVS error in 'forward_prop': provided parameter vector does " \
                                 "not match length of NN parameter vector")

        #--------------------------------------------------------------------
        #Do forward propagation
        a = phi.copy() #initialize running output using input layer
        a_layers_all = {0 : a}  #initialize a_layers_all = {layer_id : a}
        z_layers_all = {0 : np.NaN} #initialize  z_layers_all = {layer_id, z_l}: no input for layer_id = 0


        #Start looping at layer_id = 1, since layer_id = 0 is the input layer
        for layer_id in np.arange(1, self.n_layers_total, 1):

            a_prev = a.copy()

            x_l = self.layers[layer_id].x_l
            b_l = self.layers[layer_id].b_l
            activation_l = self.layers[layer_id].activation

            #Do checks and exit if necessary
            if x_l is None:
                raise ValueError("PVS error in 'forward_prop': no weights to use.")

            if phi.ndim == 1:

                if b_l is None:
                    z_l = np.dot(np.transpose(x_l), a_prev)
                else:
                    z_l = np.dot( np.transpose(x_l), a_prev) + b_l

                a, _ = fun_Activations.fun_Activations(activation= activation_l,
                                                       u = z_l,
                                                       output_Gradient=False,
                                                       axis = 1) #axis = 1 means each row of input corresponds to an activation input

            elif phi.ndim == 2:
                #  if phi is a matrix, ROW (j) corresponds to the feature vector
                # for the jth data point in training data

                #Make sure b_l is a row vector rather than one dim
                # np broadcasting rules us to add this to z_l

                if b_l is None:
                    z_l = np.matmul(a_prev, x_l)
                else:
                    # Make sure b_l is a row vector rather than one dim
                    # np broadcasting rules us to add this to z_l
                    b_l_rowvect = b_l[np.newaxis, :]
                    z_l = np.matmul(a_prev, x_l) + b_l_rowvect

                a, _ = fun_Activations.fun_Activations(activation=activation_l,
                                                       u=z_l,
                                                       output_Gradient=False,
                                                       axis=1)  # axis = 1 means each row of input corresponds to an activation input

            #Regardless of dimensions, update the dictionaries for backpropagation
            a_layers_all.update({ layer_id : a})  # initialize a_layers_all = {layer_id : a}
            z_layers_all.update({layer_id : z_l})
        #end: for loop up to self.n_layers_total

        #Assign output
        a_output = a.copy()
        # a_layers_all =  {layer_id : a} already updated
        # z_layers_all = {layer_id : z_l} already updated


        return a_output, a_layers_all, z_layers_all
#----------------------------------------------------------------------------------------

class Neural_Network_Layer(object):

    def __init__(self, layer_id, n_nodes, activation, description):
        self.layer_id = layer_id      #index of this layer within NN
        self.n_nodes = n_nodes       #nr of nodes in this layer
        self.activation = activation #string giving activation functionfor layer
        self.description = description  #layer description for ease of reference
        self.add_bias = False  #False means no bias vector, True means with bias vector
        self.b_l = None    # bias vector for this layer. default = None
        self.b_l_length = None # integer containing length of b_l, used for reshaping; default = None
        self.x_l = None    # weights matrix for this layer. default = None
        self.x_l_shape = None  # tuple containing x_l.shape, used for reshaping; default = None


#----------------------------------------------------------------------------------------


