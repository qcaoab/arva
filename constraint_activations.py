import torch


def asset_constraint_activation(nn_out, params):
    
    softmax = torch.nn.Softmax(dim=1)
    sigmoid = torch.nn.Sigmoid()
    
    size_max = torch.tensor(params["factor_constraints_dict"]["Size_Lo30"], device= params["device"])
    value_max = torch.tensor(params["factor_constraints_dict"]["Value_Hi30"], device= params["device"]) 

    # each factor asset prop is calculated with independent sigmoid and then scaled accoring to max as specified in params
    # the total factor proportion is then used to scale the basic assets appropriately so proportions sum to 1. 
    
    if params["asset_basket_id"] ==  "Paper_FactorInv_Factor2":
    
        size_prop = sigmoid(nn_out[:,3]) * size_max
        value_prop = sigmoid(nn_out[:,4]) * value_max
        
        basic_prop = softmax(nn_out[:,0:3]) * (1 - (size_prop+value_prop))[:,None]
    else:
        raise ValueError("asset basket constraint not coded")
    
    # check if props sum to 1:
    with torch.no_grad():
        
        prop_sums = basic_prop.sum(dim=1) + size_prop + value_prop
        if any(torch.round(prop_sums, decimals = 6) != 1):
            raise ValueError("asset props don't sum to one")
    
    a_t_n_output = torch.cat((basic_prop, size_prop[:,None], value_prop[:,None]), 1)
        
    return a_t_n_output


def w_custom_activation(nn_out, g_prev, params):
    
    sigmoid = torch.nn.Sigmoid()
    formulation = params["w_constraint_activation"]
    
    q_min = torch.tensor(params["q_min"], device= params["device"])
    q_max = torch.tensor(params["q_max"], device= params["device"])
    
    if formulation == "yy_fix_jan29":
        custom_sigmoid = sigmoid(nn_out)
       
        max_qmin_w = torch.maximum(torch.ones(g_prev.size(), device=params["device"])*q_min, g_prev)
        min_outer_qmax = torch.minimum(max_qmin_w, torch.ones(g_prev.size(), device=params["device"])*q_max)
        
        q_n = q_min + (min_outer_qmax - q_min)*(custom_sigmoid)
        
        return q_n
     
    if formulation  == "yy_fix_feb3":
        
        x = torch.maximum(g_prev - q_min, torch.zeros(g_prev.size(), device=params["device"])) \
            * torch.exp(nn_out)
        
        max_qmin_w = torch.maximum(torch.ones(g_prev.size(), device=params["device"])*q_min, g_prev)
        min_outer_qmax = torch.minimum(max_qmin_w, torch.ones(g_prev.size(), device=params["device"])*q_max)
        
        q_n = q_min + 2*(min_outer_qmax - q_min) * (sigmoid(x) - 0.5)
        
        return q_n

        
# # yuying fix v1
        # custom_sigmoid = torch.sigmoid(torch.maximum(torch.zeros(g_prev.size(), device = params["device"]), 
        #                                              g_prev)*torch.exp(nn_out))
       
        # q_n = q_min + 2*(q_max - q_min)*(custom_sigmoid - 0.5)
             
        
        #   #yuying fix v1, with w-q_min
        # custom_sigmoid = torch.sigmoid(torch.maximum(torch.zeros(g_prev.size(), device = params["device"]), 
        #                                              g_prev-q_min)*torch.exp(nn_out))
       
        # q_n = q_min + 2*(q_max - q_min)*(custom_sigmoid - 0.5)
        
        # #   MC jan29 fix, with w-q_min and altered simplified range term.  
        # custom_sigmoid = torch.sigmoid(torch.maximum(torch.zeros(g_prev.size(), device = params["device"]), 
        #                                              g_prev-q_min)*torch.exp(nn_out))
       
        # min_qmax_w = torch.minimum(torch.ones(g_prev.size(), device=params["device"])*q_max, g_prev)
        
        # q_n = q_min + 2*(min_qmax_w - q_min)*(custom_sigmoid - 0.5)
        
        #   YY jan29 fix, described in note.  
        custom_sigmoid = torch.sigmoid(nn_out)
       
        max_qmin_w = torch.maximum(torch.ones(g_prev.size(), device=params["device"])*q_min, g_prev)
        min_outer_qmax = torch.minimum(max_qmin_w, torch.ones(g_prev.size(), device=params["device"])*q_max)
        
        q_n = q_min + (min_outer_qmax - q_min)*(custom_sigmoid)
        
        # #yuying feb3 fix
        
        # x = torch.maximum(g_prev - q_min, torch.zeros(g_prev.size(), device=params["device"])) \
        #     * torch.exp(nn_out)
        
        # max_qmin_w = torch.maximum(torch.ones(g_prev.size(), device=params["device"])*q_min, g_prev)
        # min_outer_qmax = torch.minimum(max_qmin_w, torch.ones(g_prev.size(), device=params["device"])*q_max)
        
        
        # q_n = q_min + 2*(min_outer_qmax - q_min) * (torch.sigmoid(x) - 0.5)

               
        
        # yuying fix v2, with border addendum
        # custom_sigmoid = torch.sigmoid(torch.maximum(torch.zeros(g_prev.size(), device = params["device"]), 
        #                                               g_prev*torch.exp(nn_out)))

        # max_qmin_w = torch.maximum(torch.ones(g_prev.size(), device=params["device"])*q_min, g_prev)
        # min_outer_qmax = torch.minimum(max_qmin_w, torch.ones(g_prev.size(), device=params["device"])*q_max)
        
        # q_n = q_min + 2*(min_outer_qmax - q_min)*(custom_sigmoid - 0.5)
