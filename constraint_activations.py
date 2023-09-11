import torch


def asset_constraint_activation(nn_out, params):
    
    softmax = torch.nn.Softmax(dim=1)
    sigmoid = torch.nn.Sigmoid()
    
    

    # each factor asset prop is calculated with independent sigmoid and then scaled accoring to max as specified in params
    # the total factor proportion is then used to scale the basic assets appropriately so proportions sum to 1. 
    
    if params["asset_basket_id"] ==  "Paper_FactorInv_Factor2" and not params["dynamic_total_factorprop"]:
        
        size_max = torch.tensor(params["factor_constraints_dict"]["Size_Lo30"], device= params["device"])
        value_max = torch.tensor(params["factor_constraints_dict"]["Value_Hi30"], device= params["device"]) 
    
        size_prop = sigmoid(nn_out[:,3]) * size_max
        value_prop = sigmoid(nn_out[:,4]) * value_max
        basic_prop = softmax(nn_out[:,0:3]) * (1 - (size_prop+value_prop))[:,None]
        
        a_t_n_output = torch.cat((basic_prop, size_prop[:,None], value_prop[:,None]), 1)
    
    elif params["asset_basket_id"] == "Paper_FactorInv_Factor2" and params["dynamic_total_factorprop"]:
        
        total_factor_max = torch.tensor(params["factor_constraints_dict"]["total_factor_max"], device= params["device"])   
        
        factor_total = sigmoid(nn_out[:,5]) * total_factor_max
        factor_total = factor_total[:,None]
        
        factor_prop = softmax(nn_out[:,0:3]) * factor_total
        basic_prop = softmax(nn_out[:,3:5]) * (1 - factor_total)
        
        a_t_n_output = torch.cat((basic_prop, factor_prop), 1)
        
        #["T30", "B10", "VWD", "Size_Lo30", "Value_Hi30"]
        
    elif params["asset_basket_id"] ==  "4_factor_1927" and not params["dynamic_total_factorprop"]:
        
        size_max = torch.tensor(params["factor_constraints_dict"]["Size_Lo30"], device= params["device"])
        value_max = torch.tensor(params["factor_constraints_dict"]["Value_Hi30"], device= params["device"])
        div_max = torch.tensor(params["factor_constraints_dict"]["Div_Hi30"], device= params["device"])
        mom_max = torch.tensor(params["factor_constraints_dict"]["Mom_Hi30"], device= params["device"])
    
        size_prop = sigmoid(nn_out[:,3]) * size_max
        value_prop = sigmoid(nn_out[:,4]) * value_max
        div_prop = sigmoid(nn_out[:,5]) * div_max
        mom_prop = sigmoid(nn_out[:,6]) * mom_max
        
        basic_prop = softmax(nn_out[:,0:3]) * (1 - (size_prop + value_prop + div_prop + mom_prop))[:,None]
        
        a_t_n_output = torch.cat((basic_prop, size_prop[:,None], value_prop[:,None], div_prop[:,None], mom_prop[:,None]), 1)
    
    elif params["asset_basket_id"] ==  "4_factor_1927" and params["dynamic_total_factorprop"]:
        
        total_factor_max = torch.tensor(params["factor_constraints_dict"]["total_factor_max"], device= params["device"])   
               
        factor_total = sigmoid(nn_out[:,7]) * total_factor_max
        factor_total = factor_total[:,None]
        
        factor_prop = softmax(nn_out[:,3:7]) * factor_total
        basic_prop = softmax(nn_out[:,0:3]) * (1 - factor_total)
        
        a_t_n_output = torch.cat((basic_prop, factor_prop), 1)
    
    elif params["asset_basket_id"] ==  "Paper_FactorInv_Factor4" and not params["dynamic_total_factorprop"]:
        
        size_max = torch.tensor(params["factor_constraints_dict"]["Size_Lo30"], device= params["device"])
        value_max = torch.tensor(params["factor_constraints_dict"]["Value_Hi30"], device= params["device"])
        vol_max = torch.tensor(params["factor_constraints_dict"]["Vol_Lo20"], device= params["device"])
        mom_max = torch.tensor(params["factor_constraints_dict"]["Mom_Hi30"], device= params["device"])
    
        size_prop = sigmoid(nn_out[:,3]) * size_max
        value_prop = sigmoid(nn_out[:,4]) * value_max
        vol_prop = sigmoid(nn_out[:,5]) * vol_max
        mom_prop = sigmoid(nn_out[:,6]) * mom_max
        
        basic_prop = softmax(nn_out[:,0:3]) * (1 - (size_prop + value_prop + vol_prop + mom_prop))[:,None]
        
        a_t_n_output = torch.cat((basic_prop, size_prop[:,None], value_prop[:,None], vol_prop[:,None], mom_prop[:,None]), 1)
    
    elif params["asset_basket_id"] ==  "Paper_FactorInv_Factor4" and params["dynamic_total_factorprop"]:
        
        total_factor_max = torch.tensor(params["factor_constraints_dict"]["total_factor_max"], device= params["device"])   
               
        factor_total = sigmoid(nn_out[:,7]) * total_factor_max
        factor_total = factor_total[:,None]
        
        factor_prop = softmax(nn_out[:,3:7]) * factor_total
        basic_prop = softmax(nn_out[:,0:3]) * (1 - factor_total)
        
        a_t_n_output = torch.cat((basic_prop, factor_prop), 1)
    
    elif params["asset_basket_id"] ==  "3factor_mc" and not params["dynamic_total_factorprop"]:
        
        size_max = torch.tensor(params["factor_constraints_dict"]["Size_Lo30"], device= params["device"])
        value_max = torch.tensor(params["factor_constraints_dict"]["Value_Hi30"], device= params["device"])
        mom_max = torch.tensor(params["factor_constraints_dict"]["Mom_Hi30"], device= params["device"])
    
        size_prop = sigmoid(nn_out[:,3]) * size_max
        value_prop = sigmoid(nn_out[:,4]) * value_max
        mom_prop = sigmoid(nn_out[:,5]) * mom_max
        
        basic_prop = softmax(nn_out[:,0:3]) * (1 - (size_prop + value_prop + mom_prop))[:,None]
        
        a_t_n_output = torch.cat((basic_prop, size_prop[:,None], value_prop[:,None], mom_prop[:,None]), 1)
    
    elif params["asset_basket_id"] ==  "3factor_mc" and params["dynamic_total_factorprop"]:
        
        total_factor_max = torch.tensor(params["factor_constraints_dict"]["total_factor_max"], device= params["device"])   
               
        factor_total = sigmoid(nn_out[:,6]) * total_factor_max
        factor_total = factor_total[:,None]
        
        factor_prop = softmax(nn_out[:,3:6]) * factor_total
        basic_prop = softmax(nn_out[:,0:3]) * (1 - factor_total)
        
        a_t_n_output = torch.cat((basic_prop, factor_prop), 1)
        
    elif params["asset_basket_id"] == "7_Factor_plusEWD" and params["dynamic_total_factorprop"]:
        
        total_factor_max = torch.tensor(params["factor_constraints_dict"]["total_factor_max"], device= params["device"])   
        
        factor_total = sigmoid(nn_out[:,13]) * total_factor_max
        factor_total = factor_total[:,None]
        
        factor_prop = softmax(nn_out[:,0:9]) * factor_total
        basic_prop = softmax(nn_out[:,9:13]) * (1 - factor_total)
        
        a_t_n_output = torch.cat((factor_prop, basic_prop), 1)
    
    elif params["asset_basket_id"] == "5_Factor_plusEWD" and params["dynamic_total_factorprop"]:
        
        total_factor_max = torch.tensor(params["factor_constraints_dict"]["total_factor_max"], device= params["device"])   
        
        factor_total = sigmoid(nn_out[:,9]) * total_factor_max
        factor_total = factor_total[:,None]
        
        factor_prop = softmax(nn_out[:,0:5]) * factor_total
        basic_prop = softmax(nn_out[:,5:9]) * (1 - factor_total)
        
        a_t_n_output = torch.cat((factor_prop, basic_prop), 1)
    
    elif params["asset_basket_id"] == "5_Factor_plusEWD" and not params["dynamic_total_factorprop"]:
        
        size_max = torch.tensor(params["factor_constraints_dict"]["Size_Lo30"], device= params["device"])
        value_max = torch.tensor(params["factor_constraints_dict"]["Value_Hi30"], device= params["device"])
        vol_max = torch.tensor(params["factor_constraints_dict"]["Vol_Lo20"], device= params["device"])
        mom_max = torch.tensor(params["factor_constraints_dict"]["Mom_Hi30"], device= params["device"])
        div_max = torch.tensor(params["factor_constraints_dict"]["Div_Hi30"], device= params["device"])
        
        size_prop = sigmoid(nn_out[:,0]) * size_max
        value_prop = sigmoid(nn_out[:,1]) * value_max
        mom_prop = sigmoid(nn_out[:,2]) * mom_max
        vol_prop = sigmoid(nn_out[:,3]) * vol_max
        div_prop = sigmoid(nn_out[:,4]) * div_max
        
        basic_prop = softmax(nn_out[:,5:9]) * (1 - (size_prop + value_prop + vol_prop + mom_prop + div_prop))[:,None]
        
        a_t_n_output = torch.cat((size_prop[:,None], value_prop[:,None], mom_prop[:,None], vol_prop[:,None], div_prop[:,None], basic_prop), 1)
        
        #["Size_Lo30", "Value_Hi30", "Mom_Hi30", "Vol_Lo20", "Div_Hi30",  "T30", "B10", "VWD", "EWD"]
        
    else:
        raise ValueError("asset basket constraint not coded")
    
    # check if props sum to 1:
    # with torch.no_grad():
        
    #     prop_sums = basic_prop.sum(dim=1) + size_prop + value_prop
    #     if any(torch.round(prop_sums, decimals = 6) != 1):
    #         raise ValueError("asset props don't sum to one")
           
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
