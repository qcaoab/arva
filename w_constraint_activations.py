import torch


def custom_activation(nn_out, g_prev, params):
    
    formulation = params["w_constraint_activation"]
    
    q_min = torch.tensor(params["q_min"], device= params["device"])
    q_max = torch.tensor(params["q_max"], device= params["device"])
    
    if formulation == "yy_fix_jan29":
        custom_sigmoid = torch.sigmoid(nn_out)
       
        max_qmin_w = torch.maximum(torch.ones(g_prev.size(), device=params["device"])*q_min, g_prev)
        min_outer_qmax = torch.minimum(max_qmin_w, torch.ones(g_prev.size(), device=params["device"])*q_max)
        
        q_n = q_min + (min_outer_qmax - q_min)*(custom_sigmoid)
        
        return q_n
     
    if formulation  == "yy_fix_feb3":
        
        x = torch.maximum(g_prev - q_min, torch.zeros(g_prev.size(), device=params["device"])) \
            * torch.exp(nn_out)
        
        max_qmin_w = torch.maximum(torch.ones(g_prev.size(), device=params["device"])*q_min, g_prev)
        min_outer_qmax = torch.minimum(max_qmin_w, torch.ones(g_prev.size(), device=params["device"])*q_max)
        
        q_n = q_min + 2*(min_outer_qmax - q_min) * (torch.sigmoid(x) - 0.5)
        
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
