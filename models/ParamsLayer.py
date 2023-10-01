import torch.nn as nn
import torch.nn.functional as F
import torch
import numpy as np


class ParamsLayer(nn.Module):
    def __init__(self, args, isp, **kwargs) -> None:
        super().__init__(**kwargs)
        self.args = args
        self.isp  = isp
        self.img_size = args.crop_size
        self.step_flag = args.step_flag
        self.params_layer = None
        
    def set_params_layer(self, params_list=None):
        if params_list is None:
            params_list = self.isp.params_default_list
        
        params_len = len(params_list)    
        assert params_len == self.isp.get_params_num(), 'params_list must have the same length as params_name_list'
        
        normed_list = self.isp.get_normed_list(params_list)
        params_layer = []
        for i in range(params_len):
            params_layer.append(torch.tensor(normed_list[i]).expand(1,1,self.img_size, self.img_size).float())
            
        params_layer = torch.cat(params_layer, dim=1)
        self.params_layer = torch.clamp(params_layer, 0, 1)
        if self.step_flag == 3:
            self.params_layer.requires_grad = True
        else:
            self.params_layer.requires_grad = False
            
    def forward(self, input):
        pass
    
    def get_params_list(self):
        params_list = []
        for i in range(self.isp.get_params_num()):
            params_list.append(self.params_layer[0, i, :, :].cpu().detach().numpy().mean())
        return params_list
    
    def update_params_layer(self):
        self.params_layer = self.params_layer.detach()
        normed_list       = self.get_params_list()
        denormed_list     = self.isp.get_denormed_list(normed_list)
        
        for i in range(self.isp.get_params_num()):
            params_name     = self.isp.params_name_list[i]
            constraint_info = self.isp.params_constraint_dict[params_name]
            lower_bound     = self.isp.params_range_dict[params_name][0]
            upper_bound     = self.isp.params_range_dict[params_name][1]
            default_value   = self.isp.params_range_dict[params_name][2]
            params_type     = self.isp.params_type_list[i]
            params_value    = denormed_list[i]
            if params_type == type(1):
                step = 1
            else:
                step = (upper_bound - lower_bound)*0.01
            if constraint_info == 'Indp':
                pass
            elif constraint_info == 'Inc':
                pre_value = denormed_list[i-1]
                if params_value < pre_value:
                    params_value = pre_value
            elif constraint_info == 'Dec':
                pre_value = denormed_list[i-1]
                if params_value > pre_value:
                    params_value = pre_value
            elif constraint_info == 'StrInc':
                pre_value = denormed_list[i-1]
                if params_value <= pre_value:
                    params_value = pre_value + step
            elif constraint_info == 'StrDec':
                pre_value = denormed_list[i-1]
                if params_value >= pre_value:
                    params_value = pre_value - step
            elif constraint_info == 'Default':
                params_value = default_value
            else:
                pass
            new_value = self.isp._normalize(params_value, lower_bound, upper_bound, 0, 1)
            self.params_layer[:, i:i+1, :, :] = torch.tensor(new_value).expand(1,1,self.img_size, self.img_size)
            
        self.params_layer = torch.clamp(self.params_layer, 0, 1)
        self.params_layer.requires_grad = True
                
                
if __name__ == '__main__':
    import os
    import sys
    #add dir
    dir_name = os.path.dirname(os.path.abspath(__file__))
    sys.path.append(os.path.join(dir_name,'..'))
    
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--opt', type=str, default='./options/unet_step1.yaml')
    args = parser.parse_args()
    from tools.utils import parse_opt
    parse_opt(args)
    from isp.ispparams import ISPParams
    isp_params = ISPParams(args)
    
    params_layer = ParamsLayer(args, isp_params)
    params_layer.set_params_layer()
    # params_layer.init_params_layer([2,3,4,5,6,7,8,9])
    print(params_layer.params_layer.shape)
    print(params_layer.get_params_list())