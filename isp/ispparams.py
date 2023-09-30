import importlib
import numpy as np

class ISPParams:
    
    def __init__(self, args):
        super(ISPParams, self).__init__()
        self.args = args
        
        isp_constraint_module       = importlib.import_module(args.isp_constraint)
        isp_range_module            = importlib.import_module(args.isp_range)
        
        self.params_constraint_dict = isp_constraint_module.get_params_constraint()
        self.params_range_dict      = isp_range_module.get_params_range(args.gain)
        self.params_name_list       = list(self.params_constraint_dict.keys())
        self.params_default_list    = [self.params_range_dict[k][2] for k in self.params_name_list]
        self.params_type_list       = [type(item) for item in self.params_default_list]
        
    def get_params_num(self):
        return len(self.params_name_list)
    
    def write_params_txt(self, save_path):
        with open(save_path + '/params.txt', 'w') as f:
            for params_name in self.params_name_list:
                f.write('%s:\t %d-%d\n' % (params_name, 
                                           self.params_range_dict[params_name][0], 
                                           self.params_range_dict[params_name][1]))
    
    def get_normed_list(self, params_list):
        assert len(params_list) == len(self.params_name_list), 'params_list must have the same length as params_name_list'
        normed_list = []
        for i, item in enumerate(params_list):
            lower_bound = self.params_range_dict[self.params_name_list[i]][0]
            upper_bound = self.params_range_dict[self.params_name_list[i]][1]
            normed_value = self._normalize(item, lower_bound, upper_bound, 0, 1)
            
            normed_list.append(normed_value)

        return normed_list
    
    def get_denormed_list(self, normed_list):
        assert len(normed_list) == len(self.params_name_list), 'normed_list must have the same length as params_name_list'
        denormed_list = []
        for i, item in enumerate(normed_list):
            lower_bound = self.params_range_dict[self.params_name_list[i]][0]
            upper_bound = self.params_range_dict[self.params_name_list[i]][1]
            denormed_value = self._normalize(item, 0, 1, lower_bound, upper_bound)
            if self.params_type_list[i] == type(1):
                denormed_value = int(np.round(denormed_value))
            denormed_list.append(denormed_value)

        return denormed_list
    
                
    def _normalize(self, x, a, b, c, d):
        """
        x from (a, b) to (c, d)
        """
        assert a<= b, 'a must be less than b'
        assert c<= d, 'c must be less than d'
        
        if a == b:
            return 0
        
        return (float(x)-a)/(float(b)-a)*(d-c) + c
    
    def write_results(self, normed_list, save_path):
        assert len(normed_list) == len(self.params_name_list), 'normed_list must have the same length as params_name_list'
        denormed_list = self.get_denormed_list(normed_list)
        result = str(denormed_list)
        with open(save_path, 'w') as f:
            f.write(result)            
    
    
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
    
    isp_params = ISPParams(args)
    # isp_params.write_params_txt('./')
    print(isp_params.params_name_list)
    print(isp_params.params_default_list)
    print(isp_params.params_type_list)
    print(isp_params.get_normed_list(isp_params.params_default_list))
    # isp_params.write_results(isp_params.get_normed_list(isp_params.params_default_list), './results.txt')