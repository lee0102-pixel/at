import yaml
from collections import OrderedDict
import numpy as np
import random
import torch
import os

def setup_seed(seed):
    # https://blog.csdn.net/hyk_1996/article/details/84307108  参考这里
    # cuDNN
    torch.backends.cudnn.benchmark = False  # Benchmark模式提升计算速度， 程序在开始时花费一点额外时间, 由于机制复杂，默认为False
    torch.backends.cudnn.deterministic = True  # cudnn中的随机
    # Python & NumPy
    np.random.seed(seed)  # 以防其他地方进行了numpy随机处理(如RandomCrop、RandomHorizontalFlip等)
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    # PyTorch
    torch.manual_seed(seed)  # CPU随机种子
    torch.cuda.manual_seed(seed)  # GPU随机种子
    torch.cuda.manual_seed_all(seed)  # 多GPU，所有GPU
    
def ordered_yaml():
    try:
        from yaml import CDumper as Dumper
        from yaml import CLoader as Loader
    except ImportError:
        from yaml import Dumper, Loader
    _mapping_tag = yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG
    
    def dict_representer(dumper, data):
        return dumper.represent_dict(data.iteritems())
    def dict_constructor(loader, node):
        return OrderedDict(loader.construct_pairs(node))
    Dumper.add_representer(OrderedDict, dict_representer)
    Loader.add_constructor(_mapping_tag, dict_constructor)
    return Dumper, Loader

def load_yaml(path):
    with open(path, 'r') as f:
        Dumper, Loader = ordered_yaml()
        data = yaml.load(f, Loader=Loader)
    return data

def parse_opt(args):
    if args.opt:
        opt_path = args.opt
        args.opt = load_yaml(opt_path)
        args.opt_path = opt_path
        
        for k,v in args.opt['isp'].items():
            setattr(args, k, v)
            
        for k,v in args.opt['global_settings'].items():
            setattr(args, k, v)
            
        for k, v in args.opt['train'].items():
            setattr(args, k, v)
        
        for k, v in args.opt['network'].items():
            setattr(args, k, v)
            
        for k, v in args.opt['fabric'].items():
            setattr(args, k, v)
            
        for k, v in args.opt['loggers'].items():
            setattr(args, k, v)
    return args


def get_psnr(p0, p1, peak=255.):
    return 10*np.log10(peak**2/np.mean((1.*p0-1.*p1)**2))


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--opt', type=str, default='./options/unet_step1.yaml')
    args = parser.parse_args()
    parse_opt(args)
    
    print(args)