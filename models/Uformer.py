import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import os
import sys
#add dir
dir_name = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(dir_name,'..'))
from models.ParamsLayer import ParamsLayer


class Uformer(ParamsLayer):
    def __init__(self, args, isp, **kwargs) -> None:
        super().__init__(args, isp, **kwargs)