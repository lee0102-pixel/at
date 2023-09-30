import cv2
import numpy as np
from torch.utils.data import Dataset, DataLoader
import random
import os
import torch


class RGB2RGB(Dataset):
    def __init__(self, args, isp):
        super().__init__()
        self.args        = args
        self.isp         = isp
        self.crop_size   = args.crop_size
        self.data_txt    = args.data_txt
        self.step_flag   = args.step_flag
        self.is_train    = args.is_train
        self.input_list  = []
        self.gt_list     = []
        self.params_list = []
        
        assert os.path.exists(self.data_txt), 'data_txt does not exist'
        
        with open(self.data_txt, 'r') as f:
            for line in f.readlines():
                line = line.strip('\n')
                input_path, gt_path, params = line.split(',')
                self.input_list.append(input_path)
                self.gt_list.append(gt_path)
                self.params_list.append(params)
                
    def __len__(self):
        return len(self.input_list)
    
    def __getitem__(self, idx):
        input_path = self.input_list[idx]
        gt_path    = self.gt_list[idx]
        params     = self.params_list[idx]
        
        input_img = cv2.imread(input_path)
        gt_img    = cv2.imread(gt_path)
        input_img = cv2.cvtColor(input_img, cv2.COLOR_BGR2RGB)
        gt_img    = cv2.cvtColor(gt_img, cv2.COLOR_BGR2RGB)
        
        h,w,_ = input_img.shape
        
        if self.is_train:
            r = random.randint(0, h-self.crop_size)
            w = random.randint(0, w-self.crop_size)
        else:
            r = (h-self.crop_size)//2
            w = (w-self.crop_size)//2

        input_img = input_img[r:r+self.crop_size, w:w+self.crop_size, :]
        gt_img    = gt_img[r:r+self.crop_size, w:w+self.crop_size, :]
        input_img = input_img.astype(np.float32)/255.0
        gt_img    = gt_img.astype(np.float32)/255.0
        
        with open(params, 'r') as f:
            params = f.readlines()[0].strip('\n')[1:-1].split(',')
            # print(params)
            params = [float(item) for item in params]
        params = self.isp.get_normed_list(params)
        
        input_img = torch.from_numpy(input_img).permute(2,0,1)
        gt_img    = torch.from_numpy(gt_img).permute(2,0,1)
        params    = torch.tensor(params)
        base_name = os.path.basename(input_path)
        
        return input_img, gt_img, params, base_name
    
    
def train_dataloader(args, isp):
    args.is_train = True
    args.data_txt = args.train_txt
    dataset = RGB2RGB(args, isp)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
    return dataloader

def val_dataloader(args, isp):
    args.is_train = False
    args.data_txt = args.val_txt
    dataset = RGB2RGB(args, isp)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
    return dataloader

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
    
    train_loader = train_dataloader(args, isp_params)
    print(len(train_loader))
    for data in train_loader:
        input_img, gt_img, params, base_name = data
        print(input_img.shape)
        print(gt_img.shape)
        print(params.shape)
        print(base_name)
        # break