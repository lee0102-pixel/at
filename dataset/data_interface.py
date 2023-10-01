import inspect
import importlib
import pickle as pkl
import pytorch_lightning as pl
from pytorch_lightning.utilities.types import EVAL_DATALOADERS
from torch.utils.data import DataLoader

class DInterface(pl.LightningDataModule):

    def __init__(self, args, isp, **kwargs):
        super().__init__()
        self.args         = args
        self.isp          = isp
        self.num_workers  = args.num_workers
        self.dataset_name = args.dataset_name
        self.kwargs       = kwargs
        self.batch_size   = args.batch_size
        self.load_data_module()

    def load_data_module(self):
        name = self.dataset_name
        file_name = name.lower()
        try:
            self.data_module = getattr(importlib.import_module(
                '.'+file_name, package='dataset'), name)
        except:
            raise ValueError(
                f'Invalid Dataset File Name or Invalid Class Name data.{file_name}.{name}'
            )
    
    def setup(self, stage: str) -> None:
        if stage == 'fit' or stage is None:
            self.args.is_train = True
            self.args.data_txt = self.args.train_txt
            self.train_dataset = self.data_module(self.args, self.isp, **self.kwargs)
            self.args.is_train = False
            self.args.data_txt = self.args.val_txt
            self.val_dataset   = self.data_module(self.args, self.isp, **self.kwargs)
        if stage == 'test' or stage is None:
            self.args.is_train = False
            self.args.data_txt = self.args.val_txt
            self.test_dataset  = self.data_module(self.args, self.isp, **self.kwargs)

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers)
    
    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers)
    
    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers)
    

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

    dinterface = DInterface(args, isp_params)

    dinterface.setup('fit')
    train_loader = dinterface.train_dataloader()
    print(len(train_loader))
    for data in train_loader:
        input_img, gt_img, params, base_name = data
        print(input_img.shape)
        print(gt_img.shape)
        print(params.shape)
        print(base_name)
        break
    dinterface.setup('test')
    test_loader = dinterface.test_dataloader()
    print(len(test_loader))
    for data in test_loader:
        input_img, gt_img, params, base_name = data
        print(input_img.shape)
        print(gt_img.shape)
        print(params.shape)
        print(base_name)
        break