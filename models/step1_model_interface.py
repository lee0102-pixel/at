import os
import sys

from torch import Tensor

dir_name = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(dir_name,'..'))

from typing import Any, Optional
from pytorch_lightning.utilities.types import STEP_OUTPUT
import torch
import importlib
import torch.optim.lr_scheduler as lrs
import pytorch_lightning as pl
import losses.losses as losses
from torchmetrics.image import PeakSignalNoiseRatio

class MInterface(pl.LightningModule):
    def __init__(self, args, isp, **kwargs):
        super().__init__()
        self.args = args
        self.isp = isp
        self.kwargs = kwargs
        self.load_model()
        self.configure_loss()

    def forward(self, x, params_batch=None):
        return self.model(x, params_batch)
    
    def training_step(self, batch, batch_idx):
        x, y, params_batch, _ = batch
        y_hat = self.forward(x, params_batch)
        loss, loss_dict = self.loss_function(y_hat, y)
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True)

        return loss
    
    def validation_step(self, batch, batch_idx):
        x, y, params_batch, picname = batch
        y_hat = self.forward(x, params_batch)
        loss, loss_dict = self.loss_function(y_hat, y)

        self.log('val_loss', loss, on_step=False, on_epoch=True, prog_bar=True)
        PSNR = PeakSignalNoiseRatio(data_range=1.0)
        psnr = PSNR(y_hat.detach().cpu(), y.detach().cpu())
        self.log('psnr', psnr, on_step=False, on_epoch=True, prog_bar=True)

        return {'val_loss': loss, 'psnr': psnr}
    
    def test_step(self, batch, batch_idx):
        return self.validation_step(batch, batch_idx)
    
    def on_validation_epoch_end(self):
        self.print('')

    def configure_optimizers(self):
        if self.args.optimizer == 'adam':
            optimizer = torch.optim.Adam(self.parameters(), 
                                         lr=self.args.init_lr, 
                                         betas=(0.9, 0.999), 
                                         eps=1e-8, 
                                         weight_decay=self.args.weight_decay)
        elif self.optimizer == 'adamw':
            optimizer = torch.optim.AdamW(self.parameters(), 
                                          lr=self.args.init_lr, 
                                          betas=(0.9, 0.999), 
                                          eps=1e-8, 
                                          weight_decay=self.args.weight_decay)
        else:
            raise Exception("Optimizer error!")
    
        if self.args.scheduler == 'step':
            scheduler = lrs.StepLR(optimizer, 
                                   step_size=self.args.step_size, 
                                   gamma=self.args.gamma)
        elif self.args.scheduler == 'cosine':
            scheduler = lrs.CosineAnnealingLR(optimizer, 
                                              T_max=self.args.T_max, 
                                              eta_min=self.args.eta_min)
        else:
            raise Exception("Scheduler error!")
        
        return [optimizer], [scheduler]
    
    def configure_loss(self):
        self.loss_function = losses.MixLoss(args=self.args)

    def load_model(self):
        net_type = self.args.net_type
        try:
            module = getattr(importlib.import_module(
                '.'+net_type, package='models'), net_type)
        except:
            raise ValueError(
                f'Invalid Network Type or Invalid Class Name models.{net_type}.{net_type}'
            )
        self.model = module(self.args, self.isp)
        
    def backward(self, loss):
        loss.backward()

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

    net = MInterface(args, isp_params)

    print(net)