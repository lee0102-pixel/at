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


class ConvBlock(nn.Module):
    def __init__(self, in_channel, out_channel, strides=1):
        super(ConvBlock, self).__init__()
        self.strides = strides
        self.in_channel=in_channel
        self.out_channel=out_channel
        self.block = nn.Sequential(
            nn.Conv2d(in_channel, out_channel, kernel_size=3, stride=strides, padding=1),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(out_channel, out_channel, kernel_size=3, stride=strides, padding=1),
            nn.LeakyReLU(inplace=True),
        )
        self.conv11 = nn.Conv2d(in_channel, out_channel, kernel_size=1, stride=strides, padding=0)

    def forward(self, x):
        out1 = self.block(x)
        out2 = self.conv11(x)
        out = out1 + out2
        return out


class UNet(ParamsLayer):
    def __init__(self, args, isp, block=ConvBlock):
        super(UNet, self).__init__(args, isp)

        in_channel     = self.args.colors
        out_channel    = self.args.colors
        dim            = self.args.dim
        params_num     = self.isp.get_params_num()
        self.step_flag = self.args.step_flag
        
        self.avg_pool1  = nn.AvgPool2d(kernel_size=2, stride=2)
        self.avg_pool2  = nn.AvgPool2d(kernel_size=4, stride=4)
        self.avg_pool3  = nn.AvgPool2d(kernel_size=8, stride=8)
        self.avg_pool4  = nn.AvgPool2d(kernel_size=16, stride=16)
        
        self.ConvBlock1 = ConvBlock(in_channel+params_num, dim, strides=1)
        self.pool1      = nn.Conv2d(dim,dim,kernel_size=4, stride=2, padding=1)

        self.ConvBlock2 = block(dim+params_num, dim*2, strides=1)
        self.pool2      = nn.Conv2d(dim*2,dim*2,kernel_size=4, stride=2, padding=1)
       
        self.ConvBlock3 = block(dim*2+params_num, dim*4, strides=1)
        self.pool3      = nn.Conv2d(dim*4,dim*4,kernel_size=4, stride=2, padding=1)
       
        self.ConvBlock4 = block(dim*4+params_num, dim*8, strides=1)
        self.pool4      = nn.Conv2d(dim*8, dim*8,kernel_size=4, stride=2, padding=1)

        self.ConvBlock5 = block(dim*8+params_num, dim*16, strides=1)

        self.upv6       = nn.ConvTranspose2d(dim*16, dim*8, 2, stride=2)
        self.ConvBlock6 = block(dim*16, dim*8, strides=1)

        self.upv7       = nn.ConvTranspose2d(dim*8, dim*4, 2, stride=2)
        self.ConvBlock7 = block(dim*8, dim*4, strides=1)

        self.upv8       = nn.ConvTranspose2d(dim*4, dim*2, 2, stride=2)
        self.ConvBlock8 = block(dim*4, dim*2, strides=1)

        self.upv9       = nn.ConvTranspose2d(dim*2, dim, 2, stride=2)
        self.ConvBlock9 = block(dim*2, dim, strides=1)

        self.conv10     = nn.Conv2d(dim, out_channel, kernel_size=3, stride=1, padding=1)
        
        
        if self.step_flag == 1:
            for p in self.parameters():
                p.requires_grad = True
        else:
            for p in self.parameters():
                p.requires_grad = False

        self.set_params_layer()

    def forward(self, x, params_batch=None):
        if params_batch is not None:
            params_layer = params_batch.view(params_batch.size(0), params_batch.size(1), 1, 1)
            params_layer = params_layer.repeat(1, 1, x.size(2), x.size(3))
        else:
            pass
        
        
        if self.step_flag <= 2:
            conv1 = self.ConvBlock1(torch.cat([x, params_layer], dim=1))
        else:
            conv1 = self.ConvBlock1(torch.cat([x, self.params_layer.repeat(x.size(0),1,1,)], dim=1))
        pool1 = self.pool1(conv1)

        if self.step_flag <= 2:
            conv2 = self.ConvBlock2(torch.cat([pool1, self.avg_pool1(params_layer)], dim=1))
        else:
            conv2 = self.ConvBlock2(torch.cat([pool1, self.avg_pool1(self.params_layer.repeat(x.size(0),1,1,))], dim=1))
        pool2 = self.pool2(conv2)

        if self.step_flag <= 2:
            conv3 = self.ConvBlock3(torch.cat([pool2, self.avg_pool2(params_layer)], dim=1))
        else:
            conv3 = self.ConvBlock3(torch.cat([pool2, self.avg_pool2(self.params_layer.repeat(x.size(0),1,1,))], dim=1))
        pool3 = self.pool3(conv3)

        if self.step_flag <= 2:
            conv4 = self.ConvBlock4(torch.cat([pool3, self.avg_pool3(params_layer)], dim=1))
        else:
            conv4 = self.ConvBlock4(torch.cat([pool3, self.avg_pool3(self.params_layer.repeat(x.size(0),1,1,))], dim=1))
        pool4 = self.pool4(conv4)

        if self.step_flag <= 2:
            conv5 = self.ConvBlock5(torch.cat([pool4, self.avg_pool4(params_layer)], dim=1))
        else:
            conv5 = self.ConvBlock5(torch.cat([pool4, self.avg_pool4(self.params_layer.repeat(x.size(0),1,1,))], dim=1))

        up6 = self.upv6(conv5)
        up6 = torch.cat([up6, conv4], 1)
        conv6 = self.ConvBlock6(up6)

        up7 = self.upv7(conv6)
        up7 = torch.cat([up7, conv3], 1)
        conv7 = self.ConvBlock7(up7)

        up8 = self.upv8(conv7)
        up8 = torch.cat([up8, conv2], 1)
        conv8 = self.ConvBlock8(up8)

        up9 = self.upv9(conv8)
        up9 = torch.cat([up9, conv1], 1)
        conv9 = self.ConvBlock9(up9)

        conv10 = self.conv10(conv9)
        
        if self.args.use_skip:
            out = x + conv10
        else:
            out = conv10

        return out


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--opt', type=str, default='./options/unet_step1.yaml')
    args = parser.parse_args()
    from tools.utils import parse_opt
    parse_opt(args)
    from isp.ispparams import ISPParams
    isp_params = ISPParams(args)
    
    net = UNet(args, isp_params)
    x = torch.randn(1, 1, 512, 512)
    param_batchs = torch.randn(1, 5)
    y_hat = net(x, param_batchs)
    print(y_hat.shape)