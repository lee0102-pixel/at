import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import sys
# add dir
dir_name = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(dir_name,'.'))
import lpips

class TVLoss(nn.Module):
    def __init__(self):
        super(TVLoss, self).__init__()

    def forward(self, x, *args):
        batch_size = x.size()[0]
        h_x = x.size()[2]
        w_x = x.size()[3]
        count_h = self.tensor_size(x[:, :, 1:, :])
        count_w = self.tensor_size(x[:, :, :, 1:])
        h_tv = torch.pow((x[:, :, 1:, :] - x[:, :, :h_x - 1, :]), 2).sum()
        w_tv = torch.pow((x[:, :, :, 1:] - x[:, :, :, :w_x - 1]), 2).sum()
        return 2 * (h_tv / count_h + w_tv / count_w) / batch_size

    @staticmethod
    def tensor_size(t):
        return t.size()[1] * t.size()[2] * t.size()[3]



class CharbonnierLoss(nn.Module):
    """Charbonnier Loss (L1)"""

    def __init__(self, eps=1e-3):
        super(CharbonnierLoss, self).__init__()
        self.eps = eps

    def forward(self, x, y):
        diff = x - y
        # loss = torch.sum(torch.sqrt(diff * diff + self.eps))
        loss = torch.mean(torch.sqrt((diff * diff) + (self.eps*self.eps)))
        return loss

    
class AlexLoss(nn.Module):
    """Alex Loss"""

    def __init__(self):
        super(AlexLoss, self).__init__()
        with torch.no_grad():
            self.loss_fn = lpips.LPIPS(net='alex').eval()

    def forward(self, x, y):
        loss = torch.mean(self.loss_fn.forward(x, y, normalize=True))
        return loss
    
class MixLoss(nn.Module):
    """Mix Loss"""

    def __init__(self, args):
        super(MixLoss, self).__init__()
        self.loss_type = args.loss_type
        self.colors = args.colors
        
        
        if 'Alex' in self.loss_type:
            with torch.no_grad():
                self.loss_fn_alex = lpips.LPIPS(net='alex').eval()
        if 'VGG' in self.loss_type:
            with torch.no_grad():
                self.loss_fn_vgg16 = lpips.LPIPS(net='vgg').eval()
        if 'L1' in self.loss_type:
            self.loss_fn_l1 = CharbonnierLoss()
        if 'MSE' in self.loss_type:
            self.loss_fn_mse = torch.nn.MSELoss()
        if 'TV' in self.loss_type:
            self.loss_fn_tv = TVLoss()
        if 'TC' in self.loss_type:
            self.loss_fn_tc = CharbonnierLoss()

    def forward(self, x, y=None, out_tc=None):
        loss = 0
        loss_dict = {}
        if self.colors == 1:
            x = x.repeat(1,3,1,1)
            y = y.repeat(1,3,1,1)
            if out_tc is not None:
                out_tc = out_tc.repeat(1,3,1,1)
                
        for loss_type in self.loss_type.split('+'):
            loss_weight = float(loss_type.split('*')[0])
            cur_loss_type = loss_type.split('*')[-1]
            if cur_loss_type == 'L1':
                l1_loss = loss_weight * self.loss_fn_l1(x, y)
                if out_tc is not None:
                    l1_loss += loss_weight * self.loss_fn_l1(out_tc, y)
                loss += l1_loss
                loss_dict['L1'] = l1_loss
            elif cur_loss_type == 'MSE':
                l2_loss = loss_weight * self.loss_fn_mse(x, y)
                if out_tc is not None:
                    l2_loss += loss_weight * self.loss_fn_mse(out_tc, y)
                loss += l2_loss
                loss_dict['MSE'] = l2_loss
            elif cur_loss_type == 'Alex':
                lpips_loss = loss_weight * torch.mean(self.loss_fn_alex.forward(x, y))
                if out_tc is not None:
                    lpips_loss += loss_weight * torch.mean(self.loss_fn_alex.forward(out_tc, y))
                loss += lpips_loss
                loss_dict['Alex'] = lpips_loss
            elif cur_loss_type == 'VGG':
                lpips_loss = loss_weight * torch.mean(self.loss_fn_vgg16.forward(x, y))
                if out_tc is not None:
                    lpips_loss += loss_weight * torch(self.loss_fn_vgg16.forward(out_tc, y))
                loss += lpips_loss
                loss_dict['VGG'] = lpips_loss
            elif cur_loss_type == 'TC' and out_tc is not None:
                tc_loss = loss_weight * self.loss_fn_tc(out_tc, x)
                loss += tc_loss
                loss_dict['TC'] = tc_loss
            elif cur_loss_type == 'TV':
                tv_loss = loss_weight * self.loss_fn_tv(x)
                loss += tv_loss
                loss_dict['TV'] = tv_loss

        return loss, loss_dict

    
if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--opt', type=str, default='./options/unet_step1.yaml')
    args = parser.parse_args()
    sys.path.append(os.path.join(dir_name,'..'))
    import tools.utils as tools
    tools.parse_opt(args)
    
    out_ = torch.ones(1,3,256,256)
    gt_ = torch.zeros(1,3,256,256)
    
    loss = MixLoss(args)
    
    loss, loss_dict = loss.forward(out_, gt_)
    print(loss)