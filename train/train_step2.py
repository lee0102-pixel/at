import os
import sys

dir_name = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(dir_name,'..'))

import argparse
import tools
from dataset.rgb2rgb import *
import losses.losses as losses
from isp.ispparams import ISPParams
import torch
import lightning as L
from tqdm import tqdm
import time
from lightning.fabric.loggers import TensorBoardLogger

if __name__ == '__main__':
    #### Arguments ####
    parser = argparse.ArgumentParser()
    parser.add_argument('--opt', type=str, default='./options/unet_step2.yaml')
    args = parser.parse_args()
    tools.parse_opt(args)
    device = torch.device('mps')
    
    #### Logs ####
    now_time = time.strftime('%Y-%m-%d-%H-%M-%S', time.localtime(time.time()))
    log_dir = os.path.join(args.log_dir, args.exp_name, now_time)
    os.makedirs(log_dir, exist_ok=True)
    logname = os.path.join(log_dir, 'log.txt')
    model_dir = os.path.join(log_dir, 'models')
    os.makedirs(model_dir, exist_ok=True)
    
    tb_dir = os.path.join(log_dir, 'tensorboard')
    os.makedirs(tb_dir, exist_ok=True)
    logger = TensorBoardLogger(root_dir=tb_dir)
    fabric = L.Fabric(loggers=logger)
    
    #### set seed ####
    tools.setup_seed(args.seed)
    
    #### ISP ####
    isp = ISPParams(args)
    isp.write_params_txt(log_dir)
    
    #### Model ####
    model = tools.get_arch(args, isp)
    model = model.to(device)
    
    
    with open(logname, 'w') as f:
        f.write(str(args))
        f.write(str(model))
        f.write('\n')
        
    #### Resume ####
    checkpoint = torch.load(args.pretrain_path)
    model.load_state_dict(checkpoint['model'])
    print('====> Resume from ckpt %f' % args.pretrain_path)
    
    #### Loss ####
    criterion = losses.MixLoss(args=args)
    criterion = criterion.to(device)
     
    #### Dataset ####
    train_loader = train_dataloader(args, isp)
    val_loader   = val_dataloader(args, isp)
    
    len_trainset = len(train_loader.dataset)
    len_valset = len(val_loader.dataset)
    print('====>Train set length: %d' % len_trainset)
    print('====>Val set length: %d' % len_valset)
    
    #### Init params ####
    params_list = model.get_params_list()
    # denormed_list = model.isp.get_denormed_list(params_list)
    print('====> Init params: %s' % str(params_list))
    
    #### Train&Eval ####
    print('====>End Epoch {}'.format(args.epochs))
    best_psnr = 0
    best_epoch = 0
    train_iter = 0
    eval_iter = 0
    LR = args.init_lr
    
    for epoch in range(args.epochs):
        epoch_start_time = time.time()
        epoch_loss = 0
        model.eval()
        
        for i, data in enumerate(tqdm(train_loader)):
            # optimizer = torch.optim.Adam([model.module.params_layer], LR) # for dataparaller
            optimizer = torch.optim.Adam([model.params_layer], LR)
            train_iter += 1
            x, y, _, _ = data
            x = x.to(device)
            y = y.to(device)

            y_hat = model(x)
            loss, loss_dict = criterion(y_hat, y)
            
            epoch_loss += loss.item()
            
            if train_iter % args.log_freq == 0:
                fabric.log_dict(loss_dict, train_iter)
            
            optimizer.zero_grad()
            loss.backward()
            
            for idx in range(len(model.isp.params_name_list)):
                avg_grad = model.params_layer.grad.data[0, idx, :, :].mean().float()
                model.params_layer.grad.data[0, idx, :, :] = avg_grad
            
            optimizer.step()
            model.update_params_layer()
            
            
        epoch_loss /= len(train_loader)
        
        model.eval()
        with torch.no_grad():
            psnr_tmp = 0
            for i, data in enumerate(tqdm(val_loader)):
                eval_iter += 1
                x, y, _, picname = data
                x = x.to(device)
                y = y.to(device)
                
                y_hat = model(x)

                psnr_tmp += tools.get_psnr(y_hat.detach().cpu().numpy(), y.detach().cpu().numpy(), peak=1.0)
                
                if eval_iter % args.log_freq == 0:
                    fabric.loggers[0].experiment.add_images(picname[0], torch.cat([x[0:1], y_hat[0:1], y[0:1]], dim=0), eval_iter)
                    
        psnr_tmp /= len(val_loader)
        fabric.log('psnr', psnr_tmp, epoch)
        
        if psnr_tmp > best_psnr:
            best_psnr = psnr_tmp
            best_epoch = epoch
            params_list = model.get_params_list()
            model.isp.write_results(params_list, os.path.join(log_dir, 'best_params.txt'))

        info = 'Ep: {}\tLoss {:.4f}\tPSNR {:.4f}\tBest_Ep {}\tBest_PSNR {:.4f}\tTime: {:.4f}\tLR {:.6f}'
        
        print("------------------------------------------------------------------")
        print(info.format(epoch, 
                          epoch_loss, 
                          psnr_tmp, 
                          best_epoch, 
                          best_psnr,
                          time.time()-epoch_start_time,
                          LR))
        print("------------------------------------------------------------------")
        
        with open(logname, 'a') as f:
            f.write("------------------------------------------------------------------\n")
            f.write(info.format(epoch, 
                                epoch_loss, 
                                psnr_tmp, 
                                best_epoch, 
                                best_psnr,
                                time.time()-epoch_start_time,
                                LR))
            f.write("------------------------------------------------------------------\n")

        params_list = model.get_params_list()
        model.isp.write_results(params_list, os.path.join(log_dir, 'last_params.txt'))

        if (epoch+1) % 2 == 0:
            LR *= 0.8
        if (epoch+1) %10 == 0:
            LR = 0.002