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
    parser.add_argument('--opt', type=str, default='./options/unet_step1.yaml')
    parser.add_argument('--gpus', type=str, default='0')
    args = parser.parse_args()
    tools.parse_opt(args)
    # print(args)
    device = torch.device(args.device)
    # print(device)
    
    #### Logs ####
    now_time = time.strftime('%Y-%m-%d-%H-%M-%S', time.localtime(time.time()))
    # print(now_time)
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
        
    #### optimizer ####
    start_epoch = 0
    if args.optimizer == 'adam':
        optimizer = torch.optim.Adam(model.parameters(), lr=args.init_lr, betas=(0.9, 0.999), eps=1e-8, weight_decay=args.weight_decay)
    elif args.optimizer == 'adamw':
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.init_lr, betas=(0.9, 0.999), eps=1e-8, weight_decay=args.weight_decay)
    else:
        raise Exception("Optimizer error!")
    
    # #### DataParallel ####
    # if torch.cuda.device_count() > 1:
    #     os.environ["CUDA_VISIBLE_DEVICES"] = args.gpus
    #     print("Let's use", torch.cuda.device_count(), "GPUs!")
    #     model = torch.nn.DataParallel(model)
    #     model.cuda()
    
    #### scheduler ####
    if args.scheduler == 'step':
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=args.step_size, gamma=args.gamma)
    elif args.scheduler == 'cosine':
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.T_max, eta_min=args.eta_min)
    else:
        raise Exception("Scheduler error!")
    
       
    #### Resume ####
    if args.resume:
        checkpoint = torch.load(args.pretrain_path)
        start_epoch = checkpoint['epoch']
        model.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        scheduler.load_state_dict(checkpoint['scheduler'])
        
        print('------------------------------------------------------------------------------')
        print('Resume from epoch %d' % start_epoch)
        print('Resume from lr %f' % optimizer.param_groups[0]['lr'])
        print('Resume from ckpt %f' % args.pretrain_path)
        print('------------------------------------------------------------------------------')
        
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
    
    #### Train&Eval ####
    print('====> Start Epoch {} End Epoch {}'.format(start_epoch, args.epochs))
    best_psnr = 0
    best_epoch = 0
    train_iter = 0
    eval_iter = 0
    for epoch in range(start_epoch, args.epochs):
        epoch_start_time = time.time()
        epoch_loss = 0
        model.train()
        
        #### Train ####
        for i, data in enumerate(tqdm(train_loader)):
            train_iter += 1
            x, y, params_batch, _ = data
            x = x.to(device)
            y = y.to(device)
            params_batch = params_batch.to(device)
            
            optimizer.zero_grad()
            y_hat = model(x, params_batch)
            loss, loss_dict = criterion(y_hat, y)
            if train_iter % args.log_freq == 0:
                fabric.log_dict(loss_dict, train_iter)
            epoch_loss += loss.item()
            loss.backward()
            optimizer.step()
            
        epoch_loss /= len(train_loader)
        
        #### Eval ####
        model.eval()
        with torch.no_grad():
            psnr_tmp = 0
            for i, data in enumerate(tqdm(val_loader)):
                eval_iter += 1
                x, y, params_batch, picname = data
                x = x.to(device)
                y = y.to(device)
                params_batch = params_batch.to(device)
                
                y_hat = model(x, params_batch)
                psnr_tmp += tools.get_psnr(y_hat.detach().cpu().numpy(), y.detach().cpu().numpy(), peak=1.0)
                if eval_iter % args.log_freq == 0:
                    fabric.loggers[0].experiment.add_images(picname[0], torch.cat([x[0:1], y_hat[0:1], y[0:1]], dim=0), eval_iter)
        psnr_tmp /= len(val_loader)
        fabric.log('psnr', psnr_tmp, epoch)
        
        if psnr_tmp > best_psnr:
            best_psnr = psnr_tmp
            best_epoch = epoch
            checkpoint = {
                'epoch': epoch,
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'scheduler': scheduler.state_dict(),
            }
            torch.save(checkpoint, os.path.join(model_dir, 'best.pth'))
        info = 'Ep: {}\tLoss {:.4f}\tPSNR {:.4f}\tBest_Ep {}\tBest_PSNR {:.4f}\tTime: {:.4f}\tLR {:.6f}'
        print("------------------------------------------------------------------")
        print(info.format(epoch,
                          epoch_loss,
                          psnr_tmp,
                          best_epoch,
                          best_psnr,
                          time.time()-epoch_start_time,
                          optimizer.param_groups[0]['lr']))
        print("------------------------------------------------------------------")
        scheduler.step()
        
        with open(logname, 'a') as f:
            f.write("------------------------------------------------------------------\n")
            f.write(info.format(epoch,
                                epoch_loss,
                                psnr_tmp,
                                best_epoch,
                                best_psnr,
                                time.time()-epoch_start_time,
                                optimizer.param_groups[0]['lr']))
            f.write("------------------------------------------------------------------\n")
            
        checkpoint = {
                'epoch': epoch,
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'scheduler': scheduler.state_dict(),
            }
        torch.save(checkpoint, os.path.join(model_dir, 'last.pth'))
        if epoch % args.save_freq == 0:
            torch.save(checkpoint, os.path.join(model_dir, 'epoch_%d.pth' % epoch))
            print('Model saved at epoch %d' % epoch)