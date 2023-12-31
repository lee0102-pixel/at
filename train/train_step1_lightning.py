import os
import sys

dir_name = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(dir_name,'..'))
import time
import pytorch_lightning as pl
from argparse import ArgumentParser
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import TensorBoardLogger
import pytorch_lightning.callbacks as plc

from models import MInterface
from dataset import DInterface
from isp.ispparams import ISPParams

from tools.utils import parse_opt

def load_callbacks(args):
    callbacks = []
    callbacks.append(plc.ModelSummary(max_depth=-1))
    callbacks.append(plc.LearningRateMonitor(logging_interval='epoch'))
    callbacks.append(plc.ModelCheckpoint(dirpath=args.ckpt_dir,
                                         monitor='psnr',
                                         save_top_k=3,
                                         mode='max',
                                         save_weights_only=True,
                                         filename='{epoch}-{psnr:.4f}'))
    return callbacks


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--opt', type=str, default='./options/unet_step1.yaml')
    args = parser.parse_args()
    parse_opt(args)
    isp = ISPParams(args)
    os.makedirs(args.default_root_dir, exist_ok=True)
    os.makedirs(args.tb_dir, exist_ok=True)
    os.makedirs(args.ckpt_dir, exist_ok=True)
    isp.write_params_txt(args.default_root_dir)

    pl.seed_everything(args.seed, workers=True)

    data_module = DInterface(args, isp)
    model = MInterface(args, isp)

    logger = TensorBoardLogger(save_dir=args.tb_dir)
    callbacks = load_callbacks(args)

    trainer = Trainer(logger=logger, callbacks=callbacks, **args.opt['trainer'])
    if args.resume:
        trainer.fit(model, data_module, ckpt_path=args.pretrain_path)
    else:
        trainer.fit(model, data_module)
