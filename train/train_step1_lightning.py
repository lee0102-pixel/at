import os
import time
import pytorch_lightning as pl
from argparse import ArgumentParser
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import TensorBoardLogger
from lightning.pytorch.callbacks import ModelSummary, ModelCheckpoint

from models import MInterface
from dataset import DInterface
from isp.ispparams import ISPParams

from tools.utils import parse_opt

def load_callbacks(args):
    callbacks = []
    callbacks.append(ModelSummary(max_depth=-1))
    checkpoint_callback = ModelCheckpoint(dirpath=os.path.join(args.default_root_dir, args.exp_name),
                                          filename='{epoch}-{val_loss:.4f}',
                                          save_top_k=5,
                                          verbose=True,
                                          monitor='psnr',
                                          mode='max',
                                          prefix='')
    callbacks.append(checkpoint_callback)
    return callbacks


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--opt', type=str, default='./options/unet_step1_lightning.yaml')
    args = parser.parse_args()
    parse_opt(args)

    isp = ISPParams(args)
    isp.write_params_txt(args.log_dir)

    pl.seed_everything(args.seed, workers=True)

    data_module = DInterface(args, isp)
    model = MInterface(args, isp)

    logger = TensorBoardLogger(save_dir=args.log_dir, name=args.exp_name)
    args.callbacks = load_callbacks(args)
    args.logger = logger

    trainer = Trainer.from_argparse_args(args)
    if args.resume:
        trainer.fit(model, data_module, ckpt_path=args.pretrain_path)
    else:
        trainer.fit(model, data_module)
