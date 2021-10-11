
import os
import time
import argparse
import datetime
import logging

import numpy as np

import torch

from config import cfg
from models import build_model
from datasets import build_loader
from loss import build_loss
from metrics import pixel_acc
from optimizer import build_optimizer
from scheduler import build_scheduler
from logger import create_logger
from utils import setup_seed
from trainer import Trainer

def parse_option():
    parser = argparse.ArgumentParser('DeeplabV3 training script', add_help=False)
    parser.add_argument('--cfg', type=str, help='config file path')
    parser.add_argument('--gpu', type=str, help='GPU to use')
    parser.add_argument('--is_save_cfg', type=bool, default=True, help='If save train config file')

    args, unparsed = parser.parse_known_args()

    return args

def main(config):

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    train_data_loader, valid_data_loader = build_loader(config)

    model = build_model(config)
    model.to(device)

    optimizer = build_optimizer(config, model)
    lr_scheduler = build_scheduler(config, optimizer)
    criterion = build_loss()
    acc = pixel_acc

    trainer = Trainer(logger, model, criterion, 
                        acc, optimizer, config, 
                        device, train_data_loader, valid_data_loader, lr_scheduler)

    logger.info("Start training")
    start_time = time.time()
    trainer.train()
    total_time = time.time() - start_time

    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    logging.info('Training time {}'.format(total_time_str))


if __name__ == '__main__':

    args = parse_option()

    cfg.merge_from_file(args.cfg)

    if not os.path.isdir(cfg.OUTPUT):
        os.makedirs(cfg.OUTPUT)
    log_dir = os.path.join(cfg.OUTPUT, cfg.LOG.dir_name)
    if not os.path.isdir(log_dir):
        os.makedirs(log_dir)

    logger = create_logger(log_dir)

    if args.is_save_cfg:
        path = os.path.join(cfg.OUTPUT, "config.yml")
        with open(path, 'w') as f:
            f.write("{}".format(cfg))
        logger.info(f"Full config saved to {path}")
    
    setup_seed(cfg.TRAIN.seed)
    main(cfg)
