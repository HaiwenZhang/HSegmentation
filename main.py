
import os
import time
import argparse
import datetime
import logging

from utils import (
    is_str, 
    mkdir_or_exist,
    get_default_logger,
    print_log,
    setup_seed, 
    setup_workspace)

import numpy as np

import torch

from models import build_model
from datasets import build_loader
from loss import build_loss
from metrics import build_metrics
from optimizer import build_optimizer
from scheduler import build_scheduler
from trainer import Trainer

from .config import get_cfg_defaults

def parse_option():
    parser = argparse.ArgumentParser('Segmentation Train', add_help=False)
    parser.add_argument('--cfg', type=str, help='config file path')
    parser.add_argument('--gpu', type=str, help='GPU to use')

    args, unparsed = parser.parse_known_args()

    cfg = get_cfg_defaults()
    cfg.merge_from_file(args.cfg)
    cfg.merge_from_list(args.opts)
    cfg.freeze()

    return args, cfg

def main(config, logger):

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    train_data_loader, valid_data_loader = build_loader(config)

    model = build_model(config)
    model.to(device)

    optimizer = build_optimizer(config, model)
    lr_scheduler = build_scheduler(config, optimizer)
    criterion = build_loss()
    metrics = build_metrics(config)


    trainer = Trainer(config, model, criterion, metrics, logger)

    print_log("Start training")
    start_time = time.time()

    trainer.fit(train_data_loader, valid_data_loader, lr_scheduler=lr_scheduler)

    total_time = time.time() - start_time

    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print_log('Training time {}'.format(total_time_str))



if __name__ == '__main__':

    args, cfg= parse_option()

    if is_str(cfg.WORK_DIR):
        work_dir = os.path.abspath(cfg.WORK_DIR)
        mkdir_or_exist(work_dir)
    else:
        raise ValueError('"work_dir" must be set')

    if is_str(cfg.LOG.dir_name):
        log_dir = os.path.abspath(os.path.abspath(work_dir, cfg.LOG.dir_name))
        mkdir_or_exist(log_dir)
    else:
        raise ValueError('"Log dir" must be set')

    timestamp = time.strftime('%Y%m%d_%H%M%S', time.localtime())
    log_file = os.path.join(log_dir, f'{timestamp}.log')
    
    logger = get_default_logger("HSeg", log_file=log_file)

    if args.is_save_cfg:
        path = os.path.join(cfg.OUTPUT, "config.yml")
        with open(path, 'w') as f:
            f.write("{}".format(cfg))

        print_log(f"Full config save to {path}")
    
    setup_seed(cfg.TRAIN.seed)
    main(cfg, logger)
