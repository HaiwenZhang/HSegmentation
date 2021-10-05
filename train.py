
import os
import time
import argparse
import datetime
import logging

import numpy as np
from tqdm import tqdm

import torch
import torch.backends.cudnn as cudnn

from models import build_model
from datasets import build_loader
from loss import build_loss, pixel_acc
from optimizer import build_optimizer
from logger import set_logger
from utils import AverageMeter, Params, setup_seed

def parse_option():
    parser = argparse.ArgumentParser('DeeplabV3 training script', add_help=False)

    parser.add_argument('--batch-size', type=int, help="batch size for single GPU")
    parser.add_argument('--data-path', type=str, help='path to dataset')

    parser.add_argument('--output', default='output', type=str, metavar='PATH',
                        help='root of output folder, the full path is <output>/<model_name>/<tag> (default: output)')

    args, unparsed = parser.parse_known_args()

    return args

def main(params):
    data_loader_train, data_loader_val = build_loader(params.data_dir)

    model = build_model(params.num_class)

    if params.cuda:
        model.cuda()

    logging.info(str(model))

    optimizer = build_optimizer(params, model)


    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logging.info(f"number of params: {n_parameters}")

    #lr_scheduler = build_scheduler(config, optimizer, len(data_loader_train))

    criterion = build_loss()

    max_accuracy = 0.0

    logging.info("Start training")
    start_time = time.time()
    for epoch in range(0, params.epochs):

        train_one_epoch(params, model, criterion, data_loader_train, optimizer, epoch)

        acc, loss = validate(params, data_loader_val, model)

        max_accuracy = max(max_accuracy, acc)
        #logger.info(f'Max accuracy: {max_accuracy:.2f}%')

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    logging.info('Training time {}'.format(total_time_str))


def train_one_epoch(params, model, criterion, data_loader, optimizer, epoch):
    model.train()
    optimizer.zero_grad()

    num_steps = len(data_loader)
    batch_time = AverageMeter()
    loss_meter = AverageMeter()

    start = time.time()
    end = time.time()

    with tqdm(total=len(data_loader)) as t:
        for idx, (samples, targets) in enumerate(data_loader):

            if params.cuda:
                samples = samples.cuda(non_blocking=True)
                targets = targets.cuda(non_blocking=True)

            outputs = model(samples)

            loss = criterion(outputs, targets)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            #lr_scheduler.step_update(epoch * num_steps + idx)

            loss_meter.update(loss.item(), targets.size(0))
            batch_time.update(time.time() - end)
            end = time.time()

            #if idx % config.PRINT_FREQ == 0:
            #    lr = optimizer.param_groups[0]['lr']
            #    memory_used = torch.cuda.max_memory_allocated() / (1024.0 * 1024.0)
            #    etas = batch_time.avg * (num_steps - idx)
            #    logger.info(
            #        f'Train: [{epoch}/{config.TRAIN.EPOCHS}][{idx}/{num_steps}]\t'
            #        f'eta {datetime.timedelta(seconds=int(etas))} lr {lr:.6f}\t'
            #        f'time {batch_time.val:.4f} ({batch_time.avg:.4f})\t'
            #        f'loss {loss_meter.val:.4f} ({loss_meter.avg:.4f})\t'
            #        f'mem {memory_used:.0f}MB')

        t.set_postfix(loss="{:05.3f}".format(loss_meter.avg))        

        epoch_time = time.time() - start
        logging.info(f"EPOCH {epoch} training takes {datetime.timedelta(seconds=int(epoch_time))}")


@torch.no_grad()
def validate(params, data_loader, model):
    criterion = torch.nn.CrossEntropyLoss()
    model.eval()

    batch_time = AverageMeter()
    loss_meter = AverageMeter()
    pixel_acc_meter = AverageMeter()

    end = time.time()
    for idx, (images, target) in enumerate(data_loader):

        if params.cuda:
            images = images.cuda(non_blocking=True)
            target = target.cuda(non_blocking=True)

        # compute output
        output = model(images)

        # measure accuracy and record loss
        loss = criterion(output, target)
        acc = pixel_acc(output, target)


        loss_meter.update(loss.item(), target.size(0))
        pixel_acc_meter.update(acc)

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        #if idx % config.PRINT_FREQ == 0:
        #    memory_used = torch.cuda.max_memory_allocated() / (1024.0 * 1024.0)
        #    logging.info(
        #        f'Test: [{idx}/{len(data_loader)}]\t'
        #        f'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
        #        f'Loss {loss_meter.val:.4f} ({loss_meter.avg:.4f})\t'
        #        f'Mem {memory_used:.0f}MB')
    return pixel_acc_meter.avg, loss_meter.avg


if __name__ == '__main__':
    #_, config = parse_option()
    
    #torch.cuda.set_device()

    # linear scale the learning rate according to total batch size, may not be optimal
    #linear_scaled_lr = config.TRAIN.BASE_LR * config.DATA.BATCH_SIZE * dist.get_world_size() / 512.0
    #linear_scaled_warmup_lr = config.TRAIN.WARMUP_LR * config.DATA.BATCH_SIZE * dist.get_world_size() / 512.0
    #linear_scaled_min_lr = config.TRAIN.MIN_LR * config.DATA.BATCH_SIZE * dist.get_world_size() / 512.0


    #os.makedirs(config.OUTPUT, exist_ok=True)
    #logger = create_logger(output_dir=config.OUTPUT, dist_rank=dist.get_rank(), name=f"{config.MODEL.NAME}")

    # print config

    json_path = "params.json"

    assert os.path.isfile(
        json_path), "No json configuration file found at {}".format(json_path)

    params = Params(json_path)

    params.cuda = torch.cuda.is_available()

    setup_seed(params.seed)
    main(params)
