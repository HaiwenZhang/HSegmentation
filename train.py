
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
    data_loader_train, data_loader_val = build_loader(params)

    model = build_model(params.num_class)

    if params.cuda:
        model.cuda()

    logging.info(str(model))

    optimizer = build_optimizer(params, model)

    criterion = build_loss()

    max_accuracy = 0.0

    logging.info("Start training")
    start_time = time.time()
    for epoch in range(0, params.epochs):

        train_one_epoch(params, model, criterion, data_loader_train, optimizer, epoch, params.epochs)

        acc, loss = validate(params, data_loader_val, model)

        max_accuracy = max(max_accuracy, acc)

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    logging.info('Training time {}'.format(total_time_str))


def train_one_epoch(params, model, criterion, dataloader, optimizer, epoch, num_epochs):
    model.train()
    optimizer.zero_grad()

    loss_meter = AverageMeter()
    start = time.time()


    # Use tqdm for progress bar
    with tqdm(total=len(dataloader)) as t:
        for i, (samples, targets) in enumerate(dataloader):
            # move to GPU if available
            if params.cuda:
                samples, targets = samples.cuda(
                    non_blocking=True), targets.cuda(non_blocking=True)

            # compute model output and loss
            outputs = model(samples)
            loss = criterion(outputs, targets)

            # clear previous gradients, compute gradients of all variables wrt loss
            optimizer.zero_grad()
            loss.backward()

            # performs updates using calculated gradients
            optimizer.step()

            # update the average loss

            loss_meter.update(loss.item(), targets.size(0))

            t.set_description(f"Epoch [{epoch}/{num_epochs}]")
            t.set_postfix(loss="{:05.3f}".format(loss_meter.avg))    
            t.update()

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

    return pixel_acc_meter.avg, loss_meter.avg


if __name__ == '__main__':

    json_path = "params.json"

    assert os.path.isfile(
        json_path), "No json configuration file found at {}".format(json_path)

    params = Params(json_path)

    params.cuda = torch.cuda.is_available()

    setup_seed(params.seed)
    main(params)
