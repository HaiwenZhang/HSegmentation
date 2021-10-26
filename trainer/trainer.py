import time
import tqdm
import torch
import os.path as osp
from collections import OrderedDict
from utils.logger import print_log
from utils.misc import get_gpu_memory_use, get_time_str, is_gpu_avaliable

from ..utils import is_str, mkdir_or_exist
from .tracker import AverageMeter

try:
    # noinspection PyUnresolvedReferences
    from apex import amp
except ImportError:
    amp = None

TRAIN_BAR_COLOR = "#00BFFF"
VALID_BAR_COLOR = "#00FF00"

class Trainer(object):
    """
    
    - ``fit()``
    - ``train_one_epoch()``
    - ``val()``
    - ``save_checkpoint()``
    
    """


    def __init__(self,
                config,
                model,
                optimizer,
                criterion,
                metrics,
                logger=None):

        self.config = config
        if amp is not None:
            self.model, self.optimizer = amp.initialize(model, optimizer)
        else:
            self.model = model
            self.optimizer = optimizer

        
        self.criterion = criterion
        self.metrics = metrics
        self.logger = logger
        self.max_epochs = config.TRAIN.MAX_EPOCHS
        self.timestamp = get_time_str()
        self._epoch = 0


    def fit(self,
            train_loader,
            valid_loader, 
            lr_scheduler=None):
        
        for epoch in range(self._epoch, self.max_epochs):
            train_metrics = self.train_one_epoch(train_loader, epoch, lr_scheduler)

            if epoch % self.config.TRAIN.PRINT_FREQ == 0:
                gpu_memory = get_gpu_memory_use()
                lr = self.current_lr()
                momentums = self.current_momentum()

            metrics = self.val(valid_loader, epoch)
            msg = f"Train: []"
            print_log()


    def train_one_epoch(self, 
                        data_loader, 
                        epoch,
                        lr_scheduler=None):
        self.model.train()
        self.model.mode = "train"
        time.sleep(2)

        loss_meter = AverageMeter()
        aAcc_meter = AverageMeter()

        metrics = OrderedDict()

        num_steps = len(data_loader)
        # Use tqdm for progress bar
        with tqdm(total=num_steps, colour=TRAIN_BAR_COLOR) as t:
            for idx, (data, target) in enumerate(data_loader):
                data, target = data.to(self.device), target.to(self.device)

                self.optimizer.zero_grad()
                output = self.model(data)
                loss = self.model.criterion(output, target)

                with amp.scale_loss(loss, self.optimizer) as scaled_loss:
                    scaled_loss.backward()
                self.optimizer.step()

                acc = self.model.metric(output, target)

                loss_meter.update(loss.item())
                aAcc_meter.update(acc.item())

                t.set_description(f"Epoch Train [{epoch}/{self.total_epochs}]")
                t.set_postfix(train_loss="{:05.3f}".format(loss.item()), 
                            train_acc="{:05.3f}".format(acc.item()))    
                t.update()

        if self.lr_scheduler is not None:
            self.lr_scheduler.step()

        metrics["loss"] = loss_meter.avg
        metrics["aAcc"] = aAcc_meter.avg

        return metrics

    @torch.no_grad()
    def val(self, 
            data_loader,
            epoch):
        self.model.eval()
        self.model.mode = "train"
        time.sleep(2)


        loss_meter = AverageMeter()
        aAcc_meter = AverageMeter()
        iou_meter = AverageMeter()

        for idx, (images, target) in enumerate(data_loader):
            if is_gpu_avaliable:
                images = images.cuda(non_blocking=True)
                target = target.cuda(non_blocking=True)

            output = self.model(images)
            loss = self.model.criterion(output, target)
                




    def current_lr(self):
        """Get current learning rates.

        Returns:
            list[float] | dict[str, list[float]]: Current learning rates of all
                param groups. If the runner has a dict of optimizers, this
                method will return a dict.
        """
        if isinstance(self.optimizer, torch.optim.Optimizer):
            lr = [group['lr'] for group in self.optimizer.param_groups]
        elif isinstance(self.optimizer, dict):
            lr = dict()
            for name, optim in self.optimizer.items():
                lr[name] = [group['lr'] for group in optim.param_groups]
        else:
            raise RuntimeError(
                'lr is not applicable because optimizer does not exist.')
        return lr

    
    def current_momentum(self):
        """Get current momentums.

        Returns:
            list[float] | dict[str, list[float]]: Current momentums of all
                param groups. If the runner has a dict of optimizers, this
                method will return a dict.
        """

        def _get_momentum(optimizer):
            momentums = []
            for group in optimizer.param_groups:
                if 'momentum' in group.keys():
                    momentums.append(group['momentum'])
                elif 'betas' in group.keys():
                    momentums.append(group['betas'][0])
                else:
                    momentums.append(0)
            return momentums

        if self.optimizer is None:
            raise RuntimeError(
                'momentum is not applicable because optimizer does not exist.')
        elif isinstance(self.optimizer, torch.optim.Optimizer):
            momentums = _get_momentum(self.optimizer)
        elif isinstance(self.optimizer, dict):
            momentums = dict()
            for name, optim in self.optimizer.items():
                momentums[name] = _get_momentum(optim)
        return momentums

    def _print_metrics(metrics):
        

        print_log()