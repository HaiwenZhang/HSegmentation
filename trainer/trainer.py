import os
import time
import datetime
import torch
from tqdm import tqdm

from .tracker import MetricTracker

class Trainer(object):
    """
    
    """
    def __init__(self, logger, model, criterion, metric, optimizer, config, device,
                 train_data_loader, valid_data_loader, lr_scheduler=None):
        self.config = config
        self.logger = logger

        self.model = model
        self.criterion = criterion
        self.metric = metric
        self.optimizer = optimizer
        
        self.num_epochs = config.TRAIN.num_epochs
        self.start_epoch = config.TRAIN.start_epoch
        self.total_epochs = self.num_epochs - self.start_epoch

        self.device = device
        self.train_data_loader = train_data_loader
 
        self.valid_data_loader = valid_data_loader
        self.do_validation = self.valid_data_loader is not None
        self.lr_scheduler = lr_scheduler

        self.train_metrics = MetricTracker("loss", "acc")
        self.valid_metrics = MetricTracker("loss", "acc")

        self.early_stop = config.TRAIN.early_stop

        if config.TRAIN.resume:
            self._resume_checkpoint(config)

    def train(self):
        """
        Full training logic
        """
        not_improved_count = 0
        for epoch in range(self.start_epoch, self.num_epochs + 1):
            self._train_epoch(epoch)

            # get valid acc
            valid_acc = self.valid_metrics.avg('acc')

            self._save_checkpoint(epoch, valid_acc)

    def _train_epoch(self, epoch):
        """
        Training logic for an epoch

        :param epoch: Current epoch number
        """
        start = time.time()

        self.model.train()
        self.train_metrics.reset()

         # Use tqdm for progress bar
        with tqdm(total=len(self.train_data_loader)) as t:
            for _, (data, target) in enumerate(self.train_data_loader):
                data, target = data.to(self.device), target.to(self.device)

                self.optimizer.zero_grad()
                output = self.model(data)
                loss = self.criterion(output, target)
                loss.backward()
                self.optimizer.step()

                acc = self.metric(output, target)

                self.train_metrics.update("loss", loss.item())
                self.train_metrics.update("acc", acc.item())

                t.set_description(f"Epoch [{epoch}/{self.total_epochs}]")
                t.set_postfix(train_loss="{:05.3f}".format(loss.item()), 
                            train_acc="{:05.3f}".format(acc.item()))    
                t.update()

        if self.do_validation:
            self._valid_epoch(epoch)

        if self.lr_scheduler is not None:
            self.lr_scheduler.step()

        epoch_time = time.time() - start
        memory_used = torch.cuda.max_memory_allocated() / (1024.0 * 1024.0)
        self.logger.info(
                f"Train: [{epoch}/{self.total_epochs}]\t"
                f"one epoch training takes time {datetime.timedelta(seconds=int(epoch_time))}\t"
                f"train loss: {self.train_metrics.avg('loss'):.4f} val loss: {self.valid_metrics.avg('loss'):.4f}\t"
                f"train acc: {self.train_metrics.avg('acc'):.4f} val acc: {self.valid_metrics.avg('acc'):.4f}\t"
                f"mem {memory_used:.0f}MB")

    def _valid_epoch(self, epoch):
        """
        Validate after training an epoch

        :param epoch: Integer, current training epoch.
        """
        self.model.eval()
        self.valid_metrics.reset()
        with torch.no_grad():
            for batch_idx, (data, target) in enumerate(self.valid_data_loader):
                data, target = data.to(self.device), target.to(self.device)

                output = self.model(data)
                loss = self.criterion(output, target)
                
                self.valid_metrics.update("loss", loss.item())
                self.valid_metrics.update("acc", self.metric(output, target))

    def _save_checkpoint(self, epoch, max_accuracy):
        """
        Saving checkpoints

        :param epoch: current epoch number
        :param log: logging information of the epoch
        :param save_best: if True, rename the saved checkpoint to 'model_best.pth'
        """
        state = {
            'model': self.model.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'max_accuracy': max_accuracy,
            'epoch': epoch,
            'config': self.config
        }

        save_path = os.path.join(self.config.OUTPUT, f'ckpt_epoch_{epoch}.pth')
        torch.save(state, save_path)
        self.logger.info("Saving checkpoint: {} ...".format(save_path))
        # if save_best:
        #     best_path = os.path.join(self.config.OUTPUT, f'model_best.pth')
        #     torch.save(state, best_path)
        #     self.logger.info("Saving current best: model_best.pth ...")

    def _resume_checkpoint(self):
        """
        Resume from saved checkpoints

        """
        

