import os
import cv2
import time
import datetime
from apex.amp.handle import scale_loss
import torch
from tqdm import tqdm

try:
    # noinspection PyUnresolvedReferences
    from apex import amp
except ImportError:
    amp = None

from .tracker import MetricTracker
from datasets import label2image


TRIAN_BAR_COLOR = "#00BFFF"
VALID_BAR_COLOR = "#00FF00"



def train():
    pass


class Trainer(object):
    """
    
    """
    def __init__(self, logger, model, criterion, metric, optimizer, config, device,
                 train_data_loader, valid_data_loader, lr_scheduler=None):
        self.config = config
        self.logger = logger

        self.model = model
        self.optimizer = optimizer
        self.criterion = criterion
        self.metric = metric

        self.model, self.optimizer = amp.initialize(model, optimizer, opt_level="O1")
        
        self.num_epochs = config.TRAIN.num_epochs
        self.start_epoch = config.TRAIN.start_epoch
        self.total_epochs = self.num_epochs - self.start_epoch

        self.device = device
        self.train_data_loader = train_data_loader
 
        self.valid_data_loader = valid_data_loader
        self.do_validation = self.valid_data_loader is not None
        self.lr_scheduler = lr_scheduler

        self.train_metrics = MetricTracker("loss", "aAcc")
        self.valid_metrics = MetricTracker("loss", "aAcc", "mIou", "mAcc")

        self.early_stop = config.TRAIN.early_stop

        if config.TRAIN.resume:
            self._resume_checkpoint(config)

    def train(self):
        """
        Full training logic
        """
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

        num_steps = len(self.train_data_loader)
        # Use tqdm for progress bar
        with tqdm(total=num_steps, colour=TRIAN_BAR_COLOR) as t:
            for idx, (data, target) in enumerate(self.train_data_loader):
                data, target = data.to(self.device), target.to(self.device)

                self.optimizer.zero_grad()
                output = self.model(data)
                loss = self.criterion(output, target)

                with amp.scale_loss(loss, self.optimizer) as scaled_loss:
                    scaled_loss.backward()
                self.optimizer.step()

                acc = self.metric(output, target)

                self.train_metrics.update("loss", loss.item())
                self.train_metrics.update("acc", acc.item())


                if self.lr_scheduler is not None:
                    self.lr_scheduler.step()


                t.set_description(f"Epoch Train [{epoch}/{self.total_epochs}]")
                t.set_postfix(train_loss="{:05.3f}".format(loss.item()), 
                            train_acc="{:05.3f}".format(acc.item()))    
                t.update()

        if self.do_validation:
            self._valid_epoch(epoch)


        epoch_time = time.time() - start

        self.logger.info(
                f"Train: [{epoch}/{self.total_epochs}]\t"
                f"one epoch training takes time {datetime.timedelta(seconds=int(epoch_time))}\t"
                f"train loss: {self.train_metrics.avg('loss'):.4f} val loss: {self.valid_metrics.avg('loss'):.4f}\t"
                f"train acc: {self.train_metrics.avg('acc'):.4f} val acc: {self.valid_metrics.avg('acc'):.4f}\t")

    def _valid_epoch(self, epoch):
        """
        Validate after training an epoch

        :param epoch: Integer, current training epoch.
        """
        self.model.eval()
        self.valid_metrics.reset()
        with torch.no_grad():
             # Use tqdm for progress bar
            with tqdm(total=len(self.valid_data_loader), colour=VALID_BAR_COLOR) as t:
                for batch_idx, (data, target) in enumerate(self.train_data_loader):
                    data, target = data.to(self.device), target.to(self.device)

                    output = self.model(data)
                    loss = self.criterion(output, target)
                    acc = self.metric(output, target)
                
                    self.valid_metrics.update("loss", loss.item())
                    self.valid_metrics.update("acc", acc.item())

                    self._save_samplers(epoch, batch_idx, data, target, output)

                    t.set_description(f"Epoch Valid [{epoch}/{self.total_epochs}]")
                    t.set_postfix(valid_loss="{:05.3f}".format(loss.item()), 
                            valid_acc="{:05.3f}".format(acc.item()))    
                    t.update()


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
        

    def _save_samplers(self, epoch, batch_idx, data, label, pred):

        sample_num=1
        
        pred = pred.argmax(dim=1)



        image = data[sample_num,:].clone().detach().cpu().numpy().transpose((1, 2, 0)) * 255
        pred = pred[sample_num,:].clone()
        label = label[sample_num,:].clone()

        pred = label2image(pred, self.device).detach().cpu().numpy()
        label = label2image(label, self.device).detach().cpu().numpy()

        valid_dir = os.path.join(self.config.OUTPUT, self.config.VALID.dir)
        if not os.path.isdir(valid_dir):
            os.makedirs(valid_dir)

        image_path = os.path.join(valid_dir, f"epoch_{epoch}_batch_idx_{batch_idx}.jpg")
        pred_path = os.path.join(valid_dir, f"epoch_{epoch}_batch_idx_{batch_idx}_pred.png")
        label_path = os.path.join(valid_dir, f"epoch_{epoch}_batch_idx_{batch_idx}_gt.png")

        cv2.imwrite(image_path, image)
        cv2.imwrite(pred_path, pred)
        cv2.imwrite(label_path, label)
        


