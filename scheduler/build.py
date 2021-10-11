
from .poly_lr import PolyLRScheduler

def build_scheduler(config, optimizer):

    lr_scheduler = PolyLRScheduler(
        optimizer,
        max_iters= config.TRAIN.num_epochs,
        power=0.9
    )

    return lr_scheduler