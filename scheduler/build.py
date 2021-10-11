
from .poly_lr import PolyLRScheduler

def build_scheduler(config, optimizer):

    lr_scheduler = PolyLRScheduler(
        optimizer,
        t_initial=config.TRAIN.num_epochs,
        lr_min=config.TRAIN.lr_min,
        warmup_lr_init=config.TRAIN.warmup_lr,
        warmup_t=config.TRAIN.warmup_steps,
        t_in_epochs=False
    )

    return lr_scheduler