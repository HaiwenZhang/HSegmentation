
from .poly import PolynomialLR

def build_scheduler(config, optimizer):

    lr_scheduler = PolynomialLR(
        optimizer,
        max_iters= config.TRAIN.num_epochs,
        power=0.9
    )

    return lr_scheduler