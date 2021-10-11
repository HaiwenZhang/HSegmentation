import torch
from torch.nn import parameter


def build_optimizer(config, model):
    optimizer = torch.optim.SGD(params=[
        {'params': model.backbone.parameters(), 'lr': 0.1*config.TRAIN.base_lr},
        {'params': model.aspp.parameters(), 'lr': config.TRAIN.base_lr},
    ], lr=config.TRAIN.base_lr, momentum=config.TRAIN.momentum, weight_decay=config.TRAIN.weight_decay)

    return optimizer