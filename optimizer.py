import torch
from torch.nn import parameter


def build_optimizer(config, model):
   optimizer = torch.optim.SGD(
       model.parameters(),
       lr=config.TRAIN.base_lr
   ) 

   return optimizer