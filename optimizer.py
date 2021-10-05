import torch
from torch.nn import parameter


def build_optimizer(params, model):
   optimizer = torch.optim.SGD(
       model.parameters(),
       lr=params.learning_rate
   ) 

   return optimizer