import torch
from torch.nn import functional as F


def build_loss():
    return loss

def loss(inputs, targets):
    loss = F.cross_entropy(inputs, targets)
    return loss
