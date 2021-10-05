import torch
from torch.nn import functional as F


def build_loss():
    return loss

def loss(inputs, targets):
    #return F.cross_entropy(inputs, targets, reduction='none').mean(1).mean(1)
    loss = torch.nn.CrossEntropyLoss(inputs, targets)
    return loss


def pixel_acc(pred, label):
        _, preds = torch.max(pred, dim=1)
        valid = (label >= 0).long()
        acc_sum = torch.sum(valid * (preds == label).long())
        pixel_sum = torch.sum(valid)
        acc = acc_sum.float() / (pixel_sum.float() + 1e-10)
        return acc