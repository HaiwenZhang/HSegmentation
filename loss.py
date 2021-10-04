from torch.nn import functional as F

def build_loss():
    return loss

def loss(inputs, targets):
    return F.cross_entropy(inputs, targets, reduction='none').mean(1).mean(1)