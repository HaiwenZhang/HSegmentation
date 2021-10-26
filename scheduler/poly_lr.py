from torch.optim.lr_scheduler import _LRScheduler, StepLR

class PolyLRScheduler(_LRScheduler):
    def __init__(self, optimizer, max_epoch, power=0.9, min_lr=1e-6):
        self.power = power
        self.max_epoch = max_epoch 
        self.min_lr = min_lr
        super(PolyLRScheduler, self).__init__(optimizer)
    
    def get_lr(self):
        coeff = (1 - self.last_epoch /self.max_iters) ** self.power
        return [(base_lr - self.min_lr) * coeff + self.min_lr for base_lr in self.base_lrs]