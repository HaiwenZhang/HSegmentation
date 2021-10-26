import os
import random
import time
import torch

import numpy as np

def is_gpu_avaliable():
    return torch.cuda.is_available()

def is_str(x):
    """Whether the input is an string instance.

    Note: This method is deprecated since python 2 is no longer supported.
    """
    return isinstance(x, str)

def get_time_str():
    return time.strftime('%Y%m%d_%H%M%S', time.localtime())


def get_gpu_memory_use():
    return torch.cuda.max_memory_allocated() / (1024.0 * 1024.0)


def set_random_seed(seed=24, deterministic=False):
    """Set random seed.

    Args:
        seed (int): Seed to be used.
        deterministic (bool): Whether to set the deterministic option for
            CUDNN backend, i.e., set `torch.backends.cudnn.deterministic`
            to True and `torch.backends.cudnn.benchmark` to False.
            Default: False.
    """

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False



