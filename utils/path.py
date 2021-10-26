import os
import time
import os.path as osp

from .misc import is_str

def check_file_exist(filename, msg_tmpl='file "{}" does not exist'):
    if not osp.isfile(filename):
        raise FileNotFoundError(msg_tmpl.format(filename))


def mkdir_or_exist(dir_name, mode=0o777):
    if dir_name == "":
        return
    dir_name = osp.expanduser(dir_name)
    os.makedirs(dir_name, mode=mode, exist_ok=True)

def setup_workspace(work_dir, log_dir_name):
    
    if is_str(work_dir):
        work_dir = osp.abspath(work_dir)
        mkdir_or_exist(work_dir)
    else:
        raise ValueError('"work_dir" must be set')

    if is_str(log_dir_name):
        log_dir = osp.abspath(osp.abspath(work_dir, log_dir_name))
        mkdir_or_exist(log_dir)
    else:
        raise ValueError('"Log dir" must be set')

    timestamp = time.strftime('%Y%m%d_%H%M%S', time.localtime())
    log_file = osp.join(log_dir, f'{timestamp}.log')

    return log_file