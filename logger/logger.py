import os
import sys
import logging
import functools


def set_logger(log_path):
    """Set the logger to log info in terminal and file `log_path`.
    In general, it is useful to have a logger so that every output to the terminal is saved
    in a permanent file. Here we save it to `model_dir/train.log`.
    Example:
    ```
    logging.info("Starting training...")
    ```
    Args:
        log_path: (string) where to log
    """
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    if not logger.handlers:
        # Logging to a file
        file_handler = logging.FileHandler(log_path)
        file_handler.setFormatter(logging.Formatter('%(asctime)s:%(levelname)s: %(message)s'))
        logger.addHandler(file_handler)

        # Logging to console
        stream_handler = logging.StreamHandler()
        stream_handler.setFormatter(logging.Formatter('%(message)s'))
        logger.addHandler(stream_handler)

#@functools.lru_cache()
#def create_logger(output_dir, dist_rank=0, name=''):
    # create logger
#    logger = logging.getLogger(name)
#    logger.setLevel(logging.DEBUG)
#    logger.propagate = False

    # create formatter
#    fmt = '[%(asctime)s %(name)s] (%(filename)s %(lineno)d): %(levelname)s %(message)s'
#    color_fmt = colored('[%(asctime)s %(name)s]', 'green') + \
#                colored('(%(filename)s %(lineno)d)', 'yellow') + ': %(levelname)s %(message)s'

    # create console handlers for master process
#    if dist_rank == 0:
#        console_handler = logging.StreamHandler(sys.stdout)
#        console_handler.setLevel(logging.DEBUG)
#        console_handler.setFormatter(
#            logging.Formatter(fmt=color_fmt, datefmt='%Y-%m-%d %H:%M:%S'))
#        logger.addHandler(console_handler)

    # create file handlers
#    file_handler = logging.FileHandler(os.path.join(output_dir, f'log_rank{dist_rank}.txt'), mode='a')
#    file_handler.setLevel(logging.DEBUG)
#    file_handler.setFormatter(logging.Formatter(fmt=fmt, datefmt='%Y-%m-%d %H:%M:%S'))
#    logger.addHandler(file_handler)

#    return logger
