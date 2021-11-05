from yacs.config import CfgNode as CN

_C = CN()
_C.WORK_DIR = "./work_dir"

_C.DATASET = CN()
_C.DATASET.root_dataset = "./data/"
_C.DATASET.num_class = 21
# multiscale train/test, size of short edge (int or tuple)
_C.DATASET.image_size = (256, 256)
_C.DATASET.crop_size = (224, 224)
# downsampling rate of the segmentation label
_C.DATASET.segm_downsampling_rate = 16
_C.DATASET.batch_size = 16

# -----------------------------------------------------------------------------
# Model
# -----------------------------------------------------------------------------
_C.MODEL = CN()
# The number of feature channels between encoder and decoder
_C.MODEL.fc_dim = 2048


# -----------------------------------------------------------------------------
# Training
# -----------------------------------------------------------------------------
_C.TRAIN = CN()
# epochs to train for
_C.TRAIN.num_epochs = 20
# epoch to start training. useful if continue from a checkpoint
_C.TRAIN.start_epoch = 0

_C.TRAIN.optim = "SGD"
_C.TRAIN.base_lr = 5e-4
_C.TRAIN.momentum = 0.9
_C.TRAIN.weight_decay = 1e-4
_C.TRAIN.lr_min = 5e-6
_C.TRAIN.warmup_lr = 5e-7
_C.TRAIN.warmup_steps = 10
_C.TRAIN.early_stop = 10
_C.TRAIN.resume = False
_C.TRAIN.mnt_mode = "max"
# number of data loading workers
_C.TRAIN.workers = 4
# manual seed
_C.TRAIN.seed = 304
_C.TRAIN.PRINT_FREQ = 2

# -----------------------------------------------------------------------------
# VALID
# -----------------------------------------------------------------------------
_C.VALID = CN()
_C.VALID.dir = "valid"

# -----------------------------------------------------------------------------
# Model
# -----------------------------------------------------------------------------
_C.MODEL = CN()
# the checkpoint to evaluate on
_C.MODEL.check_point = "seg_train.pth"
# number of feature channels between encoder and decoder
_C.MODEL.fc_dim = 2048

_C.LOG = CN()
_C.LOG.dir_name = "log"
_C.LOG.train_file = "train.log"
_C.LOG.tensorboard_file = "tensorborad"



def get_cfg_defaults():
  """Get a yacs CfgNode object with default values for my_project."""
  # Return a clone so that the defaults will not be altered
  # This is for the "local variable" use pattern
  return _C.clone()