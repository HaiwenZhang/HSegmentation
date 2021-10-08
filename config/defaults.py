from yacs.config import CfgNode as CN

_C = CN()
_C.OUTPUT = "./output"

_C.DATASET = CN()
_C.DATASET.root_dataset = "./data/"
_C.DATASET.num_class = 21
# multiscale train/test, size of short edge (int or tuple)
_C.DATASET.image_size = (256, 256)
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
_C.TRAIN.lr_encoder = 0.02
_C.TRAIN.lr_decoder = 0.02
# power in poly to drop LR
_C.TRAIN.lr_pow = 0.9
# momentum for sgd, beta1 for adam
_C.TRAIN.beta1 = 0.9
# weights regularizer
_C.TRAIN.weight_decay = 1e-4

_C.TRAIN.early_stop = 10
_C.TRAIN.resume = False
_C.TRAIN.mnt_mode = "max"
# number of data loading workers
_C.TRAIN.workers = 4
# manual seed
_C.TRAIN.seed = 304


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