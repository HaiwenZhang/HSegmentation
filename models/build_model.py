import torch
import torch.nn as nn

from .backbones import create_resnet50, ResnetDilated
from .heads import DeeplabV3

def build_model(config):

    pretrained_resnet50 = create_resnet50()
    resnet50_dilate = ResnetDilated(pretrained_resnet50, dilate_scale=config.DATASET.segm_downsampling_rate)

    deeplab_v3 = DeeplabV3(resnet50_dilate, config.DATASET.num_class, config.MODEL.fc_dim)

    return deeplab_v3


def build_predict_model(ckpt_path, num_class=21, fc_dim=2048, dilate_scale=8):

    pretrained_resnet50 = create_resnet50(pretrained=False)
    resnet50_dilate = ResnetDilated(pretrained_resnet50, dilate_scale=dilate_scale)

    deeplab_v3 = DeeplabV3(resnet50_dilate, num_class, fc_dim)

    checkpoint = torch.load(ckpt_path, map_location=torch.device('cpu'))
    deeplab_v3.load_state_dict(checkpoint["model"])
    print("Resume model from %s" % ckpt_path)
    del checkpoint
    return deeplab_v3

