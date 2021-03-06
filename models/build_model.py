import torch
import torch.nn as nn

from .backbones import create_resnet50
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

class ResnetDilated(nn.Module):
    def __init__(self, orig_resnet, dilate_scale=16):
        super(ResnetDilated, self).__init__()
        from functools import partial

        if dilate_scale == 8:
            orig_resnet.layer3.apply(
                partial(self._nostride_dilate, dilate=2))
            orig_resnet.layer4.apply(
                partial(self._nostride_dilate, dilate=4))
        elif dilate_scale == 16:
            orig_resnet.layer4.apply(
                partial(self._nostride_dilate, dilate=2))

        # take pretrained resnet, except AvgPool and FC
        self.conv1 = orig_resnet.conv1
        self.bn1 = orig_resnet.bn1
        self.relu = orig_resnet.relu
        self.maxpool = orig_resnet.maxpool

        self.layer1 = orig_resnet.layer1
        self.layer2 = orig_resnet.layer2
        self.layer3 = orig_resnet.layer3
        self.layer4 = orig_resnet.layer4

    def _nostride_dilate(self, m, dilate):
        classname = m.__class__.__name__
        if classname.find('Conv') != -1:
            # the convolution with stride
            if m.stride == (2, 2):
                m.stride = (1, 1)
                if m.kernel_size == (3, 3):
                    m.dilation = (dilate//2, dilate//2)
                    m.padding = (dilate//2, dilate//2)
            # other convoluions
            else:
                if m.kernel_size == (3, 3):
                    m.dilation = (dilate, dilate)
                    m.padding = (dilate, dilate)

    def forward(self, x, return_feature_maps=False):

        x = self.relu(self.bn1(self.conv1(x)))
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        return x