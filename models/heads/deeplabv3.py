import torch 
import torch.nn as nn
from torch.nn import functional as F

from ..backbones import create_resnet50

class ASPPConv(nn.Module):

    def __init__(self, in_channels, out_channels, dilation):
        super(ASPPConv, self).__init__()

        self.conv = nn.Conv2d(in_channels, out_channels, 
                                kernel_size=3, padding=dilation, 
                                dilation=dilation, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU()

    def forward(self, x):
        out = self.conv(x)
        out = self.bn(out)
        out = self.relu(out)
        return out

class ASPP1x1Conv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ASPP1x1Conv, self).__init__()

        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU()
    
    def forward(self, x):
        out = self.conv(x)
        out = self.bn(out)
        out = self.relu(out)
        return out

class ASPPPooling(nn.Module):

    def __init__(self, in_channels, out_channels):
        super(ASPPPooling, self).__init__()
        self.pooling = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Conv2d(in_channels, out_channels, 
                                kernel_size=1, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU()

    def forward(self, x):
        size = x.shape[-2:]
        out = self.pooling(x)
        out = self.conv(out)
        out = self.bn(out)
        out = self.relu(out)
        return F.interpolate(out, size=size, mode='bilinear', align_corners=False)


class ASPP(nn.Module):

    def __init__(self, in_channels, out_channels):
        super(ASPP, self).__init__()

        rate1 = 6
        rate2 = 12
        rate3 = 18

        self.global_avg_pooling = ASPPPooling(in_channels, out_channels)
        self.branch1 = ASPP1x1Conv(in_channels, out_channels)
        self.branch2 = ASPPConv(in_channels, out_channels, rate1)
        self.branch3 = ASPPConv(in_channels, out_channels, rate2)
        self.branch4 = ASPPConv(in_channels, out_channels, rate3)

        self.conv_cat = nn.Sequential(
            nn.Conv2d(5 * out_channels, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU()
        )

    def forward(self, x):

        conv1x1 = self.branch1(x)
        conv3x3_6 = self.branch2(x)
        conv3x3_12 = self.branch3(x)
        conv3x3_18 = self.branch4(x)

        global_feature = self.global_avg_pooling(x)

        results = torch.cat([conv1x1, conv3x3_6, conv3x3_12, conv3x3_18, global_feature], dim=1)

        output = self.conv_cat(results)
        return output


class DeeplabV3(nn.Module):

    def __init__(self, backbone, num_class, encode_channels=2048):
        super(DeeplabV3, self).__init__()

        self.backbone = backbone
        self.aspp = ASPP(encode_channels, num_class)

    def forward(self, x):
        size = x.shape[-2:]
        output = self.backbone(x)
        output = self.aspp(output)
        output = F.interpolate(output, size=size, mode='bilinear', align_corners=False)
        return output




