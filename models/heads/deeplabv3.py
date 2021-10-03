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

        self.aspp_pooling = ASPPPooling(in_channels, out_channels)
        self.aspp1 = ASPP1x1Conv(in_channels, out_channels)
        self.aspp2 = ASPPConv(in_channels, out_channels, rate1)
        self.aspp3 = ASPPConv(in_channels, out_channels, rate2)
        self.aspp4 = ASPPConv(in_channels, out_channels, rate3)

        self.projects = nn.Sequential(
            nn.Conv2d(5 * out_channels, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU()
        )

    def forward(self, x):
        res = []
        res.append(self.aspp_pooling(x))
        res.append(self.aspp1(x))
        res.append(self.aspp2(x))
        res.append(self.aspp3(x))
        res.append(self.aspp4(x))
        results = torch.cat(res, dim=1)

        output = self.projects(results)
        return output


class DeeplabV3(nn.Module):

    def __init__(self, num_class):
        super(DeeplabV3, self).__init__()

        self.backbone = create_resnet50()
        self.aspp = ASPP(2048, num_class)

    def forward(self, x):
        output = self.backbone(x)
        output = self.aspp(output)
        return output




