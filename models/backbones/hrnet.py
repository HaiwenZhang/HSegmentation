
import torch
import torch.nn as nn
from torch.nn.modules.batchnorm import BatchNorm2d


"""
Implement HRNet18

State 1
| 1x1,64  |
| 3x3,64  | x4 x1
| 1x1,256 |

State 2
| 3x3,2C | x4 x1
| 3x3,2C |

State 3
| 3x3,2C | x4 x4
| 3x3,2C |

State 4
| 3x3,2C | x4 x3
| 3x3,2C |
"""


_BN_MOMENTUM = 0.1
_RELU_INPLACE = True

def conv3x3(in_channels, out_channels, stride=1, padding=1):
    """ 3x3 convolution with padding"""
    return nn.Conv2d(in_channels, out_channels, 
                        kernel_size=3, stride=stride, 
                        padding=padding, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_channels, channels, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        out_channels = self.expansion * channels
        self.conv1 = conv3x3(in_channels, out_channels, stride=stride)
        self.bn1 = nn.BatchNorm2d(out_channels, momentum=_BN_MOMENTUM)
        self.act1 = nn.ReLU(inplace=_RELU_INPLACE)
        self.conv2 = conv3x3(out_channels, out_channels)
        self.bn2 = nn.BatchNorm2d(out_channels, momentum=_BN_MOMENTUM)
        self.act2 = nn.ReLU(inplace=_RELU_INPLACE)
        self.downsample = downsample

    def forward(self, x):
        residual = x
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.act1(x)

        x = self.conv2(x)
        x = self.bn2(x)

        if self.downsample is not None:
            residual = self.downsample(x)
        
        x = x + residual
        x = self.act2(x)

        return x





class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_channels, channels, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        out_channels = self.expansion * channels
        self.conv1 = nn.Conv2d()



class HRNet18(nn.Module):

    def __init__(self, cfg):
        super(HRNet18, self).__init__()

        self.conv1 = conv3x3(3, 64, stride=2, padding=1)
        self.bn1 = nn.BatchNorm2d(64, momentum=_BN_MOMENTUM)
        self.act1 = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(64, 64, stride=2, padding=1)
        self.bn2 = nn.BatchNorm2d(64, momentum=_BN_MOMENTUM)
        self.act2 = nn.ReLU(inplace=True)

        self.layer1 = None
        self.stage2 = None
        self.state3 = None
        self.state4 = None


        self.last_layer = nn.Sequential(
            nn.Conv2d(),
            nn.BatchNorm2d(),
            nn.ReLU(inplace=)
        )


    def stages(self, x):

        x = self.layer1(x)

        xl = 


    def forward(self, x):

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.act1(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.act2(x)

        #Stages
