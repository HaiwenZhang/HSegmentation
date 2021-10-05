import torch
import torch.nn as nn
try:
    from torch.hub import load_state_dict_from_url
except ImportError:
    from torch.utils.model_zoo import load_url as load_state_dict_from_url

resnet50_model_url = "https://download.pytorch.org/models/resnet50-0676ba61.pth"

def conv3x3(in_channels, out_channels, stride=1, dilation=1):
    return nn.Conv2d(in_channels, out_channels, 
            kernel_size=3, stride=stride, padding=dilation, dilation=dilation, bias=False)
    

def conv1x1(in_channels, out_channels, stride = 1):
    return nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False)


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_channels, out_channels, stride=1, dilation=1, downsample=None):
        super(Bottleneck, self).__init__()

        self.in_channels = in_channels

        self.conv1 = conv1x1(in_channels, out_channels)
        self.bn1 = nn.BatchNorm2d(out_channels)

        self.conv2 = conv3x3(out_channels, out_channels, stride, dilation)
        self.bn2 = nn.BatchNorm2d(out_channels)

        self.conv3 = conv1x1(out_channels, out_channels * self.expansion)
        self.bn3 = nn.BatchNorm2d(out_channels * self.expansion)
        self.relu = nn.ReLU()
        self.downsample = downsample

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)
        out = self.relu(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class ResNet50(nn.Module):

    def __init__(self, num_class=1000):
        super(ResNet50, self).__init__()

        self.in_channels = 64
        self.layers = [3, 4, 6, 3]
        self.dilation = 1

        self.conv1 = nn.Conv2d(3, self.in_channels, kernel_size=7, stride=2, padding=3, 
                                bias=False)
        self.bn1 = nn.BatchNorm2d(self.in_channels)
        self.relu = nn.ReLU()
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.layer1 = self._make_layer(64, self.layers[0])
        self.layer2 = self._make_layer(128, self.layers[1], stride=2)
        self.layer3 = self._make_layer(256, self.layers[2], stride=2)
        self.layer4 = self._make_layer(512, self.layers[3], stride=2)
                                       
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(2048, num_class)


    def forward(self, x):
        x = self.conv1(x)

        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)

        return x


    def _make_layer(self, channels, blocks, stride=1):
        downsample = nn.Sequential(
            conv1x1(self.in_channels, channels * Bottleneck.expansion, stride),
            nn.BatchNorm2d(channels * Bottleneck.expansion),
        )

        layers = []
        layers.append(Bottleneck(self.in_channels, channels, stride, self.dilation, downsample))
        self.in_channels = channels * Bottleneck.expansion
        for _ in range(1, blocks):
            layers.append(Bottleneck(self.in_channels, channels))

        return nn.Sequential(*layers)

def create_resnet50(pretrained=True):
    model = ResNet50()
    if pretrained:
        pretrained_dict = load_state_dict_from_url(resnet50_model_url, progress=True)
        model.load_state_dict(pretrained_dict)
    return model



if __name__ == "__main__":
    model = create_resnet50()

    print(model)

    img = torch.rand((16, 3, 256, 256))
    o = model(img)
    print(o.shape)


