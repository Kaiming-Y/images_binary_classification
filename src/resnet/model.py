import torch
from torch import nn


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_channels, out_channels, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=out_channels
                               , kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.downsample = downsample

    def forward(self, x):
        identity = x
        if self.downsample is not None:
            identity = self.downsample(identity)
        conv1_out = self.relu(self.bn1(self.conv1(x)))
        conv2_out = self.bn2(self.conv2(conv1_out))
        out_add = conv2_out + identity

        out = self.relu(out_add)

        return out


class BasicBlock2(nn.Module):
    expansion = 4 

    def __init__(self, in_channels, out_channels, stride=1, downsample=None):
        super(BasicBlock2, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=1,
                               stride=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU()

        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3,
                               stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)

        self.conv3 = nn.Conv2d(out_channels, out_channels, kernel_size=1,
                               stride=1, padding=1, bias=False)
        self.bn3 = nn.BatchNorm2d(out_channels * self.expansion)
        self.downsample = downsample

    def forward(self, x):
        identity = x
        if self.downsample is not None:
            identity = self.downsample(identity)

        conv1_out = self.relu(self.bn1(self.conv1(x)))
        conv2_out = self.relu(self.bn2(self.conv2(conv1_out)))
        conv3_out = self.bn3(self.conv3(conv2_out))

        out_add = conv3_out + identity

        out = self.relu(out_add)

        return out


class ResNet(nn.Module):
    def __init__(self, block, block_num, num_classes=1000, include_top=True):
        super(ResNet, self).__init__()
        self.include_top = include_top
        self.in_channels = 64

        self.conv1 = nn.Conv2d(3, out_channels=self.in_channels, kernel_size=7, stride=2,
                               padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(self.in_channels)
        self.relu = nn.ReLU()
        self.max_pool = nn.MaxPool2d(3, stride=2, padding=1)

        self.layer1 = self._make_layer(block, 64, block_num[0])
        self.layer2 = self._make_layer(block, 128, block_num[1], stride=2)
        self.layer3 = self._make_layer(block, 256, block_num[2], stride=2)
        self.layer4 = self._make_layer(block, 512, block_num[3], stride=2)

        if self.include_top:
            self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
            self.fc = nn.Linear(512 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')

    def forward(self, x):
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.max_pool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        if self.include_top:
            x = self.avgpool(x)
            x = torch.flatten(x, 1)
            x = self.fc(x)

        return x

    def _make_layer(self, block, channels, block_num, stride=1):
        downsample = None
        if stride != 1 or self.in_channels != channels * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.in_channels, channels * block.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(channels * block.expansion)
            )
        layers = []
        layers.append(block(self.in_channels, channels, downsample=downsample, stride=stride))
        self.in_channels = channels * block.expansion

        for _ in range(1, block_num):
            layers.append(block(self.in_channels, channels))

        return nn.Sequential(*layers)


def resnet18(num_classes=1000, include_top=True):
    # Pretrained weight link 'https://download.pytorch.org/models/resnet18-5c106cde.pth',
    return ResNet(BasicBlock, [2, 2, 2, 2], num_classes=num_classes, include_top=include_top)


def resnet34(num_classes=1000, include_top=True):
    # Pretrained weight link https://download.pytorch.org/models/resnet34-333f7ec4.pth
    return ResNet(BasicBlock, [3, 4, 6, 3], num_classes=num_classes, include_top=include_top)


def resnet101(num_classes=1000, include_top=True):
    # Pretrained weight link https://download.pytorch.org/models/resnet50-19c8e357.pth
    return ResNet(BasicBlock2, [3, 4, 23, 3], num_classes=num_classes, include_top=include_top)


if __name__ == "__main__":
    net = resnet34()
    test = torch.rand((1, 3, 214, 214))
    out = net(test)
    print(out.shape)
