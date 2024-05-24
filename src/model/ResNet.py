from typing import Optional, Callable, Type, Union, List, Any
import torch
import torch.nn as nn
from torch import Tensor
import torch.nn.functional as F


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(
            self,
            in_planes: int,
            planes: int,
            stride: int = 1,
            downsample: Optional[nn.Module] = None,
            groups: int = 1,
            base_width: int = 64,
            dilation: int = 1,
            norm_layer: Optional[Callable[..., nn.Module]] = None
    ) -> None:
        super(BasicBlock, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if groups != 1 or base_width != 64:
            raise ValueError('BasicBlock only supports groups=1 and base_width=64')
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in BasicBlock")

        self.conv1 = conv3x3(in_planes, planes, stride)
        self.bn1 = norm_layer(planes)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = norm_layer(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x: Tensor) -> Tensor:
        identity = x

        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = F.relu(out)

        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(
            self,
            in_planes: int,
            planes: int,
            stride: int = 1,
            downsample: Optional[nn.Module] = None,
            groups: int = 1,
            base_width: int = 64,
            dilation: int = 1,
            norm_layer: Optional[Callable[..., nn.Module]] = None
    ) -> None:
        super(Bottleneck, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d

        width = int(planes * (base_width / 64.)) * groups
        self.conv1 = conv1x1(in_planes, width)
        self.bn1 = norm_layer(width)
        self.conv2 = conv3x3(width, width, stride, groups, dilation)
        self.bn2 = norm_layer(width)
        self.conv3 = conv1x1(width, planes * self.expansion)
        self.bn3 = norm_layer(planes * self.expansion)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x: Tensor) -> Tensor:
        identity = x

        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = F.relu(out)

        return out


class ResNet(nn.Module):
    def __init__(
            self,
            block: Type[Union[BasicBlock, Bottleneck]],
            layers: List[int],
            num_classes: int = 2,
            zero_init_residual: bool = False,
            groups: int = 1,
            width_per_group: int = 64,
            replace_stride_with_dilation: Optional[List[bool]] = None,
            norm_layer: Optional[Callable[..., nn.Module]] = None
    ) -> None:
        super(ResNet, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if replace_stride_with_dilation is None:
            replace_stride_with_dilation = [False, False, False]
        if len(replace_stride_with_dilation) != 3:
            raise ValueError("replace_stride_with_dilation should be None "
                             "or a 3-element tuple, got {}".format(replace_stride_with_dilation))

        self.in_planes = 64
        self.dilation = 1
        self.groups = groups
        self.base_width = width_per_group
        self._norm_layer = norm_layer

        self.conv1 = nn.Conv2d(3, self.in_planes, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = norm_layer(self.in_planes)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2,
                                       dilate=replace_stride_with_dilation[0])
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2,
                                       dilate=replace_stride_with_dilation[1])
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2,
                                       dilate=replace_stride_with_dilation[2])
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.expansion, num_classes)

        self._init_weights()
        if zero_init_residual:
            self._init_residual_with_zero()

    def _init_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _init_residual_with_zero(self) -> None:
        for m in self.modules():
            if isinstance(m, Bottleneck):
                nn.init.constant_(m.bn3.weight, 0)
            elif isinstance(m, BasicBlock):
                nn.init.constant_(m.bn2.weight, 0)

    def _make_layer(self, block: Type[Union[BasicBlock, Bottleneck]], planes: int, blocks: int,
                    stride: int = 1, dilate: bool = False) -> nn.Sequential:
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.in_planes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.in_planes, planes * block.expansion, stride),
                norm_layer(planes * block.expansion),
            )

        layers = [block(self.in_planes, planes, stride=stride, downsample=downsample, groups=self.groups,
                        base_width=self.base_width, dilation=previous_dilation, norm_layer=norm_layer)]
        self.in_planes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.in_planes, planes, groups=self.groups,
                                base_width=self.base_width, dilation=self.dilation, norm_layer=norm_layer))

        return nn.Sequential(*layers)

    def _forward_impl(self, x: Tensor) -> Tensor:
        x = self.maxpool(F.relu(self.bn1(self.conv1(x))))

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)

        return x

    def forward(self, x: Tensor) -> Tensor:
        return self._forward_impl(x)


def _resnet(
    block: Type[Union[BasicBlock, Bottleneck]],
    layers: List[int],
    **kwargs: Any
) -> ResNet:
    return ResNet(block, layers, **kwargs)


def resnet18(**kwargs: Any) -> ResNet:
    """
    ResNet-18 model

    Args:
        **kwargs: Refer to ResNet class:

            - num_classes (int): Number of classes (The number of neurons in the last FC layer).
            - zero_init_residual (bool): Whether to initialize the last normalization layer's weights to zero,
              which helps with model training.
            - groups (int): Number of groups for grouped convolution. The default value is 1,
              which means normal convolution.
            - width_per_group (int): The width of each group for grouped convolution,
              affects the number of channels in intermediate layers.
            - replace_stride_with_dilation (Optional[List[bool]]): A list of boolean values indicating whether to replace
              stride convolution with dilated convolution in certain layers to maintain feature map resolution.
            - norm_layer (Optional[Callable[..., nn.Module]]): The normalization layer function to use, default is
              nn.BatchNorm2d. Other normalization layers can be passed.
    """
    return _resnet(BasicBlock, [2, 2, 2, 2], **kwargs)


def resnet34(**kwargs: Any):
    """
    ResNet-34 model

    Args:
        **kwargs: Refer to ResNet class:

            - num_classes (int): Number of classes (The number of neurons in the last FC layer).
            - zero_init_residual (bool): Whether to initialize the last normalization layer's weights to zero,
              which helps with model training.
            - groups (int): Number of groups for grouped convolution. The default value is 1,
              which means normal convolution.
            - width_per_group (int): The width of each group for grouped convolution,
              affects the number of channels in intermediate layers.
            - replace_stride_with_dilation (Optional[List[bool]]): A list of boolean values indicating whether to replace
              stride convolution with dilated convolution in certain layers to maintain feature map resolution.
            - norm_layer (Optional[Callable[..., nn.Module]]): The normalization layer function to use, default is
              nn.BatchNorm2d. Other normalization layers can be passed.
    """
    return _resnet(BasicBlock, [3, 4, 6, 3], **kwargs)


def resnet50(**kwargs: Any):
    """
    ResNet-50 model

    Args:
        **kwargs: Refer to ResNet class:

            - num_classes (int): Number of classes (The number of neurons in the last FC layer).
            - zero_init_residual (bool): Whether to initialize the last normalization layer's weights to zero,
              which helps with model training.
            - groups (int): Number of groups for grouped convolution. The default value is 1,
              which means normal convolution.
            - width_per_group (int): The width of each group for grouped convolution,
              affects the number of channels in intermediate layers.
            - replace_stride_with_dilation (Optional[List[bool]]): A list of boolean values indicating whether to replace
              stride convolution with dilated convolution in certain layers to maintain feature map resolution.
            - norm_layer (Optional[Callable[..., nn.Module]]): The normalization layer function to use, default is
              nn.BatchNorm2d. Other normalization layers can be passed.
    """
    return _resnet(Bottleneck, [3, 4, 6, 3], **kwargs)


def resnet101(**kwargs: Any):
    """
    ResNet-101 model

    Args:
        **kwargs: Refer to ResNet class:

            - num_classes (int): Number of classes (The number of neurons in the last FC layer).
            - zero_init_residual (bool): Whether to initialize the last normalization layer's weights to zero,
              which helps with model training.
            - groups (int): Number of groups for grouped convolution. The default value is 1,
              which means normal convolution.
            - width_per_group (int): The width of each group for grouped convolution,
              affects the number of channels in intermediate layers.
            - replace_stride_with_dilation (Optional[List[bool]]): A list of boolean values indicating whether to replace
              stride convolution with dilated convolution in certain layers to maintain feature map resolution.
            - norm_layer (Optional[Callable[..., nn.Module]]): The normalization layer function to use, default is
              nn.BatchNorm2d. Other normalization layers can be passed.
    """
    return _resnet(Bottleneck, [3, 4, 23, 3], **kwargs)


def resnet152(**kwargs: Any):
    """
    ResNet-152 model

    Args:
        **kwargs: Refer to ResNet class:

            - num_classes (int): Number of classes (The number of neurons in the last FC layer).
            - zero_init_residual (bool): Whether to initialize the last normalization layer's weights to zero,
              which helps with model training.
            - groups (int): Number of groups for grouped convolution. The default value is 1,
              which means normal convolution.
            - width_per_group (int): The width of each group for grouped convolution,
              affects the number of channels in intermediate layers.
            - replace_stride_with_dilation (Optional[List[bool]]): A list of boolean values indicating whether to replace
              stride convolution with dilated convolution in certain layers to maintain feature map resolution.
            - norm_layer (Optional[Callable[..., nn.Module]]): The normalization layer function to use, default is
              nn.BatchNorm2d. Other normalization layers can be passed.
    """
    return _resnet(Bottleneck, [3, 8, 36, 3], **kwargs)


def conv3x3(in_planes: int, out_planes: int, stride: int = 1, groups: int = 1, dilation: int = 1) -> nn.Conv2d:
    """
    3x3 convolution with padding
    Args:
        in_planes (int): Number of input channels (depth of input feature map)
        out_planes (int): Number of output channels (the depth of the output feature map)
        stride (int): Step size of the convolution kernel sliding on the input feature map. The default value is 1.
        groups (int): Number of groups for grouped convolution. The default value is 1, which means normal convolution.
        dilation (int): The dilation rate of atrous convolution. The default value is 1, which means normal convolution.
    """
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=dilation, groups=groups, bias=False, dilation=dilation)


def conv1x1(in_planes: int, out_planes: int, stride: int = 1) -> nn.Conv2d:
    """
    1x1 convolution
    Args:
        in_planes (int): Number of input channels (depth of input feature map)
        out_planes (int): Number of output channels (the depth of the output feature map)
        stride (int): Step size of the convolution kernel sliding on the input feature map. The default value is 1.
    """
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)
