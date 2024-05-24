from typing import List, Union, cast, Dict, Any
import torch
import torch.nn as nn
from torch import Tensor


class VGG(nn.Module):
    def __init__(
            self,
            features: nn.Module,
            num_classes: int = 2,
            init_weights: bool = True
    ) -> None:
        super(VGG, self).__init__()
        self.features = features
        self.avgpool = nn.AdaptiveAvgPool2d((7, 7))
        self.classifier = nn.Sequential(
            nn.Linear(512 * 7 * 7, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, num_classes),
        )
        if init_weights:
            self._initialize_weights()

    def _initialize_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)

    def forward(self, x: Tensor) -> Tensor:
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x


def make_layers(cfg: List[Union[str, int]], batch_norm: bool = False) -> nn.Sequential:
    layers: List[nn.Module] = []
    in_channels = 3
    for v in cfg:
        if v == 'M':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        else:
            v = cast(int, v)
            conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
            if batch_norm:
                layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
            else:
                layers += [conv2d, nn.ReLU(inplace=True)]
            in_channels = v
    return nn.Sequential(*layers)


cfgs: Dict[str, List[Union[str, int]]] = {
    'A': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'B': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'D': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    'E': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
}


def _vgg(cfg: str, batch_norm: bool, **kwargs: Any) -> VGG:
    return VGG(make_layers(cfgs[cfg], batch_norm=batch_norm), **kwargs)


def vgg11(**kwargs: Any) -> VGG:
    """
    VGG 11-layer model

    Args:
        **kwargs: refer to VGG class

            - num_classes (int): number of classes (The number of neurons in the last FC layer)
            - init_weights (bool): whether to initialize the weights in each layer
    """
    return _vgg('A', False, **kwargs)


def vgg11_bn(**kwargs: Any) -> VGG:
    """
    VGG 11-layer model with bath normalization

    Args:
        **kwargs: refer to VGG class

            - num_classes (int): number of classes (The number of neurons in the last FC layer)
            - init_weights (bool): whether to initialize the weights in each layer
    """
    return _vgg('A', True, **kwargs)


def vgg13(**kwargs: Any) -> VGG:
    """
    VGG 13-layer model

    Args:
        **kwargs: refer to VGG class

            - num_classes (int): number of classes (The number of neurons in the last FC layer)
            - init_weights (bool): whether to initialize the weights in each layer
    """
    return _vgg('B', False, **kwargs)


def vgg13_bn(**kwargs: Any) -> VGG:
    """
    VGG 13-layer model with bath normalization

    Args:
        **kwargs: refer to VGG class

            - num_classes (int): number of classes (The number of neurons in the last FC layer)
            - init_weights (bool): whether to initialize the weights in each layer
    """
    return _vgg('B', True, **kwargs)


def vgg16(**kwargs: Any) -> VGG:
    """
    VGG 16-layer model

    Args:
        **kwargs: refer to VGG class

            - num_classes (int): number of classes (The number of neurons in the last FC layer)
            - init_weights (bool): whether to initialize the weights in each layer
    """
    return _vgg('D', False, **kwargs)


def vgg16_bn(**kwargs: Any) -> VGG:
    """
    VGG 16-layer model with bath normalization

    Args:
        **kwargs: refer to VGG class

            - num_classes (int): number of classes (The number of neurons in the last FC layer)
            - init_weights (bool): whether to initialize the weights in each layer
    """
    return _vgg('D', True, **kwargs)


def vgg19(**kwargs: Any) -> VGG:
    """
    VGG 19-layer model

    Args:
        **kwargs: refer to VGG class

            - num_classes (int): number of classes (The number of neurons in the last FC layer)
            - init_weights (bool): whether to initialize the weights in each layer
    """
    return _vgg('E', False, **kwargs)


def vgg19_bn(**kwargs: Any) -> VGG:
    """
    VGG 19-layer model with bath normalization

    Args:
        **kwargs: refer to VGG class

            - num_classes (int): number of classes (The number of neurons in the last FC layer)
            - init_weights (bool): whether to initialize the weights in each layer
    """
    return _vgg('E', True, **kwargs)
