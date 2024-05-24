from model import GoogLeNet, VGG, ResNet
import torch.nn as nn


def get_model(model_name: str, num_classes) -> nn.Module:
    if model_name == 'googlenet':
        model = GoogLeNet.googlenet(num_classes=num_classes)
    elif model_name == 'vgg11':
        model = VGG.vgg11(num_classes=num_classes)
    elif model_name == 'vgg11_bn':
        model = VGG.vgg11_bn(num_classes=num_classes)
    elif model_name == 'vgg13':
        model = VGG.vgg13(num_classes=num_classes)
    elif model_name == 'vgg13_bn':
        model = VGG.vgg13_bn(num_classes=num_classes)
    elif model_name == 'vgg16':
        model = VGG.vgg16(num_classes=num_classes)
    elif model_name == 'vgg16_bn':
        model = VGG.vgg16_bn(num_classes=num_classes)
    elif model_name == 'vgg19':
        model = VGG.vgg19(num_classes=num_classes)
    elif model_name == 'vgg19_bn':
        model = VGG.vgg19_bn(num_classes=num_classes)
    elif model_name == 'resnet18':
        model = ResNet.resnet18(num_classes=num_classes)
    elif model_name == 'resnet34':
        model = ResNet.resnet34(num_classes=num_classes)
    elif model_name == 'resnet50':
        model = ResNet.resnet50(num_classes=num_classes)
    elif model_name == 'resnet101':
        model = ResNet.resnet101(num_classes=num_classes)
    elif model_name == 'resent152':
        model = ResNet.resnet152(num_classes=num_classes)
    else:
        raise ValueError(f'Unknown model name: {model_name}')

    return model
