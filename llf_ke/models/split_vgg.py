import math
import torch
import torch.nn as nn
from layers import conv_type
from models.builder import get_builder

# from .utils import load_state_dict_from_url
try:
    from torch.hub import load_state_dict_from_url
except ImportError:
    from torch.utils.model_zoo import load_url as load_state_dict_from_url
from typing import Union, List, Dict, Any, cast


__all__ = [
    'VGG', 'vgg11', 'vgg11_bn', 'vgg13', 'vgg13_bn', 'vgg16', 'vgg16_bn',
    'vgg19_bn', 'vgg19',
]


model_urls = {
    'vgg11': 'https://download.pytorch.org/models/vgg11-8a719046.pth',
    'vgg13': 'https://download.pytorch.org/models/vgg13-19584684.pth',
    'vgg16': 'https://download.pytorch.org/models/vgg16-397923af.pth',
    'vgg19': 'https://download.pytorch.org/models/vgg19-dcbb9e9d.pth',
    'vgg11_bn': 'https://download.pytorch.org/models/vgg11_bn-6002323d.pth',
    'vgg13_bn': 'https://download.pytorch.org/models/vgg13_bn-abd245e5.pth',
    'vgg16_bn': 'https://download.pytorch.org/models/vgg16_bn-6c64b313.pth',
    'vgg19_bn': 'https://download.pytorch.org/models/vgg19_bn-c79401a0.pth',
}


class VGG(nn.Module):

    def __init__(self, features, cfg, init_weights, builder):
        super(VGG, self).__init__()
        slim_factor = 1 #hard coded for now, needed for eval_slim
        self.features = features
        self.avgpool = nn.AdaptiveAvgPool2d((7, 7))
        self.classifier = nn.Sequential(
            builder.linear(math.ceil(512 * 7 * 7 * slim_factor), 4096, last_layer=False),
            nn.ReLU(True),
            nn.Dropout(),
            builder.linear(math.ceil(4096 * slim_factor), 4096, last_layer=False),
            nn.ReLU(True),
            nn.Dropout(),
            builder.linear(math.ceil(4096 * slim_factor), cfg.num_cls, last_layer=True)
        )
        if init_weights:
            self._initialize_weights()

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x

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

#     def _make_layer(self, builder, block, planes, blocks, stride=1, slim_factor=1):
#         downsample = None
#         if stride != 1 or self.inplanes != planes * block.expansion:
#             dconv = builder.conv1x1(math.ceil(self.inplanes * slim_factor),
#                                     math.ceil(planes * block.expansion * slim_factor), stride=stride) ## Going into a residual link
#             dbn = builder.batchnorm(math.ceil(planes * block.expansion * slim_factor))
#             if dbn is not None:
#                 downsample = nn.Sequential(dconv, dbn)
#             else:
#                 downsample = dconv

#         layers = []
#         layers.append(block(builder, self.inplanes, planes, stride, downsample, base_width=self.base_width, slim_factor=slim_factor))
#         self.inplanes = planes * block.expansion
#         for i in range(1, blocks):
#             layers.append(block(builder, self.inplanes, planes, base_width=self.base_width,
#                                 slim_factor=slim_factor))

#         return nn.Sequential(*layers)   

#         self.conv1 = builder.conv3x3(math.ceil(inplanes * slim_factor), math.ceil(planes * slim_factor), stride) ## Avoid residual links
#         self.bn1 = builder.batchnorm(math.ceil(planes * slim_factor))
#         self.relu1 = builder.activation()
#         self.relu2 = builder.activation()

#         self.conv2 = builder.conv3x3(math.ceil(planes * slim_factor),
#                                      math.ceil(planes * slim_factor))  ## Avoid residual links
#         self.bn2 = builder.batchnorm(math.ceil(planes * slim_factor), last_bn=True)  ## Avoid residual links

#         self.downsample = downsample
#         self.stride = stride


def make_layers(model_cfgs, builder, batch_norm=False):
    layers = []
    in_channels = 3
    slim_factor = 1 #hard coded for now, needed for eval_slim
    for v in model_cfgs:
        if v == 'M':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        else:
            v = cast(int, v)
            layers.append(BasicBlock(builder, in_channels, v, batch_norm, slim_factor))
#             conv2d = builder.conv3x3(math.ceil(in_channels * slim_factor), math.ceil(v * slim_factor))
#             if batch_norm:
#                 layers += [conv2d, builder.batchnorm(math.ceil(v * slim_factor)), nn.ReLU(inplace=True)]
#             else:
#                 layers += [conv2d, nn.ReLU(inplace=True)]
            in_channels = v
    return nn.Sequential(*layers)

# BasicBlock {{{
class BasicBlock(nn.Module):
    def __init__(self, builder, in_channels, v, batch_norm, slim_factor=1):
        super(BasicBlock, self).__init__()
        self.conv = builder.conv3x3(math.ceil(in_channels * slim_factor), math.ceil(v * slim_factor))
        if batch_norm:
            self.bn = builder.batchnorm(math.ceil(v * slim_factor))
        else:
            self.bn = None
        self.relu = nn.ReLU(inplace=True)
        
    def forward(self, x):
        out = self.conv(x)
        if self.bn is not None:
            out = self.bn(out)
        out = self.relu(out)
        return out

cfgs = {
    'A': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'B': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'D': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    'E': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
}


def _vgg(arch, model_cfgs, cfg, batch_norm, pretrained, progress, **kwargs):
    if pretrained:
        kwargs['init_weights'] = False
    else:
        kwargs['init_weights'] = True
    model = VGG(make_layers(model_cfgs, get_builder(cfg), batch_norm=batch_norm), cfg, builder=get_builder(cfg), **kwargs)
    if pretrained:
        state_dict = load_state_dict_from_url(model_urls[arch],
                                              progress=progress)
        model.load_state_dict(state_dict)
    return model



def vgg11(cfg, pretrained=False, progress=False, **kwargs):
    r"""VGG 11-layer model (configuration "A") from
    `"Very Deep Convolutional Networks For Large-Scale Image Recognition" <https://arxiv.org/pdf/1409.1556.pdf>`_.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _vgg('vgg11', cfgs['A'], cfg, False, pretrained, progress, **kwargs)




def vgg11_bn(cfg, pretrained=False, progress=False, **kwargs):
    r"""VGG 11-layer model (configuration "A") with batch normalization
    `"Very Deep Convolutional Networks For Large-Scale Image Recognition" <https://arxiv.org/pdf/1409.1556.pdf>`_.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _vgg('vgg11_bn', cfgs['A'], cfg, True, pretrained, progress, **kwargs)


