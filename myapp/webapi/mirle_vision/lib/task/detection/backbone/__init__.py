from dataclasses import dataclass
from enum import Enum
from typing import Tuple, Type

from torch import nn


class Backbone:

    class Name(Enum):
        RESNET18 = 'resnet18'
        RESNET34 = 'resnet34'
        RESNET50 = 'resnet50'
        RESNET101 = 'resnet101'
        RESNET152 = 'resnet152'
        RESNEXT50_32X4D = 'resnext50_32x4d'
        RESNEXT101_32X8D = 'resnext101_32x8d'
        WIDE_RESNET50_2 = 'wide_resnet50_2'
        WIDE_RESNET101_2 = 'wide_resnet101_2'
        SENET154 = 'senet154'
        SE_RESNEXT50_32X4D = 'se_resnext50_32x4d'
        SE_RESNEXT101_32X4D = 'se_resnext101_32x4d'
        NASNET_A_LARGE = 'nasnet_a_large'
        PNASNET_5_LARGE = 'pnasnet_5_large'
        RESNEST50 = 'resnest50'
        RESNEST101 = 'resnest101'
        RESNEST200 = 'resnest200'
        RESNEST269 = 'resnest269'

    OPTIONS = [it.value for it in Name]

    @staticmethod
    def from_name(name: Name) -> Type['Backbone']:
        if name == Backbone.Name.RESNET18:
            from .resnet18 import ResNet18 as T
        elif name == Backbone.Name.RESNET34:
            from .resnet34 import ResNet34 as T
        elif name == Backbone.Name.RESNET50:
            from .resnet50 import ResNet50 as T
        elif name == Backbone.Name.RESNET101:
            from .resnet101 import ResNet101 as T
        elif name == Backbone.Name.RESNET152:
            from .resnet152 import ResNet152 as T
        elif name == Backbone.Name.RESNEXT50_32X4D:
            from .resnext50_32x4d import ResNeXt50_32x4d as T
        elif name == Backbone.Name.RESNEXT101_32X8D:
            from .resnext101_32x8d import ResNeXt101_32x8d as T
        elif name == Backbone.Name.WIDE_RESNET50_2:
            from .wide_resnet50_2 import WideResNet50_2 as T
        elif name == Backbone.Name.WIDE_RESNET101_2:
            from .wide_resnet101_2 import WideResNet101_2 as T
        elif name == Backbone.Name.SENET154:
            from .senet154 import SENet154 as T
        elif name == Backbone.Name.SE_RESNEXT50_32X4D:
            from .se_resnext50_32x4d import SEResNeXt50_32x4d as T
        elif name == Backbone.Name.SE_RESNEXT101_32X4D:
            from .se_resnext101_32x4d import SEResNeXt101_32x4d as T
        elif name == Backbone.Name.NASNET_A_LARGE:
            from .nasnet_a_large import NASNet_A_Large as T
        elif name == Backbone.Name.PNASNET_5_LARGE:
            from .pnasnet_5_large import PNASNet_5_Large as T
        elif name == Backbone.Name.RESNEST50:
            from .resnest50 import ResNeSt50 as T
        elif name == Backbone.Name.RESNEST101:
            from .resnest101 import ResNeSt101 as T
        elif name == Backbone.Name.RESNEST200:
            from .resnest200 import ResNeSt200 as T
        elif name == Backbone.Name.RESNEST269:
            from .resnest269 import ResNeSt269 as T
        else:
            raise ValueError('Invalid backbone name')
        return T

    def __init__(self, pretrained: bool, num_frozen_levels: int):
        super().__init__()
        self.pretrained = pretrained
        self.num_frozen_levels = num_frozen_levels
        self.component = self._build_component()
        self._freeze_layers()

    @dataclass
    class Component:
        conv1: nn.Module
        conv2: nn.Module
        conv3: nn.Module
        conv4: nn.Module
        conv5: nn.Module
        num_conv1_out: int
        num_conv2_out: int
        num_conv3_out: int
        num_conv4_out: int
        num_conv5_out: int

    def _build_component(self) -> Component:
        raise NotImplementedError

    def _freeze_layers(self):
        modules = [self.component.conv1,
                   self.component.conv2,
                   self.component.conv3,
                   self.component.conv4,
                   self.component.conv5]

        assert 0 <= self.num_frozen_levels <= len(modules)
        freezing_modules = modules[:self.num_frozen_levels]

        for module in freezing_modules:
            module.eval()
            for parameter in module.parameters():
                parameter.requires_grad = False

    @staticmethod
    def normalization_means() -> Tuple[float, float, float]:
        raise NotImplementedError

    @staticmethod
    def normalization_stds() -> Tuple[float, float, float]:
        raise NotImplementedError
