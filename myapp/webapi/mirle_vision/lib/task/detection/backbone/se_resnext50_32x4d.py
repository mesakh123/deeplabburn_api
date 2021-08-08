from typing import Tuple

import pretrainedmodels
from torch import nn

from . import Backbone


class SEResNeXt50_32x4d(Backbone):

    def __init__(self, pretrained: bool, num_frozen_levels: int):
        super().__init__(pretrained, num_frozen_levels)

    def _build_component(self) -> Backbone.Component:
        se_resnext50_32x4d = pretrainedmodels.__dict__['se_resnext50_32x4d'](num_classes=1000, pretrained='imagenet' if self.pretrained else None)

        # list(se_resnext50_32x4d.children()) consists of following modules
        #   [0] = Sequential(Conv2d...),
        #   [1] = Sequential(SEBottleneck...),
        #   [2] = Sequential(SEBottleneck...),
        #   [3] = Sequential(SEBottleneck...),
        #   [4] = Sequential(SEBottleneck...),
        #   [5] = AvgPool2d,
        #   [6] = Linear
        children = list(se_resnext50_32x4d.children())

        conv1 = children[0][:-1]
        conv2 = nn.Sequential(children[0][-1], *children[1])
        conv3 = children[2]
        conv4 = children[3]
        conv5 = children[4]

        num_conv1_out = 64
        num_conv2_out = 256
        num_conv3_out = 512
        num_conv4_out = 1024
        num_conv5_out = 2048

        return Backbone.Component(
            conv1, conv2, conv3, conv4, conv5,
            num_conv1_out, num_conv2_out, num_conv3_out, num_conv4_out, num_conv5_out
        )

    @staticmethod
    def normalization_means() -> Tuple[float, float, float]:
        return 0.485, 0.456, 0.406

    @staticmethod
    def normalization_stds() -> Tuple[float, float, float]:
        return 0.229, 0.224, 0.225
