from typing import Tuple

import torchvision
from torch import nn

from . import Backbone


class ResNet101(Backbone):

    def __init__(self, pretrained: bool, num_frozen_levels: int):
        super().__init__(pretrained, num_frozen_levels)

    def _build_component(self) -> Backbone.Component:
        resnet101 = torchvision.models.resnet101(pretrained=self.pretrained)

        # list(resnet101.children()) consists of following modules
        #   [0] = Conv2d, [1] = BatchNorm2d, [2] = ReLU,
        #   [3] = MaxPool2d, [4] = Sequential(Bottleneck...),
        #   [5] = Sequential(Bottleneck...),
        #   [6] = Sequential(Bottleneck...),
        #   [7] = Sequential(Bottleneck...),
        #   [8] = AvgPool2d, [9] = Linear
        children = list(resnet101.children())

        conv1 = nn.Sequential(*children[:3])
        conv2 = nn.Sequential(*children[3:5])
        conv3 = children[5]
        conv4 = children[6]
        conv5 = children[7]

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
