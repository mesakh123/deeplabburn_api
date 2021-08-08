from enum import Enum
from typing import Tuple, Union
from typing import Type

from graphviz import Digraph
from torch import nn, Tensor


class Algorithm(nn.Module):

    class Name(Enum):
        MOBILENET_V2 = 'mobilenet_v2'
        GOOGLENET = 'googlenet'
        INCEPTION_V3 = 'inception_v3'
        RESNET18 = 'resnet18'
        RESNET34 = 'resnet34'
        RESNET50 = 'resnet50'
        RESNET101 = 'resnet101'
        EFFICIENTNET_B0 = 'efficientnet_b0'
        EFFICIENTNET_B5 = 'efficientnet_b5'
        EFFICIENTNET_B7 = 'efficientnet_b7'
        RESNEST50 = 'resnest50'
        RESNEST101 = 'resnest101'
        RESNEST200 = 'resnest200'
        RESNEST269 = 'resnest269'

    OPTIONS = [it.value for it in Name]

    @staticmethod
    def from_name(name: Name) -> Type['Algorithm']:
        if name == Algorithm.Name.MOBILENET_V2:
            from .mobilenet_v2 import MobileNet_v2 as T
        elif name == Algorithm.Name.GOOGLENET:
            from .googlenet import GoogLeNet as T
        elif name == Algorithm.Name.INCEPTION_V3:
            from .inception_v3 import Inception_v3 as T
        elif name == Algorithm.Name.RESNET18:
            from .resnet18 import ResNet18 as T
        elif name == Algorithm.Name.RESNET34:
            from .resnet34 import ResNet34 as T
        elif name == Algorithm.Name.RESNET50:
            from .resnet50 import ResNet50 as T
        elif name == Algorithm.Name.RESNET101:
            from .resnet101 import ResNet101 as T
        elif name == Algorithm.Name.EFFICIENTNET_B0:
            from .efficientnet_b0 import EfficientNet_B0 as T
        elif name == Algorithm.Name.EFFICIENTNET_B5:
            from .efficientnet_b5 import EfficientNet_B5 as T
        elif name == Algorithm.Name.EFFICIENTNET_B7:
            from .efficientnet_b7 import EfficientNet_B7 as T
        elif name == Algorithm.Name.RESNEST50:
            from .resnest50 import ResNeSt50 as T
        elif name == Algorithm.Name.RESNEST101:
            from .resnest101 import ResNeSt101 as T
        elif name == Algorithm.Name.RESNEST200:
            from .resnest200 import ResNeSt200 as T
        elif name == Algorithm.Name.RESNEST269:
            from .resnest269 import ResNeSt269 as T
        else:
            raise ValueError('Invalid algorithm name')
        return T

    def __init__(self, num_classes: int,
                 pretrained: bool, num_frozen_levels: int,
                 eval_center_crop_ratio: float):
        super().__init__()
        self.num_classes = num_classes
        self.pretrained = pretrained
        self.num_frozen_levels = num_frozen_levels
        self.eval_center_crop_ratio = eval_center_crop_ratio
        self.net = self._build_net()

    def _build_net(self) -> nn.Module:
        raise NotImplementedError

    def forward(self,
                padded_image_batch: Tensor,
                gt_classes_batch: Tensor = None) -> Union[Tensor,
                                                          Tuple[Tensor, Tensor]]:
        raise NotImplementedError

    def make_graph(self) -> Tuple[Digraph, str, str]:
        raise NotImplementedError

    def remove_output_module(self):
        raise NotImplementedError

    @property
    def output_module_weight(self) -> Tensor:
        raise NotImplementedError

    @property
    def last_features_module(self) -> nn.Module:
        raise NotImplementedError

    @staticmethod
    def normalization_means() -> Tuple[float, float, float]:
        raise NotImplementedError

    @staticmethod
    def normalization_stds() -> Tuple[float, float, float]:
        raise NotImplementedError
