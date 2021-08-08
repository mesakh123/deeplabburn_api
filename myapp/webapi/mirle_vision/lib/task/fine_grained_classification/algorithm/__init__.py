from enum import Enum
from typing import Tuple, Union
from typing import Type

from graphviz import Digraph
from torch import nn, Tensor


class Algorithm(nn.Module):

    class Name(Enum):
        RACNN = 'racnn'

    OPTIONS = [it.value for it in Name]

    @staticmethod
    def from_name(name: Name) -> Type['Algorithm']:
        if name == Algorithm.Name.RACNN:
            from .racnn import RACNN as T
        else:
            raise ValueError('Invalid algorithm name')
        return T

    def __init__(self, num_classes: int,
                 pretrained: bool, num_frozen_levels: int,
                 roi_resized_width: int, roi_resized_height: int):
        super().__init__()
        self.num_classes = num_classes
        self.pretrained = pretrained
        self.num_frozen_levels = num_frozen_levels
        self.roi_resized_width = roi_resized_width
        self.roi_resized_height = roi_resized_height

    def forward(self,
                padded_image_batch: Tensor,
                gt_classes_batch: Tensor = None) -> Union[Tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor],
                                                          Tuple[Tensor, Tensor, Tensor, Tensor]]:
        raise NotImplementedError

    def make_graph(self) -> Tuple[Digraph, str, str]:
        raise NotImplementedError

    def remove_output_module(self):
        raise NotImplementedError

    @staticmethod
    def normalization_means() -> Tuple[float, float, float]:
        raise NotImplementedError

    @staticmethod
    def normalization_stds() -> Tuple[float, float, float]:
        raise NotImplementedError
