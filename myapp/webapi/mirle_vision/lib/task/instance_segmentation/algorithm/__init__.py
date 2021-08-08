from enum import Enum
from typing import Tuple, List, Union
from typing import Type

from graphviz import Digraph
from torch import nn, Tensor


class Algorithm(nn.Module):

    class Name(Enum):
        MASK_RCNN = 'mask_rcnn'

    OPTIONS = [it.value for it in Name]

    @staticmethod
    def from_name(name: Name) -> Type['Algorithm']:
        if name == Algorithm.Name.MASK_RCNN:
            from .mask_rcnn import MaskRCNN as T
        else:
            raise ValueError('Invalid algorithm name')
        return T

    def __init__(self, num_classes: int, image_min_side: int, image_max_side: int):
        super().__init__()
        self.num_classes = num_classes
        self.image_min_side = image_min_side
        self.image_max_side = image_max_side

    def forward(
            self,
            padded_image_batch: Tensor,
            gt_bboxes_batch: List[Tensor] = None, padded_gt_masks_batch: List[Tensor] = None, gt_classes_batch: List[Tensor] = None
    ) -> Union[Tuple[Tensor, Tensor, Tensor, Tensor, Tensor],
               Tuple[List[Tensor], List[Tensor], List[Tensor], List[Tensor]]]:
        raise NotImplementedError

    def make_graph(self) -> Tuple[Digraph, str, str]:
        raise NotImplementedError

    def remove_output_modules(self):
        raise NotImplementedError
