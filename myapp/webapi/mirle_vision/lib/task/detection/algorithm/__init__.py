from enum import Enum
from typing import Tuple, List, Union
from typing import Type

from graphviz import Digraph
from torch import nn, Tensor

from ..backbone import Backbone


class Algorithm(nn.Module):

    class Name(Enum):
        FASTER_RCNN = 'faster_rcnn'
        FPN = 'fpn'

    OPTIONS = [it.value for it in Name]

    @staticmethod
    def from_name(name: Name) -> Type['Algorithm']:
        if name == Algorithm.Name.FASTER_RCNN:
            from .faster_rcnn import FasterRCNN as T
        elif name == Algorithm.Name.FPN:
            from .fpn import FPN as T
        else:
            raise ValueError('Invalid algorithm name')
        return T

    def __init__(self, num_classes: int,
                 backbone: Backbone,
                 anchor_ratios: List[Tuple[int, int]], anchor_sizes: List[int],
                 train_rpn_pre_nms_top_n: int, train_rpn_post_nms_top_n: int,
                 eval_rpn_pre_nms_top_n: int, eval_rpn_post_nms_top_n: int,
                 num_anchor_samples_per_batch: int, num_proposal_samples_per_batch: int, num_detections_per_image: int,
                 anchor_smooth_l1_loss_beta: float, proposal_smooth_l1_loss_beta: float,
                 proposal_nms_threshold: float, detection_nms_threshold: float):
        super().__init__()
        self.num_classes = num_classes
        self.backbone = backbone
        self.anchor_ratios = anchor_ratios
        self.anchor_sizes = anchor_sizes
        self.train_rpn_pre_nms_top_n = train_rpn_pre_nms_top_n
        self.train_rpn_post_nms_top_n = train_rpn_post_nms_top_n
        self.eval_rpn_pre_nms_top_n = eval_rpn_pre_nms_top_n
        self.eval_rpn_post_nms_top_n = eval_rpn_post_nms_top_n
        self.num_anchor_samples_per_batch = num_anchor_samples_per_batch
        self.num_proposal_samples_per_batch = num_proposal_samples_per_batch
        self.num_detections_per_image = num_detections_per_image
        self.anchor_smooth_l1_loss_beta = anchor_smooth_l1_loss_beta
        self.proposal_smooth_l1_loss_beta = proposal_smooth_l1_loss_beta
        self.proposal_nms_threshold = proposal_nms_threshold
        self.detection_nms_threshold = detection_nms_threshold

    def forward(
            self,
            padded_image_batch: Tensor,
            gt_bboxes_batch: List[Tensor] = None, gt_classes_batch: List[Tensor] = None
    ) -> Union[Tuple[Tensor, Tensor, Tensor, Tensor],
               Tuple[List[Tensor], List[Tensor], List[Tensor], List[Tensor], List[Tensor], List[Tensor]]]:
        raise NotImplementedError

    def make_graph(self) -> Tuple[Digraph, str, str]:
        raise NotImplementedError

    def remove_output_modules(self):
        raise NotImplementedError
