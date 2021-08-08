from typing import List, Union, Tuple

import torchvision
from torch import Tensor
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor
from torchvision.models.detection.rpn import AnchorGenerator as AnchorGenerator_

from . import Algorithm


class AnchorGenerator(AnchorGenerator_):

    def cached_grid_anchors(self, grid_sizes, strides):
        key = str(grid_sizes + strides)
        # do not cache to avoid bug
        # if key in self._cache:
        #     return self._cache[key]
        anchors = self.grid_anchors(grid_sizes, strides)
        self._cache[key] = anchors
        return anchors


class MaskRCNN(Algorithm):

    def __init__(self, num_classes: int, image_min_side: int, image_max_side: int):
        super().__init__(num_classes, image_min_side, image_max_side)

        mask_rcnn = torchvision.models.detection.maskrcnn_resnet50_fpn(pretrained=True,
                                                                       min_size=image_min_side,
                                                                       max_size=image_max_side)

        mask_rcnn.rpn.anchor_generator = AnchorGenerator(sizes=mask_rcnn.rpn.anchor_generator.sizes,
                                                         aspect_ratios=mask_rcnn.rpn.anchor_generator.aspect_ratios)

        in_features = mask_rcnn.roi_heads.box_predictor.cls_score.in_features
        mask_rcnn.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

        in_features_mask = mask_rcnn.roi_heads.mask_predictor.conv5_mask.in_channels
        hidden_layer = 256
        mask_rcnn.roi_heads.mask_predictor = MaskRCNNPredictor(in_features_mask,
                                                               hidden_layer,
                                                               num_classes)

        self.net = mask_rcnn

    def forward(
            self,
            padded_image_batch: Tensor,
            gt_bboxes_batch: List[Tensor] = None, padded_gt_masks_batch: List[Tensor] = None, gt_classes_batch: List[Tensor] = None
    ) -> Union[Tuple[Tensor, Tensor, Tensor, Tensor, Tensor],
               Tuple[List[Tensor], List[Tensor], List[Tensor], List[Tensor]]]:
        if self.training:
            padded_image_batch = [it for it in padded_image_batch]
            targets = []
            for gt_bboxes, gt_classes, padded_gt_masks in zip(gt_bboxes_batch, gt_classes_batch, padded_gt_masks_batch):
                target = {
                    'boxes': gt_bboxes,
                    'labels': gt_classes,
                    'masks': padded_gt_masks
                }
                targets.append(target)

            out = self.net(padded_image_batch, targets)
            return out['loss_objectness'], out['loss_rpn_box_reg'], out['loss_classifier'], out['loss_box_reg'], out['loss_mask']
        else:
            padded_image_batch = [it for it in padded_image_batch]
            out_list = self.net(padded_image_batch)

            detection_bboxes_batch = [out['boxes'] for out in out_list]
            detection_classes_batch = [out['labels'] for out in out_list]
            detection_probs_batch = [out['scores'] for out in out_list]
            detection_masks_batch = [out['masks'] for out in out_list]

            return detection_bboxes_batch, detection_classes_batch, detection_probs_batch, detection_masks_batch

    def remove_output_modules(self):
        del self.net.rpn.head.cls_logits
        del self.net.rpn.head.bbox_pred
        del self.net.roi_heads.box_predictor.cls_score
        del self.net.roi_heads.box_predictor.bbox_pred
        del self.net.roi_heads.mask_predictor.mask_fcn_logits
