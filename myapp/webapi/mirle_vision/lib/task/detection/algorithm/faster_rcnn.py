from typing import Union, Tuple, List

import torch
from torch import nn, Tensor
from torch.nn import functional as F
from torchvision.ops import RoIAlign

from . import Algorithm
from ..backbone import Backbone
from ..head.roi import ROI
from ..head.rpn import RPN


class FasterRCNN(Algorithm):

    def __init__(self, num_classes: int,
                 backbone: Backbone,
                 anchor_ratios: List[Tuple[int, int]], anchor_sizes: List[int],
                 train_rpn_pre_nms_top_n: int, train_rpn_post_nms_top_n: int,
                 eval_rpn_pre_nms_top_n: int, eval_rpn_post_nms_top_n: int,
                 num_anchor_samples_per_batch: int, num_proposal_samples_per_batch: int, num_detections_per_image: int,
                 anchor_smooth_l1_loss_beta: float, proposal_smooth_l1_loss_beta: float,
                 proposal_nms_threshold: float, detection_nms_threshold: float):
        super().__init__(num_classes,
                         backbone,
                         anchor_ratios, anchor_sizes,
                         train_rpn_pre_nms_top_n, train_rpn_post_nms_top_n,
                         eval_rpn_pre_nms_top_n, eval_rpn_post_nms_top_n,
                         num_anchor_samples_per_batch, num_proposal_samples_per_batch, num_detections_per_image,
                         anchor_smooth_l1_loss_beta, proposal_smooth_l1_loss_beta,
                         proposal_nms_threshold, detection_nms_threshold)

        self.body, num_body_out = self._build_body()
        self.rpn_head, num_rpn_extractor_out = self._build_rpn_head(num_extractor_in=num_body_out)
        self.roi_head = self._build_roi_head()

        self._roi_align = RoIAlign(output_size=(14, 14), spatial_scale=1/16, sampling_ratio=0)

    def _build_body(self) -> Tuple[nn.Module, int]:
        body = nn.Sequential(self.backbone.component.conv1, self.backbone.component.conv2,
                             self.backbone.component.conv3, self.backbone.component.conv4)
        num_body_out = self.backbone.component.num_conv4_out
        return body, num_body_out

    def _build_rpn_head(self, num_extractor_in: int) -> Tuple[RPN, int]:
        num_extractor_out = 512
        extractor = nn.Sequential(
            nn.Conv2d(in_channels=num_extractor_in, out_channels=num_extractor_out, kernel_size=3, padding=1),
            nn.ReLU()
        )
        head = RPN(extractor, num_extractor_out,
                   self.anchor_ratios, self.anchor_sizes,
                   self.train_rpn_pre_nms_top_n, self.train_rpn_post_nms_top_n,
                   self.eval_rpn_pre_nms_top_n, self.eval_rpn_post_nms_top_n,
                   self.num_anchor_samples_per_batch, self.anchor_smooth_l1_loss_beta,
                   self.proposal_nms_threshold)
        return head, num_extractor_out

    def _build_roi_head(self) -> ROI:
        num_extractor_out = self.backbone.component.num_conv5_out
        extractor = self.backbone.component.conv5
        head = ROI(extractor, num_extractor_out,
                   self.num_classes,
                   self.num_proposal_samples_per_batch, self.num_detections_per_image,
                   self.proposal_smooth_l1_loss_beta,
                   self.detection_nms_threshold)
        return head

    def forward(
            self, padded_image_batch: Tensor,
            gt_bboxes_batch: List[Tensor] = None, gt_classes_batch: List[Tensor] = None
    ) -> Union[Tuple[Tensor, Tensor, Tensor, Tensor],
               Tuple[List[Tensor], List[Tensor], List[Tensor], List[Tensor], List[Tensor], List[Tensor]]]:
        batch_size, _, padded_image_height, padded_image_width = padded_image_batch.shape

        features_batch = self.body(padded_image_batch)
        _, _, features_height, features_width = features_batch.shape

        anchor_objectnesses_batch, anchor_transformers_batch = self.rpn_head.forward(features_batch)

        anchor_bboxes = \
            self.rpn_head.generate_anchors(
                padded_image_width, padded_image_height,
                num_x_anchors=features_width, num_y_anchors=features_height,
                scale=1.0
            ).to(features_batch)
        anchor_bboxes_batch: List[Tensor] = [anchor_bboxes] * batch_size

        proposal_bboxes_batch, proposal_probs_batch = \
            self.rpn_head.generate_proposals_batch(
                anchor_bboxes_batch,
                anchor_objectnesses_batch, anchor_transformers_batch,
                padded_image_width, padded_image_height
            )

        if self.training:
            # manually generate a background bbox if no any ground-truth was provided
            for b in range(batch_size):
                gt_bboxes = gt_bboxes_batch[b]
                gt_classes = gt_classes_batch[b]
                if gt_bboxes.shape[0] == 0 and gt_classes.shape[0] == 0:
                    gt_bboxes_batch[b] = torch.tensor([
                        [padded_image_width // 4,
                         padded_image_height // 4,
                         padded_image_width // 4 + padded_image_width // 2,
                         padded_image_height // 4 + padded_image_height // 2]
                    ]).to(gt_bboxes)
                    gt_classes_batch[b] = torch.tensor([0]).to(gt_classes)

            anchor_objectness_loss_batch = torch.zeros((batch_size,)).to(features_batch)
            anchor_transformer_loss_batch = torch.zeros((batch_size,)).to(features_batch)

            for b in range(batch_size):
                gt_bboxes = gt_bboxes_batch[b]
                gt_classes = gt_classes_batch[b]
                anchor_bboxes = anchor_bboxes_batch[b]
                anchor_objectnesses = anchor_objectnesses_batch[b]
                anchor_transformers = anchor_transformers_batch[b]

                sampled_indices, sampled_gt_anchor_objectnesses, sampled_gt_anchor_transformers = \
                    self.rpn_head.sample(anchor_bboxes, gt_bboxes, gt_classes,
                                         padded_image_width, padded_image_height)

                if sampled_indices.shape[0] == 0:
                    continue

                sampled_anchor_objectnesses = anchor_objectnesses[sampled_indices]
                sampled_anchor_transformers = anchor_transformers[sampled_indices]

                anchor_objectness_loss, anchor_transformer_loss = \
                    self.rpn_head.loss(sampled_anchor_objectnesses, sampled_anchor_transformers,
                                       sampled_gt_anchor_objectnesses, sampled_gt_anchor_transformers)

                anchor_objectness_loss_batch[b] = anchor_objectness_loss
                anchor_transformer_loss_batch[b] = anchor_transformer_loss

            proposal_class_loss_batch = torch.zeros((batch_size,)).to(features_batch)
            proposal_transformer_loss_batch = torch.zeros((batch_size,)).to(features_batch)

            for b in range(batch_size):
                features = features_batch[b]
                gt_bboxes = gt_bboxes_batch[b]
                gt_classes = gt_classes_batch[b]
                proposal_bboxes = proposal_bboxes_batch[b]

                sampled_indices, sampled_gt_proposal_classes, sampled_gt_proposal_transformers = \
                    self.roi_head.sample(proposal_bboxes,
                                         gt_bboxes, gt_classes)
                sampled_proposal_bboxes = proposal_bboxes[sampled_indices]

                if sampled_proposal_bboxes.shape[0] == 0:
                    continue

                pools = self._roi_align(input=features.unsqueeze(dim=0),
                                        rois=[sampled_proposal_bboxes])
                pools = F.max_pool2d(input=pools, kernel_size=2, stride=2)

                proposal_classes, proposal_transformers = \
                    self.roi_head.forward(pools,
                                          post_extract_transform=lambda x: F.adaptive_avg_pool2d(input=x, output_size=1).view(x.shape[0], -1))

                proposal_class_loss, proposal_transformer_loss = \
                    self.roi_head.loss(proposal_classes, proposal_transformers,
                                       sampled_gt_proposal_classes, sampled_gt_proposal_transformers)

                proposal_class_loss_batch[b] = proposal_class_loss
                proposal_transformer_loss_batch[b] = proposal_transformer_loss

            return (anchor_objectness_loss_batch, anchor_transformer_loss_batch,
                    proposal_class_loss_batch, proposal_transformer_loss_batch)
        else:
            pools_batch = []
            for b in range(batch_size):
                features = features_batch[b]
                proposal_bboxes = proposal_bboxes_batch[b]

                pools = self._roi_align(input=features.unsqueeze(dim=0),
                                        rois=[proposal_bboxes])
                pools = F.max_pool2d(input=pools, kernel_size=2, stride=2)
                pools_batch.append(pools)

            detection_bboxes_batch, detection_classes_batch, detection_probs_batch = [], [], []
            for pools in pools_batch:
                proposal_classes, proposal_transformers = \
                    self.roi_head.forward(pools,
                                          post_extract_transform=lambda x: F.adaptive_avg_pool2d(input=x, output_size=1).view(x.shape[0], -1))

                detection_bboxes, detection_classes, detection_probs = \
                    self.roi_head.generate_detections(proposal_bboxes,
                                                      proposal_classes, proposal_transformers,
                                                      padded_image_width, padded_image_height)

                detection_bboxes_batch.append(detection_bboxes)
                detection_classes_batch.append(detection_classes)
                detection_probs_batch.append(detection_probs)

            return (anchor_bboxes_batch, proposal_bboxes_batch, proposal_probs_batch,
                    detection_bboxes_batch, detection_classes_batch, detection_probs_batch)

    def remove_output_modules(self):
        del self.rpn_head.anchor_objectness
        del self.rpn_head.anchor_transformer
        del self.roi_head.proposal_class
        del self.roi_head.proposal_transformer
