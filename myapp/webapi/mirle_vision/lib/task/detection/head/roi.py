from typing import Tuple, Callable

import torch
from torch import nn, Tensor
from torch.nn import functional as F
from torchvision.ops.boxes import nms, box_iou

from ....bbox import BBox
from ....extension.functional import beta_smooth_l1_loss


class ROI(nn.Module):

    def __init__(self, extractor: nn.Module, num_extractor_out: int, num_classes: int,
                 num_proposal_samples_per_batch: int, num_detections_per_image: int,
                 proposal_smooth_l1_loss_beta: float, detection_nms_threshold: float):
        super().__init__()
        self._extractor = extractor
        self._num_classes = num_classes

        self.proposal_class = nn.Linear(num_extractor_out, num_classes)
        self.proposal_transformer = nn.Linear(num_extractor_out, num_classes * 4)

        self._num_proposal_samples_per_batch = num_proposal_samples_per_batch
        self._num_detections_per_image = num_detections_per_image

        self._proposal_smooth_l1_loss_beta = proposal_smooth_l1_loss_beta
        self._detection_nms_threshold = detection_nms_threshold

        self._transformer_normalize_mean = torch.tensor([0., 0., 0., 0.], dtype=torch.float)
        self._transformer_normalize_std = torch.tensor([.1, .1, .2, .2], dtype=torch.float)

    def forward(self, features: Tensor,
                pre_extract_transform: Callable = None,
                post_extract_transform: Callable = None) -> Tuple[Tensor, Tensor]:
        if pre_extract_transform:
            features = pre_extract_transform(features)

        features = self._extractor(features)

        if post_extract_transform:
            features = post_extract_transform(features)

        proposal_classes = self.proposal_class(features)
        proposal_transformers = self.proposal_transformer(features)

        proposal_classes = proposal_classes.view(features.shape[0], self._num_classes)
        proposal_transformers = proposal_transformers.view(features.shape[0], self._num_classes, 4)

        return proposal_classes, proposal_transformers

    def sample(self, proposal_bboxes: Tensor,
               gt_bboxes: Tensor, gt_classes: Tensor) -> Tuple[Tensor, Tensor, Tensor]:
        sampled_indices = torch.arange(proposal_bboxes.shape[0]).to(proposal_bboxes.device)

        labels = torch.full((proposal_bboxes.shape[0],), -1, dtype=torch.long, device=proposal_bboxes.device)
        ious = box_iou(proposal_bboxes, gt_bboxes)

        proposal_max_ious, proposal_assignments = ious.max(dim=1)
        gt_max_ious, gt_assignments = ious.max(dim=0)

        low_quality_indices = (proposal_max_ious < 0.5).nonzero().flatten()

        addition_indices = ((ious >= 0.3) & (ious == gt_max_ious.unsqueeze(dim=0))).nonzero()[:, 0]
        addition_gt_classes = gt_classes[proposal_assignments[addition_indices]]

        high_quality_indices = (proposal_max_ious >= 0.5).nonzero().flatten()
        high_quality_gt_classes = gt_classes[proposal_assignments[high_quality_indices]]

        labels[low_quality_indices] = 0
        labels[addition_indices] = addition_gt_classes
        labels[high_quality_indices] = high_quality_gt_classes

        fg_indices = (labels > 0).nonzero().flatten()
        bg_indices = (labels == 0).nonzero().flatten()
        explicit_bg_indices = torch.cat([addition_indices[addition_gt_classes == 0],
                                         high_quality_indices[high_quality_gt_classes == 0]],
                                        dim=0)

        expected_num_fg_indices = int(self._num_proposal_samples_per_batch * 0.5)
        fg_indices = fg_indices[torch.randperm(fg_indices.shape[0])[:expected_num_fg_indices]]

        expected_num_bg_indices = self._num_proposal_samples_per_batch - fg_indices.shape[0]
        explicit_bg_indices = explicit_bg_indices[torch.randperm(explicit_bg_indices.shape[0])][:expected_num_bg_indices // 2]
        bg_indices = torch.cat([
            bg_indices[torch.randperm(bg_indices.shape[0])[:expected_num_bg_indices - explicit_bg_indices.shape[0]]],
            explicit_bg_indices
        ], dim=0).unique(dim=0)
        bg_indices = bg_indices[torch.randperm(bg_indices.shape[0])]

        bg_fg_max_ratio = 5
        bg_indices = bg_indices[:fg_indices.shape[0] * bg_fg_max_ratio + 1]  # to guarantee that at least 1 bg

        selected_indices = torch.cat([fg_indices, bg_indices], dim=0)
        selected_indices = selected_indices[torch.randperm(selected_indices.shape[0])]

        proposal_bboxes = proposal_bboxes[selected_indices]
        sampled_indices = sampled_indices[selected_indices]

        gt_bboxes = gt_bboxes[proposal_assignments[selected_indices]]
        gt_proposal_classes = labels[selected_indices]
        gt_proposal_transformers = BBox.calc_transformer(proposal_bboxes, gt_bboxes)

        return sampled_indices, gt_proposal_classes, gt_proposal_transformers

    def loss(self, proposal_classes: Tensor, proposal_transformers: Tensor,
             gt_proposal_classes: Tensor, gt_proposal_transformers: Tensor) -> Tuple[Tensor, Tensor]:
        proposal_transformers = proposal_transformers[torch.arange(end=proposal_transformers.shape[0]), gt_proposal_classes]
        transformer_normalize_mean = self._transformer_normalize_mean.to(device=gt_proposal_transformers.device)
        transformer_normalize_std = self._transformer_normalize_std.to(device=gt_proposal_transformers.device)
        gt_proposal_transformers = (gt_proposal_transformers - transformer_normalize_mean) / transformer_normalize_std  # scale up target to make regressor easier to learn

        proposal_class_loss = F.cross_entropy(input=proposal_classes, target=gt_proposal_classes)

        fg_indices = gt_proposal_classes.nonzero().flatten()
        proposal_transformer_loss = beta_smooth_l1_loss(input=proposal_transformers[fg_indices],
                                                        target=gt_proposal_transformers[fg_indices],
                                                        beta=self._proposal_smooth_l1_loss_beta)

        return proposal_class_loss, proposal_transformer_loss

    def generate_detections(self, proposal_bboxes: Tensor, proposal_classes: Tensor, proposal_transformers: Tensor,
                            padded_image_width: int, padded_image_height: int) -> Tuple[Tensor, Tensor, Tensor]:
        transformer_normalize_std = self._transformer_normalize_std.to(device=proposal_bboxes.device)
        transformer_normalize_mean = self._transformer_normalize_mean.to(device=proposal_bboxes.device)
        proposal_transformers = proposal_transformers * transformer_normalize_std + transformer_normalize_mean

        proposal_bboxes = proposal_bboxes[:, None, :].repeat(1, self._num_classes, 1)
        detection_bboxes = BBox.apply_transformer(proposal_bboxes, proposal_transformers)

        detection_bboxes = BBox.clip(detection_bboxes, left=0, top=0, right=padded_image_width, bottom=padded_image_height)
        detection_probs = F.softmax(input=proposal_classes, dim=-1)

        nms_bboxes = []
        nms_classes = []
        nms_probs = []

        for c in range(1, self._num_classes):
            class_bboxes = detection_bboxes[:, c, :]
            class_probs = detection_probs[:, c]

            kept_indices = nms(class_bboxes, class_probs, iou_threshold=self._detection_nms_threshold)
            class_bboxes = class_bboxes[kept_indices]
            class_probs = class_probs[kept_indices]

            nms_bboxes.append(class_bboxes)
            nms_classes.append(torch.full((kept_indices.shape[0],), c, dtype=torch.int, device=kept_indices.device))
            nms_probs.append(class_probs)

        nms_bboxes = torch.cat(nms_bboxes, dim=0) if len(nms_bboxes) > 0 else torch.empty(0, 4).to(detection_bboxes)
        nms_classes = torch.cat(nms_classes, dim=0) if len(nms_classes) > 0 else torch.empty(0, 4, dtype=torch.int).to(nms_bboxes.device)
        nms_probs = torch.cat(nms_probs, dim=0) if len(nms_classes) > 0 else torch.empty(0, 4).to(detection_probs)

        _, sorted_indices = torch.sort(nms_probs, dim=-1, descending=True)
        detection_bboxes = nms_bboxes[sorted_indices][:self._num_detections_per_image]
        detection_classes = nms_classes[sorted_indices][:self._num_detections_per_image]
        detection_probs = nms_probs[sorted_indices][:self._num_detections_per_image]

        return detection_bboxes, detection_classes, detection_probs
