from typing import Tuple, List, Optional

import torch
from torch import nn, Tensor
from torch.nn import functional as F
from torchvision.ops.boxes import nms, box_iou

from ....bbox import BBox
from ....extension.functional import beta_smooth_l1_loss


class RPN(nn.Module):

    def __init__(self, extractor: nn.Module, num_extractor_out: int,
                 anchor_ratios: List[Tuple[int, int]], anchor_sizes: List[int],
                 train_pre_nms_top_n: int, train_post_nms_top_n: int,
                 eval_pre_nms_top_n: int, eval_post_nms_top_n: int,
                 num_anchor_samples_per_batch: int, anchor_smooth_l1_loss_beta: float,
                 proposal_nms_threshold: float):
        super().__init__()

        self._extractor = extractor

        self._anchor_ratios = anchor_ratios
        self._anchor_sizes = anchor_sizes

        num_anchor_ratios = len(self._anchor_ratios)
        num_anchor_sizes = len(self._anchor_sizes)
        num_anchors = num_anchor_ratios * num_anchor_sizes

        self._train_pre_nms_top_n = train_pre_nms_top_n
        self._train_post_nms_top_n = train_post_nms_top_n
        self._eval_pre_nms_top_n = eval_pre_nms_top_n
        self._eval_post_nms_top_n = eval_post_nms_top_n

        self._num_anchor_samples_per_batch = num_anchor_samples_per_batch
        self._anchor_smooth_l1_loss_beta = anchor_smooth_l1_loss_beta

        self._proposal_nms_threshold = proposal_nms_threshold

        self.anchor_objectness = nn.Conv2d(in_channels=num_extractor_out, out_channels=num_anchors * 2, kernel_size=1)
        self.anchor_transformer = nn.Conv2d(in_channels=num_extractor_out, out_channels=num_anchors * 4, kernel_size=1)

    def forward(self, features_batch: Tensor) -> Tuple[Tensor, Tensor]:
        batch_size = features_batch.shape[0]

        features_batch = self._extractor(features_batch)
        anchor_objectnesses_batch = self.anchor_objectness(features_batch)
        anchor_transformers_batch = self.anchor_transformer(features_batch)

        anchor_objectnesses_batch = anchor_objectnesses_batch.permute(0, 2, 3, 1).contiguous().view(batch_size, -1, 2)
        anchor_transformers_batch = anchor_transformers_batch.permute(0, 2, 3, 1).contiguous().view(batch_size, -1, 4)

        return anchor_objectnesses_batch, anchor_transformers_batch

    def sample(self, anchor_bboxes: Tensor, gt_bboxes: Tensor, gt_classes: Tensor,
               padded_image_width: int, padded_image_height: int) -> Tuple[Tensor, Optional[Tensor], Optional[Tensor]]:
        sampled_indices = torch.arange(anchor_bboxes.shape[0]).to(anchor_bboxes.device)

        inside_indices = BBox.inside(anchor_bboxes, left=0, top=0, right=padded_image_width, bottom=padded_image_height).nonzero().flatten()
        anchor_bboxes = anchor_bboxes[inside_indices]
        sampled_indices = sampled_indices[inside_indices]

        if sampled_indices.shape[0] == 0:
            return sampled_indices, None, None

        labels = torch.full((anchor_bboxes.shape[0],), -1, dtype=torch.long, device=anchor_bboxes.device)
        ious = box_iou(anchor_bboxes, gt_bboxes)

        anchor_max_ious, anchor_assignments = ious.max(dim=1)
        gt_max_ious, gt_assignments = ious.max(dim=0)

        low_quality_indices = (anchor_max_ious < 0.3).nonzero().flatten()

        addition_indices = ((ious >= 0.1) & (ious == gt_max_ious.unsqueeze(dim=0))).nonzero()[:, 0]
        addition_gt_classes = gt_classes[anchor_assignments[addition_indices]]

        high_quality_indices = (anchor_max_ious >= 0.7).nonzero().flatten()
        high_quality_gt_classes = gt_classes[anchor_assignments[high_quality_indices]]

        labels[low_quality_indices] = 0
        labels[addition_indices] = addition_gt_classes
        labels[high_quality_indices] = high_quality_gt_classes

        fg_indices = (labels > 0).nonzero().flatten()
        bg_indices = (labels == 0).nonzero().flatten()
        explicit_bg_indices = torch.cat([addition_indices[addition_gt_classes == 0],
                                        high_quality_indices[high_quality_gt_classes == 0]],
                                        dim=0)

        expected_num_fg_indices = int(self._num_anchor_samples_per_batch * 0.5)
        fg_indices = fg_indices[torch.randperm(fg_indices.shape[0])[:expected_num_fg_indices]]

        expected_num_bg_indices = self._num_anchor_samples_per_batch - fg_indices.shape[0]
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

        anchor_bboxes = anchor_bboxes[selected_indices]
        sampled_indices = sampled_indices[selected_indices]

        gt_bboxes = gt_bboxes[anchor_assignments[selected_indices]]
        gt_anchor_objectnesses = labels[selected_indices].gt(0).long()
        gt_anchor_transformers = BBox.calc_transformer(anchor_bboxes, gt_bboxes)

        return sampled_indices, gt_anchor_objectnesses, gt_anchor_transformers

    def loss(self, anchor_objectnesses: Tensor, anchor_transformers: Tensor,
             gt_anchor_objectnesses: Tensor, gt_anchor_transformers: Tensor) -> Tuple[Tensor, Tensor]:
        cross_entropy = F.cross_entropy(input=anchor_objectnesses, target=gt_anchor_objectnesses)

        fg_indices = gt_anchor_objectnesses.nonzero().flatten()
        smooth_l1_loss = beta_smooth_l1_loss(input=anchor_transformers[fg_indices],
                                             target=gt_anchor_transformers[fg_indices],
                                             beta=self._anchor_smooth_l1_loss_beta)

        return cross_entropy, smooth_l1_loss

    def generate_anchors(self, padded_image_width: int, padded_image_height: int, num_x_anchors: int, num_y_anchors: int, scale: float) -> Tensor:
        center_ys = torch.linspace(start=0, end=padded_image_height, steps=num_y_anchors + 2)[1:-1]
        center_xs = torch.linspace(start=0, end=padded_image_width, steps=num_x_anchors + 2)[1:-1]
        ratios = torch.tensor(self._anchor_ratios, dtype=torch.float)
        ratios = ratios[:, 0] / ratios[:, 1]
        sizes = torch.tensor(self._anchor_sizes, dtype=torch.float)

        center_ys, center_xs, ratios, sizes = torch.meshgrid(center_ys, center_xs, ratios, sizes)

        center_ys = center_ys.reshape(-1)
        center_xs = center_xs.reshape(-1)
        ratios = ratios.reshape(-1)
        sizes = sizes.reshape(-1)

        widths = sizes * torch.sqrt(1 / ratios) * scale
        heights = sizes * torch.sqrt(ratios) * scale

        center_based_anchor_bboxes = torch.stack((center_xs, center_ys, widths, heights), dim=1)
        anchor_bboxes = BBox.from_center_base(center_based_anchor_bboxes)
        return anchor_bboxes

    def generate_proposals_batch(self, anchor_bboxes_batch: List[Tensor],
                                 anchor_objectnesses_batch: Tensor, anchor_transformers_batch: Tensor,
                                 padded_image_width: int, padded_image_height: int) -> Tuple[List[Tensor], List[Tensor]]:
        batch_size = len(anchor_bboxes_batch)
        pre_nms_top_n = self._train_pre_nms_top_n if self.training else self._eval_pre_nms_top_n
        post_nms_top_n = self._train_post_nms_top_n if self.training else self._eval_post_nms_top_n

        proposal_bboxes_batch, proposal_probs_batch = [], []
        for b in range(batch_size):
            proposal_bboxes = BBox.apply_transformer(anchor_bboxes_batch[b], anchor_transformers_batch[b])
            proposal_bboxes = BBox.clip(proposal_bboxes, left=0, top=0, right=padded_image_width, bottom=padded_image_height)
            proposal_probs = F.softmax(input=anchor_objectnesses_batch[b, :, 1], dim=-1)

            _, sorted_indices = torch.sort(proposal_probs, dim=-1, descending=True)
            proposal_bboxes = proposal_bboxes[sorted_indices][:pre_nms_top_n]
            proposal_probs = proposal_probs[sorted_indices][:pre_nms_top_n]

            kept_indices = nms(proposal_bboxes, proposal_probs, iou_threshold=self._proposal_nms_threshold)
            proposal_bboxes = proposal_bboxes[kept_indices][:post_nms_top_n]
            proposal_probs = proposal_probs[kept_indices][:post_nms_top_n]

            # NOTE: It is necessary to detach here for jointly training
            proposal_bboxes_batch.append(proposal_bboxes.detach())
            proposal_probs_batch.append(proposal_probs.detach())

        return proposal_bboxes_batch, proposal_probs_batch
