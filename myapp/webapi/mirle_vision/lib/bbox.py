from dataclasses import dataclass
from typing import List

import torch
from torch import Tensor


@dataclass
class BBox:

    left: float
    top: float
    right: float
    bottom: float

    def __repr__(self) -> str:
        return 'BBox[l={:.1f}, t={:.1f}, r={:.1f}, b={:.1f}, w={:.1f}, h={:.1f}]'.format(
            self.left, self.top, self.right, self.bottom, self.width, self.height)

    def tolist(self) -> List[float]:
        return [self.left, self.top, self.right, self.bottom]

    @property
    def width(self):
        return self.right - self.left

    @property
    def height(self):
        return self.bottom - self.top

    @staticmethod
    def to_center_base(bboxes: Tensor) -> Tensor:
        return torch.stack([
            (bboxes[..., 0] + bboxes[..., 2]) / 2,
            (bboxes[..., 1] + bboxes[..., 3]) / 2,
            bboxes[..., 2] - bboxes[..., 0],
            bboxes[..., 3] - bboxes[..., 1]
        ], dim=-1)

    @staticmethod
    def from_center_base(center_based_bboxes: Tensor) -> Tensor:
        return torch.stack([
            center_based_bboxes[..., 0] - center_based_bboxes[..., 2] / 2,
            center_based_bboxes[..., 1] - center_based_bboxes[..., 3] / 2,
            center_based_bboxes[..., 0] + center_based_bboxes[..., 2] / 2,
            center_based_bboxes[..., 1] + center_based_bboxes[..., 3] / 2
        ], dim=-1)

    @staticmethod
    def calc_transformer(src_bboxes: Tensor, dst_bboxes: Tensor) -> Tensor:
        center_based_src_bboxes = BBox.to_center_base(src_bboxes)
        center_based_dst_bboxes = BBox.to_center_base(dst_bboxes)
        transformers = torch.stack([
            (center_based_dst_bboxes[..., 0] - center_based_src_bboxes[..., 0]) / center_based_src_bboxes[..., 2],
            (center_based_dst_bboxes[..., 1] - center_based_src_bboxes[..., 1]) / center_based_src_bboxes[..., 3],
            torch.log(center_based_dst_bboxes[..., 2] / center_based_src_bboxes[..., 2]),
            torch.log(center_based_dst_bboxes[..., 3] / center_based_src_bboxes[..., 3])
        ], dim=-1)
        return transformers

    @staticmethod
    def apply_transformer(src_bboxes: Tensor, transformers: Tensor) -> Tensor:
        center_based_src_bboxes = BBox.to_center_base(src_bboxes)
        center_based_dst_bboxes = torch.stack([
            transformers[..., 0] * center_based_src_bboxes[..., 2] + center_based_src_bboxes[..., 0],
            transformers[..., 1] * center_based_src_bboxes[..., 3] + center_based_src_bboxes[..., 1],
            torch.exp(transformers[..., 2]) * center_based_src_bboxes[..., 2],
            torch.exp(transformers[..., 3]) * center_based_src_bboxes[..., 3]
        ], dim=-1)
        dst_bboxes = BBox.from_center_base(center_based_dst_bboxes)
        return dst_bboxes

    @staticmethod
    def inside(bboxes: Tensor, left: float, top: float, right: float, bottom: float) -> Tensor:
        return ((bboxes[..., 0] >= left) * (bboxes[..., 1] >= top) *
                (bboxes[..., 2] <= right) * (bboxes[..., 3] <= bottom))

    @staticmethod
    def clip(bboxes: Tensor, left: float, top: float, right: float, bottom: float) -> Tensor:
        bboxes[..., [0, 2]] = bboxes[..., [0, 2]].clamp(min=left, max=right)
        bboxes[..., [1, 3]] = bboxes[..., [1, 3]].clamp(min=top, max=bottom)
        return bboxes
