from math import ceil
from typing import Tuple, Dict, Any

import PIL
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch import Tensor
from torchvision.transforms import transforms


class Preprocessor:

    PROCESS_KEY_IS_TRAIN_OR_EVAL = 'is_train_or_eval'
    PROCESS_KEY_ORIGIN_WIDTH = 'origin_width'
    PROCESS_KEY_ORIGIN_HEIGHT = 'origin_height'
    PROCESS_KEY_WIDTH_SCALE = 'width_scale'
    PROCESS_KEY_HEIGHT_SCALE = 'height_scale'
    PROCESS_KEY_RIGHT_PAD = 'right_pad'
    PROCESS_KEY_BOTTOM_PAD = 'bottom_pad'

    def __init__(self,
                 image_resized_width: int, image_resized_height: int,
                 image_min_side: int, image_max_side: int,
                 image_side_divisor: int):
        super().__init__()
        self.image_resized_width = image_resized_width
        self.image_resized_height = image_resized_height
        self.image_min_side = image_min_side
        self.image_max_side = image_max_side
        self.image_side_divisor = image_side_divisor

    def process(self, image: PIL.Image.Image, is_train_or_eval: bool) -> Tuple[Tensor, Dict[str, Any]]:
        scale_for_width = 1 if self.image_resized_width == -1 else self.image_resized_width / image.width
        scale_for_height = 1 if self.image_resized_height == -1 else self.image_resized_height / image.height

        # resize according to the rules:
        #   1. scale shorter side to `image_min_side`
        #   2. after scaling, if longer side > `image_max_side`, scale longer side to `image_max_side`

        if self.image_min_side == -1:
            scale_for_shorter_side = 1
        else:
            scale_for_shorter_side = self.image_min_side / min(image.width * scale_for_width,
                                                               image.height * scale_for_height)

        if self.image_max_side == -1:
            scale_for_longer_side = 1
        else:
            longer_side_after_scaling = max(image.width * scale_for_width,
                                            image.height * scale_for_height) * scale_for_shorter_side
            scale_for_longer_side = (self.image_max_side / longer_side_after_scaling) if longer_side_after_scaling > self.image_max_side else 1

        scale_for_width *= scale_for_shorter_side * scale_for_longer_side
        scale_for_height *= scale_for_shorter_side * scale_for_longer_side

        scaled_image_width = round(image.width * scale_for_width)
        scaled_image_height = round(image.height * scale_for_height)

        image_right_pad = int(ceil(scaled_image_width / self.image_side_divisor) * self.image_side_divisor) - scaled_image_width
        image_bottom_pad = int(ceil(scaled_image_height / self.image_side_divisor) * self.image_side_divisor) - scaled_image_height

        transform = self._compose_transform(is_train_or_eval,
                                            resized_width=scaled_image_width, resized_height=scaled_image_height,
                                            right_pad=image_right_pad, bottom_pad=image_bottom_pad)
        processed_image = transform(image)

        process_dict = {
            self.PROCESS_KEY_IS_TRAIN_OR_EVAL: is_train_or_eval,
            self.PROCESS_KEY_ORIGIN_WIDTH: image.width,
            self.PROCESS_KEY_ORIGIN_HEIGHT: image.height,
            self.PROCESS_KEY_WIDTH_SCALE: scale_for_width,
            self.PROCESS_KEY_HEIGHT_SCALE: scale_for_height,
            self.PROCESS_KEY_RIGHT_PAD: image_right_pad,
            self.PROCESS_KEY_BOTTOM_PAD: image_bottom_pad
        }

        return processed_image, process_dict

    def _compose_transform(self, is_train_or_eval: bool,
                           resized_width: int, resized_height: int,
                           right_pad: int, bottom_pad: int) -> transforms.Compose:
        transform = transforms.Compose([
            transforms.Resize(size=(resized_height, resized_width)),  # interpolation `BILINEAR` is applied by default
            transforms.Pad(padding=(0, 0, right_pad, bottom_pad), fill=0),  # padding has format (left, top, right, bottom)
            transforms.ToTensor()
        ])
        return transform

    @staticmethod
    def build_noop() -> 'Preprocessor':
        return Preprocessor(image_resized_width=-1, image_resized_height=-1,
                            image_min_side=-1, image_max_side=-1,
                            image_side_divisor=1)

    @classmethod
    def inv_process_bboxes(cls, process_dict: Dict[str, Any],
                           bboxes: Tensor) -> Tensor:
        inv_bboxes = bboxes.clone()
        inv_bboxes[:, [0, 2]] /= process_dict[cls.PROCESS_KEY_WIDTH_SCALE]
        inv_bboxes[:, [1, 3]] /= process_dict[cls.PROCESS_KEY_HEIGHT_SCALE]
        return inv_bboxes

    @classmethod
    def inv_process_heatmap(cls, process_dict: Dict[str, Any],
                            heatmap: np.ndarray) -> np.ndarray:
        assert heatmap.ndim == 3
        heatmap = torch.from_numpy(heatmap).permute(2, 0, 1)

        right_pad, bottom_pad = process_dict[cls.PROCESS_KEY_RIGHT_PAD], process_dict[cls.PROCESS_KEY_BOTTOM_PAD]
        inv_heatmap = F.pad(input=heatmap,
                            pad=[0, -right_pad, 0, -bottom_pad])  # pad has format [left, right, top, bottom]

        origin_size = (process_dict[cls.PROCESS_KEY_ORIGIN_HEIGHT], process_dict[cls.PROCESS_KEY_ORIGIN_WIDTH])
        inv_heatmap = F.interpolate(input=inv_heatmap.unsqueeze(dim=0),
                                    size=origin_size,
                                    mode='bilinear',
                                    align_corners=True).squeeze(dim=0)

        return inv_heatmap.permute(1, 2, 0).numpy()

    @classmethod
    def inv_process_probmasks(cls, process_dict: Dict[str, Any],
                              probmasks: Tensor) -> Tensor:
        assert probmasks.ndim == 4

        if probmasks.shape[0] > 0:
            right_pad, bottom_pad = process_dict[cls.PROCESS_KEY_RIGHT_PAD], process_dict[cls.PROCESS_KEY_BOTTOM_PAD]
            inv_probmasks = F.pad(input=probmasks,
                                  pad=[0, -right_pad, 0, -bottom_pad])  # pad has format [left, right, top, bottom]

            origin_size = (process_dict[cls.PROCESS_KEY_ORIGIN_HEIGHT], process_dict[cls.PROCESS_KEY_ORIGIN_WIDTH])
            inv_probmasks = F.interpolate(input=inv_probmasks,
                                          size=origin_size,
                                          mode='bilinear',
                                          align_corners=True)
        else:
            inv_probmasks = probmasks

        return inv_probmasks
