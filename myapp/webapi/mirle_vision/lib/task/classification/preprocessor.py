from typing import Tuple, Dict, Any

import PIL
import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor
from torchvision.transforms import transforms

from ...preprocessor import Preprocessor as Base


class Preprocessor(Base):

    PROCESS_KEY_EVAL_CENTER_CROP_RATIO = 'eval_center_crop_ratio'

    def __init__(self,
                 image_resized_width: int, image_resized_height: int,
                 image_min_side: int, image_max_side: int,
                 image_side_divisor: int,
                 eval_center_crop_ratio: float):
        super().__init__(image_resized_width, image_resized_height,
                         image_min_side, image_max_side,
                         image_side_divisor)
        self.eval_center_crop_ratio = eval_center_crop_ratio

    def process(self, image: PIL.Image.Image, is_train_or_eval: bool) -> Tuple[Tensor, Dict[str, Any]]:
        processed_image, process_dict = super().process(image, is_train_or_eval)

        process_dict.update({
            self.PROCESS_KEY_EVAL_CENTER_CROP_RATIO: self.eval_center_crop_ratio
        })

        return processed_image, process_dict

    def _compose_transform(self, is_train_or_eval: bool,
                           resized_width: int, resized_height: int,
                           right_pad: int, bottom_pad: int) -> transforms.Compose:
        if is_train_or_eval or self.eval_center_crop_ratio == 1:
            return super()._compose_transform(is_train_or_eval, resized_width, resized_height, right_pad, bottom_pad)
        else:
            center_crop_width = int(resized_width * self.eval_center_crop_ratio)
            center_crop_height = int(resized_height * self.eval_center_crop_ratio)

            transform = transforms.Compose([
                transforms.Resize(size=(resized_height, resized_width)),  # interpolation `BILINEAR` is applied by default
                transforms.CenterCrop(size=(center_crop_height, center_crop_width)),
                transforms.Pad(padding=(0, 0, right_pad, bottom_pad), fill=0),  # padding has format (left, top, right, bottom)
                transforms.ToTensor()
            ])
            return transform

    @classmethod
    def inv_process_bboxes(cls, process_dict: Dict[str, Any], bboxes: Tensor):
        raise NotImplementedError

    @classmethod
    def inv_process_heatmap(cls, process_dict: Dict[str, Any],
                            heatmap: np.ndarray) -> np.ndarray:
        if process_dict[cls.PROCESS_KEY_IS_TRAIN_OR_EVAL] or process_dict[cls.PROCESS_KEY_EVAL_CENTER_CROP_RATIO] == 1:
            return super().inv_process_heatmap(process_dict, heatmap)
        else:
            assert heatmap.ndim == 3
            heatmap = torch.from_numpy(heatmap).permute(2, 0, 1)

            right_pad, bottom_pad = process_dict[cls.PROCESS_KEY_RIGHT_PAD], process_dict[cls.PROCESS_KEY_BOTTOM_PAD]
            inv_heatmap = F.pad(input=heatmap,
                                pad=[0, -right_pad, 0, -bottom_pad])  # pad has format [left, right, top, bottom]

            center_crop_ratio = process_dict[cls.PROCESS_KEY_EVAL_CENTER_CROP_RATIO]
            crop_margin_width = inv_heatmap.shape[2] / center_crop_ratio - inv_heatmap.shape[2]
            crop_margin_height = inv_heatmap.shape[1] / center_crop_ratio - inv_heatmap.shape[1]
            x_pad = int(crop_margin_width // 2)
            y_pad = int(crop_margin_height // 2)
            inv_heatmap = F.pad(input=inv_heatmap,
                                pad=[x_pad, x_pad, y_pad, y_pad])  # pad has format [left, right, top, bottom]

            origin_size = (process_dict[cls.PROCESS_KEY_ORIGIN_HEIGHT], process_dict[cls.PROCESS_KEY_ORIGIN_WIDTH])
            inv_heatmap = F.interpolate(input=inv_heatmap.unsqueeze(dim=0),
                                        size=origin_size,
                                        mode='bilinear',
                                        align_corners=True).squeeze(dim=0)

            return inv_heatmap.permute(1, 2, 0).numpy()
