from enum import Enum
from typing import Tuple, Optional, List

import albumentations as A
import numpy as np
import torch
from imgaug import BoundingBox, BoundingBoxesOnImage, SegmentationMapsOnImage
from imgaug.augmenters.arithmetic import Add, SaltAndPepper
from imgaug.augmenters.blur import GaussianBlur
from imgaug.augmenters.color import AddToHueAndSaturation, Grayscale
from imgaug.augmenters.contrast import LogContrast
from imgaug.augmenters.convolutional import Sharpen
from imgaug.augmenters.flip import Fliplr, Flipud
from imgaug.augmenters.geometric import Affine, Rot90
from imgaug.augmenters.meta import Sometimes, Sequential, OneOf, SomeOf
from imgaug.augmenters.size import Crop
from torch import Tensor


class Augmenter:

    class Strategy(Enum):
        ALL = 'all'
        ONE = 'one'
        SOME = 'some'

    OPTIONS = [it.value for it in Strategy]

    def __init__(self, strategy: Strategy,
                 aug_hflip_prob: float, aug_vflip_prob: float, aug_rotate90_prob: float,
                 aug_crop_prob_and_min_max: Tuple[float, Tuple[float, float]],
                 aug_zoom_prob_and_min_max: Tuple[float, Tuple[float, float]], aug_scale_prob_and_min_max: Tuple[float, Tuple[float, float]],
                 aug_translate_prob_and_min_max: Tuple[float, Tuple[float, float]], aug_rotate_prob_and_min_max: Tuple[float, Tuple[float, float]],
                 aug_shear_prob_and_min_max: Tuple[float, Tuple[float, float]],
                 aug_blur_prob_and_min_max: Tuple[float, Tuple[float, float]], aug_sharpen_prob_and_min_max: Tuple[float, Tuple[float, float]],
                 aug_color_prob_and_min_max: Tuple[float, Tuple[float, float]], aug_brightness_prob_and_min_max: Tuple[float, Tuple[float, float]],
                 aug_grayscale_prob_and_min_max: Tuple[float, Tuple[float, float]], aug_contrast_prob_and_min_max: Tuple[float, Tuple[float, float]],
                 aug_noise_prob_and_min_max: Tuple[float, Tuple[float, float]],
                 aug_resized_crop_prob_and_width_height: Tuple[float, Tuple[int, int]]):
        super().__init__()
        self.strategy = strategy
        self.aug_hflip_prob = aug_hflip_prob
        self.aug_vflip_prob = aug_vflip_prob
        self.aug_rotate90_prob = aug_rotate90_prob
        self.aug_crop_prob_and_min_max = aug_crop_prob_and_min_max
        self.aug_zoom_prob_and_min_max = aug_zoom_prob_and_min_max
        self.aug_scale_prob_and_min_max = aug_scale_prob_and_min_max
        self.aug_translate_prob_and_min_max = aug_translate_prob_and_min_max
        self.aug_rotate_prob_and_min_max = aug_rotate_prob_and_min_max
        self.aug_shear_prob_and_min_max = aug_shear_prob_and_min_max
        self.aug_blur_prob_and_min_max = aug_blur_prob_and_min_max
        self.aug_sharpen_prob_and_min_max = aug_sharpen_prob_and_min_max
        self.aug_color_prob_and_min_max = aug_color_prob_and_min_max
        self.aug_brightness_prob_and_min_max = aug_brightness_prob_and_min_max
        self.aug_grayscale_prob_and_min_max = aug_grayscale_prob_and_min_max
        self.aug_contrast_prob_and_min_max = aug_contrast_prob_and_min_max
        self.aug_noise_prob_and_min_max = aug_noise_prob_and_min_max
        self.aug_resized_crop_prob_and_width_height = aug_resized_crop_prob_and_width_height

        self.imgaug_transforms = self.build_imgaug_transforms()
        self.albumentations_transforms = self.build_albumentations_transforms()

    def build_imgaug_transforms(self) -> List:
        aug_hflip_prob = self.aug_hflip_prob
        aug_vflip_prob = self.aug_vflip_prob
        aug_rotate90_prob = self.aug_rotate90_prob
        aug_crop_prob = self.aug_crop_prob_and_min_max[0]
        aug_zoom_prob = self.aug_zoom_prob_and_min_max[0]
        aug_scale_prob = self.aug_scale_prob_and_min_max[0]
        aug_translate_prob = self.aug_translate_prob_and_min_max[0]
        aug_rotate_prob = self.aug_rotate_prob_and_min_max[0]
        aug_shear_prob = self.aug_shear_prob_and_min_max[0]
        aug_blur_prob = self.aug_blur_prob_and_min_max[0]
        aug_sharpen_prob = self.aug_sharpen_prob_and_min_max[0]
        aug_color_prob = self.aug_color_prob_and_min_max[0]
        aug_brightness_prob = self.aug_brightness_prob_and_min_max[0]
        aug_grayscale_prob = self.aug_grayscale_prob_and_min_max[0]
        aug_contrast_prob = self.aug_contrast_prob_and_min_max[0]
        aug_noise_prob = self.aug_noise_prob_and_min_max[0]

        assert 0 <= aug_hflip_prob <= 1
        assert 0 <= aug_vflip_prob <= 1
        assert 0 <= aug_rotate90_prob <= 1
        assert 0 <= aug_crop_prob <= 1
        assert 0 <= aug_zoom_prob <= 1
        assert 0 <= aug_scale_prob <= 1
        assert 0 <= aug_translate_prob <= 1
        assert 0 <= aug_rotate_prob <= 1
        assert 0 <= aug_shear_prob <= 1
        assert 0 <= aug_blur_prob <= 1
        assert 0 <= aug_sharpen_prob <= 1
        assert 0 <= aug_color_prob <= 1
        assert 0 <= aug_brightness_prob <= 1
        assert 0 <= aug_grayscale_prob <= 1
        assert 0 <= aug_contrast_prob <= 1
        assert 0 <= aug_noise_prob <= 1

        aug_crop_min_max = self.aug_crop_prob_and_min_max[1]
        aug_zoom_min_max = self.aug_zoom_prob_and_min_max[1]
        aug_scale_min_max = self.aug_scale_prob_and_min_max[1]
        aug_translate_min_max = self.aug_translate_prob_and_min_max[1]
        aug_rotate_min_max = self.aug_rotate_prob_and_min_max[1]
        aug_shear_min_max = self.aug_shear_prob_and_min_max[1]
        aug_blur_min_max = self.aug_blur_prob_and_min_max[1]
        aug_sharpen_min_max = self.aug_sharpen_prob_and_min_max[1]
        aug_color_min_max = self.aug_color_prob_and_min_max[1]
        aug_brightness_min_max = self.aug_brightness_prob_and_min_max[1]
        aug_grayscale_min_max = self.aug_grayscale_prob_and_min_max[1]
        aug_contrast_min_max = self.aug_contrast_prob_and_min_max[1]
        aug_noise_min_max = self.aug_noise_prob_and_min_max[1]

        transforms = [
            Fliplr(p=aug_hflip_prob),
            Flipud(p=aug_vflip_prob),
            Sometimes(aug_rotate90_prob,
                      Rot90(
                          k=[0, 1, 2, 3], keep_size=False
                      )),
            Sometimes(aug_crop_prob,
                      Crop(
                          percent=self.denormalize('crop', normalized_min_max=aug_crop_min_max)
                      )),
            Sometimes(aug_zoom_prob,
                      Affine(
                          scale=self.denormalize('zoom', normalized_min_max=aug_zoom_min_max),
                          fit_output=False
                      )),
            Sometimes(aug_scale_prob,
                      Affine(
                          scale=self.denormalize('scale', normalized_min_max=aug_scale_min_max),
                          fit_output=True
                      )),
            Sometimes(aug_translate_prob,
                      Affine(
                          translate_percent={'x': self.denormalize('translate', normalized_min_max=aug_translate_min_max),
                                             'y': self.denormalize('translate', normalized_min_max=aug_translate_min_max)}
                      )),
            Sometimes(aug_rotate_prob,
                      Affine(
                          rotate=self.denormalize('rotate', normalized_min_max=aug_rotate_min_max)
                      )),
            Sometimes(aug_shear_prob,
                      Affine(
                          shear=self.denormalize('shear', normalized_min_max=aug_shear_min_max)
                      )),
            Sometimes(aug_blur_prob,
                      GaussianBlur(
                          sigma=self.denormalize('blur', normalized_min_max=aug_blur_min_max)
                      )),
            Sometimes(aug_sharpen_prob,
                      Sharpen(
                          alpha=self.denormalize('sharpen', normalized_min_max=aug_sharpen_min_max),
                          lightness=1.0
                      )),
            Sometimes(aug_color_prob,
                      AddToHueAndSaturation(
                          value=self.denormalize('color', normalized_min_max=aug_color_min_max)
                      )),
            Sometimes(aug_brightness_prob,
                      Add(
                          value=self.denormalize('brightness', normalized_min_max=aug_brightness_min_max)
                      )),
            Sometimes(aug_grayscale_prob,
                      Grayscale(
                          alpha=self.denormalize('grayscale', normalized_min_max=aug_grayscale_min_max)
                      )),
            Sometimes(aug_contrast_prob,
                      LogContrast(
                          gain=self.denormalize('contrast', normalized_min_max=aug_contrast_min_max)
                      )),
            Sometimes(aug_noise_prob,
                      SaltAndPepper(
                          p=self.denormalize('noise', normalized_min_max=aug_noise_min_max)
                      ))
        ]
        return transforms

    def build_albumentations_transforms(self) -> List:
        aug_resized_crop_prob = self.aug_resized_crop_prob_and_width_height[0]

        assert 0 <= aug_resized_crop_prob <= 1

        aug_resized_crop_width_height = self.aug_resized_crop_prob_and_width_height[1]

        transforms = [
            A.RandomResizedCrop(height=aug_resized_crop_width_height[1], width=aug_resized_crop_width_height[0],
                                p=aug_resized_crop_prob)
        ]
        return transforms

    def apply(self, image: Tensor,
              bboxes: Optional[Tensor], mask_image: Optional[Tensor],
              **object_field_dict) -> Tuple:
        bbox_params = A.BboxParams(
            format='pascal_voc',
            label_fields=list(object_field_dict.keys())
        ) if bboxes is not None else None

        if self.strategy == Augmenter.Strategy.ALL:
            imgaug_augmenter = Sequential(children=self.imgaug_transforms, random_order=True)
            albumentations_augmenter = A.Compose(self.albumentations_transforms, bbox_params)
        elif self.strategy == Augmenter.Strategy.ONE:
            imgaug_augmenter = OneOf(children=self.imgaug_transforms)
            albumentations_augmenter = A.Compose([A.OneOf(self.albumentations_transforms)], bbox_params)
        elif self.strategy == Augmenter.Strategy.SOME:
            imgaug_augmenter = SomeOf(children=self.imgaug_transforms, random_order=True)
            albumentations_augmenter = A.Compose([A.OneOf([t]) for t in self.albumentations_transforms], bbox_params)
        else:
            raise ValueError('Invalid augmenter strategy')

        image = image.permute(1, 2, 0).mul(255).byte().numpy()

        if bboxes is not None:
            bboxes = bboxes.numpy()
            if mask_image is not None:
                mask_image = mask_image.numpy()

        # region ===== apply imgaug augmentation =====
        if bboxes is not None:
            bboxes = BoundingBoxesOnImage([BoundingBox(x1=it[0], y1=it[1], x2=it[2], y2=it[3]) for it in bboxes.tolist()], shape=image.shape)
            if mask_image is not None:
                mask_image = SegmentationMapsOnImage(mask_image, shape=image.shape)

        image, bboxes, mask_image = imgaug_augmenter(image=image, bounding_boxes=bboxes, segmentation_maps=mask_image)

        if bboxes is not None:
            bboxes = bboxes.clip_out_of_image()
            bboxes = np.array([[bbox.x1, bbox.y1, bbox.x2, bbox.y2] for bbox in bboxes]).reshape(-1, 4)
            areas = (bboxes[:, 2] - bboxes[:, 0]) * (bboxes[:, 3] - bboxes[:, 1])
            kept_indices = (areas >= 1).nonzero()[0]

            bboxes = bboxes[kept_indices]

            if mask_image is not None:
                mask_image = mask_image.get_arr()
                for mask_color in range(1, np.max(mask_image).item() + 1):
                    if mask_color - 1 not in kept_indices:
                        mask_image = np.where(mask_image == mask_color, 0, mask_image)

            object_field_dict = {k: [object_field_dict[k][i] for i in kept_indices.tolist()]
                                 for k in object_field_dict.keys()}
        # endregion ==================================

        # region ===== apply albumentations augmentation =====
        masks = None

        if bboxes is not None:
            bboxes = bboxes.tolist()
            if mask_image is not None:
                mask_colors = np.arange(1, np.max(mask_image).item() + 1)
                masks = np.tile(mask_image, (mask_colors.shape[0], 1, 1))
                masks = (masks == np.tile(mask_colors.reshape((-1, 1, 1)), (1, masks.shape[1], masks.shape[2]))).astype(masks.dtype)
                masks = [it for it in masks]

        aug_dict = albumentations_augmenter(image=image, bboxes=bboxes, masks=masks, **object_field_dict)

        image = aug_dict['image']
        bboxes = aug_dict['bboxes']
        masks = aug_dict['masks']
        object_field_dict = {k: aug_dict[k] for k in object_field_dict.keys()}

        if bboxes is not None:
            bboxes = np.array(bboxes).reshape(-1, 4)
            if masks is not None:
                masks = np.stack(masks, axis=0)
                mask_image = masks * np.tile(np.arange(1, masks.shape[0] + 1).reshape((-1, 1, 1)), (1, masks.shape[1], masks.shape[2]))
                mask_image = mask_image.astype(masks.dtype)
                mask_image = mask_image.max(axis=0)
        # endregion ==========================================

        image = np.ascontiguousarray(image)
        image = torch.from_numpy(image).float().div(255).permute(2, 0, 1)

        if bboxes is not None:
            bboxes = torch.from_numpy(bboxes).float()

        if mask_image is not None:
            mask_image = torch.from_numpy(mask_image).byte()

        return (image, bboxes, mask_image) + tuple(object_field_dict.values())

    @staticmethod
    def denormalize(name: str, normalized_min_max: Tuple[float, float]) -> Tuple[float, float]:
        if name == 'crop':
            # from range [0, 1] to range [0, 0.2]
            assert 0 <= normalized_min_max[0] <= normalized_min_max[1] <= 1
            func = lambda x: x / 5
        elif name == 'zoom':
            # from range [-1, 1] to range [0.5, 1.5]
            assert -1 <= normalized_min_max[0] <= normalized_min_max[1] <= 1
            func = lambda x: (x + 2) / 2
        elif name == 'scale':
            # from range [-1, 1] to range [0.5, 1.5]
            assert -1 <= normalized_min_max[0] <= normalized_min_max[1] <= 1
            func = lambda x: (x + 2) / 2
        elif name == 'translate':
            # from range [-1, 1] to range [-0.3, 0.3]
            assert -1 <= normalized_min_max[0] <= normalized_min_max[1] <= 1
            func = lambda x: x * 0.3
        elif name == 'rotate':
            # from range [-1, 1] to range [-45, 45]
            assert -1 <= normalized_min_max[0] <= normalized_min_max[1] <= 1
            func = lambda x: x * 45
        elif name == 'shear':
            # from range [-1, 1] to range [-30, 30]
            assert -1 <= normalized_min_max[0] <= normalized_min_max[1] <= 1
            func = lambda x: x * 30
        elif name == 'blur':
            # from range [0, 1] to range [0, 10]
            assert 0 <= normalized_min_max[0] <= normalized_min_max[1] <= 1
            func = lambda x: x * 10
        elif name == 'sharpen':
            # from range [0, 1] to range [0, 1]
            assert 0 <= normalized_min_max[0] <= normalized_min_max[1] <= 1
            func = lambda x: x
        elif name == 'color':
            # from range [-1, 1] to range [-50, 50]
            assert -1 <= normalized_min_max[0] <= normalized_min_max[1] <= 1
            func = lambda x: int(x * 50)
        elif name == 'brightness':
            # from range [-1, 1] to range [-50, 50]
            assert -1 <= normalized_min_max[0] <= normalized_min_max[1] <= 1
            func = lambda x: int(x * 50)
        elif name == 'grayscale':
            # from range [0, 1] to range [0, 1]
            assert 0 <= normalized_min_max[0] <= normalized_min_max[1] <= 1
            func = lambda x: x
        elif name == 'contrast':
            # from range [-1, 1] to range [0.5, 1.5]
            assert -1 <= normalized_min_max[0] <= normalized_min_max[1] <= 1
            func = lambda x: (x + 2) / 2
        elif name == 'noise':
            # from range [0, 1] to range [0, 0.5]
            assert 0 <= normalized_min_max[0] <= normalized_min_max[1] <= 1
            func = lambda x: x / 2
        else:
            raise ValueError

        return func(normalized_min_max[0]), func(normalized_min_max[1])
