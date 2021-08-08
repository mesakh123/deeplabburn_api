import argparse
import sys

import cv2
import numpy as np
from PIL import ImageDraw, Image
from mirle_vision.lib.augmenter import Augmenter
from mirle_vision.lib.preprocessor import Preprocessor
from mirle_vision.lib.task import Task
from mirle_vision.lib.task.instance_segmentation.palette import Palette
from torchvision.transforms.functional import to_pil_image


def _augment(task_name: Task.Name, path_to_data_dir: str):
    aug_strategy = Augmenter.Strategy.ALL
    aug_hflip_prob = 0.5
    aug_vflip_prob = 0
    aug_rotate90_prob = 0
    aug_crop_prob_and_min_max = (0, (0, 1))
    aug_zoom_prob_and_min_max = (0, (-1, 1))
    aug_scale_prob_and_min_max = (0, (-1, 1))
    aug_translate_prob_and_min_max = (0, (-1, 1))
    aug_rotate_prob_and_min_max = (0, (-1, 1))
    aug_shear_prob_and_min_max = (0, (-1, 1))
    aug_blur_prob_and_min_max = (0, (0, 1))
    aug_sharpen_prob_and_min_max = (0, (0, 1))
    aug_color_prob_and_min_max = (0, (-1, 1))
    aug_brightness_prob_and_min_max = (0, (-1, 1))
    aug_grayscale_prob_and_min_max = (0, (0, 1))
    aug_contrast_prob_and_min_max = (0, (-1, 1))
    aug_noise_prob_and_min_max = (0, (0, 1))
    aug_resized_crop_prob_and_width_height = (0, (224, 224))

    preprocessor = Preprocessor.build_noop()
    augmenter = Augmenter(strategy=aug_strategy,
                          aug_hflip_prob=aug_hflip_prob,
                          aug_vflip_prob=aug_vflip_prob,
                          aug_rotate90_prob=aug_rotate90_prob,
                          aug_crop_prob_and_min_max=aug_crop_prob_and_min_max,
                          aug_zoom_prob_and_min_max=aug_zoom_prob_and_min_max,
                          aug_scale_prob_and_min_max=aug_scale_prob_and_min_max,
                          aug_translate_prob_and_min_max=aug_translate_prob_and_min_max,
                          aug_rotate_prob_and_min_max=aug_rotate_prob_and_min_max,
                          aug_shear_prob_and_min_max=aug_shear_prob_and_min_max,
                          aug_blur_prob_and_min_max=aug_blur_prob_and_min_max,
                          aug_sharpen_prob_and_min_max=aug_sharpen_prob_and_min_max,
                          aug_color_prob_and_min_max=aug_color_prob_and_min_max,
                          aug_brightness_prob_and_min_max=aug_brightness_prob_and_min_max,
                          aug_grayscale_prob_and_min_max=aug_grayscale_prob_and_min_max,
                          aug_contrast_prob_and_min_max=aug_contrast_prob_and_min_max,
                          aug_noise_prob_and_min_max=aug_noise_prob_and_min_max,
                          aug_resized_crop_prob_and_width_height=aug_resized_crop_prob_and_width_height)

    if task_name == Task.Name.CLASSIFICATION:
        from mirle_vision.lib.task.classification.dataset import Dataset
        dataset = Dataset(path_to_data_dir, Dataset.Mode.UNION, preprocessor, augmenter)
    elif task_name == Task.Name.DETECTION:
        from mirle_vision.lib.task.detection.dataset import Dataset
        dataset = Dataset(path_to_data_dir, Dataset.Mode.UNION, preprocessor, augmenter, exclude_difficulty=False)
    elif task_name == Task.Name.INSTANCE_SEGMENTATION:
        from mirle_vision.lib.task.instance_segmentation.dataset import Dataset
        dataset = Dataset(path_to_data_dir, Dataset.Mode.UNION, preprocessor, augmenter, exclude_difficulty=False)
    else:
        raise ValueError

    assert len(dataset) > 0

    def on_hflip_prob_change(new_value: int):
        new_hflip_prob = round(new_value / 10, 1)
        dataset.augmenter.aug_hflip_prob = new_hflip_prob
        dataset.augmenter.imgaug_transforms = dataset.augmenter.build_imgaug_transforms()
        print('hflip_prob =', dataset.augmenter.aug_hflip_prob)

    def on_vflip_prob_change(new_value: int):
        new_vflip_prob = round(new_value / 10, 1)
        dataset.augmenter.aug_vflip_prob = new_vflip_prob
        dataset.augmenter.imgaug_transforms = dataset.augmenter.build_imgaug_transforms()
        print('vflip_prob =', dataset.augmenter.aug_vflip_prob)

    def on_rotate90_prob_change(new_value: int):
        new_rotate90_prob = round(new_value / 10, 1)
        dataset.augmenter.aug_rotate90_prob = new_rotate90_prob
        dataset.augmenter.imgaug_transforms = dataset.augmenter.build_imgaug_transforms()
        print('rotate90_prob =', dataset.augmenter.aug_rotate90_prob)

    def on_crop_prob_change(new_value: int):
        new_crop_prob = round(new_value / 10, 1)
        _, (crop_min, crop_max) = dataset.augmenter.aug_crop_prob_and_min_max
        dataset.augmenter.aug_crop_prob_and_min_max = (new_crop_prob, (crop_min, crop_max))
        dataset.augmenter.imgaug_transforms = dataset.augmenter.build_imgaug_transforms()
        print('crop_prob_and_min_max =', dataset.augmenter.aug_crop_prob_and_min_max)

    def on_crop_min_change(new_value: int):
        new_crop_min = round(new_value / 10, 1)
        crop_prob, (_, crop_max) = dataset.augmenter.aug_crop_prob_and_min_max
        dataset.augmenter.aug_crop_prob_and_min_max = (crop_prob, (new_crop_min, crop_max))
        dataset.augmenter.imgaug_transforms = dataset.augmenter.build_imgaug_transforms()
        print('crop_prob_and_min_max =', dataset.augmenter.aug_crop_prob_and_min_max)

    def on_crop_max_change(new_value: int):
        new_crop_max = round(new_value / 10, 1)
        crop_prob, (crop_min, _) = dataset.augmenter.aug_crop_prob_and_min_max
        dataset.augmenter.aug_crop_prob_and_min_max = (crop_prob, (crop_min, new_crop_max))
        dataset.augmenter.imgaug_transforms = dataset.augmenter.build_imgaug_transforms()
        print('crop_prob_and_min_max =', dataset.augmenter.aug_crop_prob_and_min_max)

    def on_zoom_prob_change(new_value: int):
        new_zoom_prob = round(new_value / 10, 1)
        _, (zoom_min, zoom_max) = dataset.augmenter.aug_zoom_prob_and_min_max
        dataset.augmenter.aug_zoom_prob_and_min_max = (new_zoom_prob, (zoom_min, zoom_max))
        dataset.augmenter.imgaug_transforms = dataset.augmenter.build_imgaug_transforms()
        print('zoom_prob_and_min_max =', dataset.augmenter.aug_zoom_prob_and_min_max)

    def on_zoom_min_change(new_value: int):
        new_zoom_min = round(new_value / 10 - 1, 1)
        zoom_prob, (_, zoom_max) = dataset.augmenter.aug_zoom_prob_and_min_max
        dataset.augmenter.aug_zoom_prob_and_min_max = (zoom_prob, (new_zoom_min, zoom_max))
        dataset.augmenter.imgaug_transforms = dataset.augmenter.build_imgaug_transforms()
        print('zoom_prob_and_min_max =', dataset.augmenter.aug_zoom_prob_and_min_max)

    def on_zoom_max_change(new_value: int):
        new_zoom_max = round(new_value / 10 - 1, 1)
        zoom_prob, (zoom_min, _) = dataset.augmenter.aug_zoom_prob_and_min_max
        dataset.augmenter.aug_zoom_prob_and_min_max = (zoom_prob, (zoom_min, new_zoom_max))
        dataset.augmenter.imgaug_transforms = dataset.augmenter.build_imgaug_transforms()
        print('zoom_prob_and_min_max =', dataset.augmenter.aug_zoom_prob_and_min_max)

    def on_scale_prob_change(new_value: int):
        new_scale_prob = round(new_value / 10, 1)
        _, (scale_min, scale_max) = dataset.augmenter.aug_scale_prob_and_min_max
        dataset.augmenter.aug_scale_prob_and_min_max = (new_scale_prob, (scale_min, scale_max))
        dataset.augmenter.imgaug_transforms = dataset.augmenter.build_imgaug_transforms()
        print('scale_prob_and_min_max =', dataset.augmenter.aug_scale_prob_and_min_max)

    def on_scale_min_change(new_value: int):
        new_scale_min = round(new_value / 10 - 1, 1)
        scale_prob, (_, scale_max) = dataset.augmenter.aug_scale_prob_and_min_max
        dataset.augmenter.aug_scale_prob_and_min_max = (scale_prob, (new_scale_min, scale_max))
        dataset.augmenter.imgaug_transforms = dataset.augmenter.build_imgaug_transforms()
        print('scale_prob_and_min_max =', dataset.augmenter.aug_scale_prob_and_min_max)

    def on_scale_max_change(new_value: int):
        new_scale_max = round(new_value / 10 - 1, 1)
        scale_prob, (scale_min, _) = dataset.augmenter.aug_scale_prob_and_min_max
        dataset.augmenter.aug_scale_prob_and_min_max = (scale_prob, (scale_min, new_scale_max))
        dataset.augmenter.imgaug_transforms = dataset.augmenter.build_imgaug_transforms()
        print('scale_prob_and_min_max =', dataset.augmenter.aug_scale_prob_and_min_max)

    def on_translate_prob_change(new_value: int):
        new_translate_prob = round(new_value / 10, 1)
        _, (translate_min, translate_max) = dataset.augmenter.aug_translate_prob_and_min_max
        dataset.augmenter.aug_translate_prob_and_min_max = (new_translate_prob, (translate_min, translate_max))
        dataset.augmenter.imgaug_transforms = dataset.augmenter.build_imgaug_transforms()
        print('translate_prob_and_min_max =', dataset.augmenter.aug_translate_prob_and_min_max)

    def on_translate_min_change(new_value: int):
        new_translate_min = round(new_value / 10 - 1, 1)
        translate_prob, (_, translate_max) = dataset.augmenter.aug_translate_prob_and_min_max
        dataset.augmenter.aug_translate_prob_and_min_max = (translate_prob, (new_translate_min, translate_max))
        dataset.augmenter.imgaug_transforms = dataset.augmenter.build_imgaug_transforms()
        print('translate_prob_and_min_max =', dataset.augmenter.aug_translate_prob_and_min_max)

    def on_translate_max_change(new_value: int):
        new_translate_max = round(new_value / 10 - 1, 1)
        translate_prob, (translate_min, _) = dataset.augmenter.aug_translate_prob_and_min_max
        dataset.augmenter.aug_translate_prob_and_min_max = (translate_prob, (translate_min, new_translate_max))
        dataset.augmenter.imgaug_transforms = dataset.augmenter.build_imgaug_transforms()
        print('translate_prob_and_min_max =', dataset.augmenter.aug_translate_prob_and_min_max)

    def on_rotate_prob_change(new_value: int):
        new_rotate_prob = round(new_value / 10, 1)
        _, (rotate_min, rotate_max) = dataset.augmenter.aug_rotate_prob_and_min_max
        dataset.augmenter.aug_rotate_prob_and_min_max = (new_rotate_prob, (rotate_min, rotate_max))
        dataset.augmenter.imgaug_transforms = dataset.augmenter.build_imgaug_transforms()
        print('rotate_prob_and_min_max =', dataset.augmenter.aug_rotate_prob_and_min_max)

    def on_rotate_min_change(new_value: int):
        new_rotate_min = round(new_value / 10 - 1, 1)
        rotate_prob, (_, rotate_max) = dataset.augmenter.aug_rotate_prob_and_min_max
        dataset.augmenter.aug_rotate_prob_and_min_max = (rotate_prob, (new_rotate_min, rotate_max))
        dataset.augmenter.imgaug_transforms = dataset.augmenter.build_imgaug_transforms()
        print('rotate_prob_and_min_max =', dataset.augmenter.aug_rotate_prob_and_min_max)

    def on_rotate_max_change(new_value: int):
        new_rotate_max = round(new_value / 10 - 1, 1)
        rotate_prob, (rotate_min, _) = dataset.augmenter.aug_rotate_prob_and_min_max
        dataset.augmenter.aug_rotate_prob_and_min_max = (rotate_prob, (rotate_min, new_rotate_max))
        dataset.augmenter.imgaug_transforms = dataset.augmenter.build_imgaug_transforms()
        print('rotate_prob_and_min_max =', dataset.augmenter.aug_rotate_prob_and_min_max)

    def on_shear_prob_change(new_value: int):
        new_shear_prob = round(new_value / 10, 1)
        _, (shear_min, shear_max) = dataset.augmenter.aug_shear_prob_and_min_max
        dataset.augmenter.aug_shear_prob_and_min_max = (new_shear_prob, (shear_min, shear_max))
        dataset.augmenter.imgaug_transforms = dataset.augmenter.build_imgaug_transforms()
        print('shear_prob_and_min_max =', dataset.augmenter.aug_shear_prob_and_min_max)

    def on_shear_min_change(new_value: int):
        new_shear_min = round(new_value / 10 - 1, 1)
        shear_prob, (_, shear_max) = dataset.augmenter.aug_shear_prob_and_min_max
        dataset.augmenter.aug_shear_prob_and_min_max = (shear_prob, (new_shear_min, shear_max))
        dataset.augmenter.imgaug_transforms = dataset.augmenter.build_imgaug_transforms()
        print('shear_prob_and_min_max =', dataset.augmenter.aug_shear_prob_and_min_max)

    def on_shear_max_change(new_value: int):
        new_shear_max = round(new_value / 10 - 1, 1)
        shear_prob, (shear_min, _) = dataset.augmenter.aug_shear_prob_and_min_max
        dataset.augmenter.aug_shear_prob_and_min_max = (shear_prob, (shear_min, new_shear_max))
        dataset.augmenter.imgaug_transforms = dataset.augmenter.build_imgaug_transforms()
        print('shear_prob_and_min_max =', dataset.augmenter.aug_shear_prob_and_min_max)

    def on_blur_prob_change(new_value: int):
        new_blur_prob = round(new_value / 10, 1)
        _, (blur_min, blur_max) = dataset.augmenter.aug_blur_prob_and_min_max
        dataset.augmenter.aug_blur_prob_and_min_max = (new_blur_prob, (blur_min, blur_max))
        dataset.augmenter.imgaug_transforms = dataset.augmenter.build_imgaug_transforms()
        print('blur_prob_and_min_max =', dataset.augmenter.aug_blur_prob_and_min_max)

    def on_blur_min_change(new_value: int):
        new_blur_min = round(new_value / 10, 1)
        blur_prob, (_, blur_max) = dataset.augmenter.aug_blur_prob_and_min_max
        dataset.augmenter.aug_blur_prob_and_min_max = (blur_prob, (new_blur_min, blur_max))
        dataset.augmenter.imgaug_transforms = dataset.augmenter.build_imgaug_transforms()
        print('blur_prob_and_min_max =', dataset.augmenter.aug_blur_prob_and_min_max)

    def on_blur_max_change(new_value: int):
        new_blur_max = round(new_value / 10, 1)
        blur_prob, (blur_min, _) = dataset.augmenter.aug_blur_prob_and_min_max
        dataset.augmenter.aug_blur_prob_and_min_max = (blur_prob, (blur_min, new_blur_max))
        dataset.augmenter.imgaug_transforms = dataset.augmenter.build_imgaug_transforms()
        print('blur_prob_and_min_max =', dataset.augmenter.aug_blur_prob_and_min_max)

    def on_sharpen_prob_change(new_value: int):
        new_sharpen_prob = round(new_value / 10, 1)
        _, (sharpen_min, sharpen_max) = dataset.augmenter.aug_sharpen_prob_and_min_max
        dataset.augmenter.aug_sharpen_prob_and_min_max = (new_sharpen_prob, (sharpen_min, sharpen_max))
        dataset.augmenter.imgaug_transforms = dataset.augmenter.build_imgaug_transforms()
        print('sharpen_prob_and_min_max =', dataset.augmenter.aug_sharpen_prob_and_min_max)

    def on_sharpen_min_change(new_value: int):
        new_sharpen_min = round(new_value / 10, 1)
        sharpen_prob, (_, sharpen_max) = dataset.augmenter.aug_sharpen_prob_and_min_max
        dataset.augmenter.aug_sharpen_prob_and_min_max = (sharpen_prob, (new_sharpen_min, sharpen_max))
        dataset.augmenter.imgaug_transforms = dataset.augmenter.build_imgaug_transforms()
        print('sharpen_prob_and_min_max =', dataset.augmenter.aug_sharpen_prob_and_min_max)

    def on_sharpen_max_change(new_value: int):
        new_sharpen_max = round(new_value / 10, 1)
        sharpen_prob, (sharpen_min, _) = dataset.augmenter.aug_sharpen_prob_and_min_max
        dataset.augmenter.aug_sharpen_prob_and_min_max = (sharpen_prob, (sharpen_min, new_sharpen_max))
        dataset.augmenter.imgaug_transforms = dataset.augmenter.build_imgaug_transforms()
        print('sharpen_prob_and_min_max =', dataset.augmenter.aug_sharpen_prob_and_min_max)

    def on_color_prob_change(new_value: int):
        new_color_prob = round(new_value / 10, 1)
        _, (color_min, color_max) = dataset.augmenter.aug_color_prob_and_min_max
        dataset.augmenter.aug_color_prob_and_min_max = (new_color_prob, (color_min, color_max))
        dataset.augmenter.imgaug_transforms = dataset.augmenter.build_imgaug_transforms()
        print('color_prob_and_min_max =', dataset.augmenter.aug_color_prob_and_min_max)

    def on_color_min_change(new_value: int):
        new_color_min = round(new_value / 10 - 1, 1)
        color_prob, (_, color_max) = dataset.augmenter.aug_color_prob_and_min_max
        dataset.augmenter.aug_color_prob_and_min_max = (color_prob, (new_color_min, color_max))
        dataset.augmenter.imgaug_transforms = dataset.augmenter.build_imgaug_transforms()
        print('color_prob_and_min_max =', dataset.augmenter.aug_color_prob_and_min_max)

    def on_color_max_change(new_value: int):
        new_color_max = round(new_value / 10 - 1, 1)
        color_prob, (color_min, _) = dataset.augmenter.aug_color_prob_and_min_max
        dataset.augmenter.aug_color_prob_and_min_max = (color_prob, (color_min, new_color_max))
        dataset.augmenter.imgaug_transforms = dataset.augmenter.build_imgaug_transforms()
        print('color_prob_and_min_max =', dataset.augmenter.aug_color_prob_and_min_max)

    def on_brightness_prob_change(new_value: int):
        new_brightness_prob = round(new_value / 10, 1)
        _, (brightness_min, brightness_max) = dataset.augmenter.aug_brightness_prob_and_min_max
        dataset.augmenter.aug_brightness_prob_and_min_max = (new_brightness_prob, (brightness_min, brightness_max))
        dataset.augmenter.imgaug_transforms = dataset.augmenter.build_imgaug_transforms()
        print('brightness_prob_and_min_max =', dataset.augmenter.aug_brightness_prob_and_min_max)

    def on_brightness_min_change(new_value: int):
        new_brightness_min = round(new_value / 10 - 1, 1)
        brightness_prob, (_, brightness_max) = dataset.augmenter.aug_brightness_prob_and_min_max
        dataset.augmenter.aug_brightness_prob_and_min_max = (brightness_prob, (new_brightness_min, brightness_max))
        dataset.augmenter.imgaug_transforms = dataset.augmenter.build_imgaug_transforms()
        print('brightness_prob_and_min_max =', dataset.augmenter.aug_brightness_prob_and_min_max)

    def on_brightness_max_change(new_value: int):
        new_brightness_max = round(new_value / 10 - 1, 1)
        brightness_prob, (brightness_min, _) = dataset.augmenter.aug_brightness_prob_and_min_max
        dataset.augmenter.aug_brightness_prob_and_min_max = (brightness_prob, (brightness_min, new_brightness_max))
        dataset.augmenter.imgaug_transforms = dataset.augmenter.build_imgaug_transforms()
        print('brightness_prob_and_min_max =', dataset.augmenter.aug_brightness_prob_and_min_max)

    def on_grayscale_prob_change(new_value: int):
        new_grayscale_prob = round(new_value / 10, 1)
        _, (grayscale_min, grayscale_max) = dataset.augmenter.aug_grayscale_prob_and_min_max
        dataset.augmenter.aug_grayscale_prob_and_min_max = (new_grayscale_prob, (grayscale_min, grayscale_max))
        dataset.augmenter.imgaug_transforms = dataset.augmenter.build_imgaug_transforms()
        print('grayscale_prob_and_min_max =', dataset.augmenter.aug_grayscale_prob_and_min_max)

    def on_grayscale_min_change(new_value: int):
        new_grayscale_min = round(new_value / 10, 1)
        grayscale_prob, (_, grayscale_max) = dataset.augmenter.aug_grayscale_prob_and_min_max
        dataset.augmenter.aug_grayscale_prob_and_min_max = (grayscale_prob, (new_grayscale_min, grayscale_max))
        dataset.augmenter.imgaug_transforms = dataset.augmenter.build_imgaug_transforms()
        print('grayscale_prob_and_min_max =', dataset.augmenter.aug_grayscale_prob_and_min_max)

    def on_grayscale_max_change(new_value: int):
        new_grayscale_max = round(new_value / 10, 1)
        grayscale_prob, (grayscale_min, _) = dataset.augmenter.aug_grayscale_prob_and_min_max
        dataset.augmenter.aug_grayscale_prob_and_min_max = (grayscale_prob, (grayscale_min, new_grayscale_max))
        dataset.augmenter.imgaug_transforms = dataset.augmenter.build_imgaug_transforms()
        print('grayscale_prob_and_min_max =', dataset.augmenter.aug_grayscale_prob_and_min_max)

    def on_contrast_prob_change(new_value: int):
        new_contrast_prob = round(new_value / 10, 1)
        _, (contrast_min, contrast_max) = dataset.augmenter.aug_contrast_prob_and_min_max
        dataset.augmenter.aug_contrast_prob_and_min_max = (new_contrast_prob, (contrast_min, contrast_max))
        dataset.augmenter.imgaug_transforms = dataset.augmenter.build_imgaug_transforms()
        print('contrast_prob_and_min_max =', dataset.augmenter.aug_contrast_prob_and_min_max)

    def on_contrast_min_change(new_value: int):
        new_contrast_min = round(new_value / 10 - 1, 1)
        contrast_prob, (_, contrast_max) = dataset.augmenter.aug_contrast_prob_and_min_max
        dataset.augmenter.aug_contrast_prob_and_min_max = (contrast_prob, (new_contrast_min, contrast_max))
        dataset.augmenter.imgaug_transforms = dataset.augmenter.build_imgaug_transforms()
        print('contrast_prob_and_min_max =', dataset.augmenter.aug_contrast_prob_and_min_max)

    def on_contrast_max_change(new_value: int):
        new_contrast_max = round(new_value / 10 - 1, 1)
        contrast_prob, (contrast_min, _) = dataset.augmenter.aug_contrast_prob_and_min_max
        dataset.augmenter.aug_contrast_prob_and_min_max = (contrast_prob, (contrast_min, new_contrast_max))
        dataset.augmenter.imgaug_transforms = dataset.augmenter.build_imgaug_transforms()
        print('contrast_prob_and_min_max =', dataset.augmenter.aug_contrast_prob_and_min_max)

    def on_noise_prob_change(new_value: int):
        new_noise_prob = round(new_value / 10, 1)
        _, (noise_min, noise_max) = dataset.augmenter.aug_noise_prob_and_min_max
        dataset.augmenter.aug_noise_prob_and_min_max = (new_noise_prob, (noise_min, noise_max))
        dataset.augmenter.imgaug_transforms = dataset.augmenter.build_imgaug_transforms()
        print('noise_prob_and_min_max =', dataset.augmenter.aug_noise_prob_and_min_max)

    def on_noise_min_change(new_value: int):
        new_noise_min = round(new_value / 10, 1)
        noise_prob, (_, noise_max) = dataset.augmenter.aug_noise_prob_and_min_max
        dataset.augmenter.aug_noise_prob_and_min_max = (noise_prob, (new_noise_min, noise_max))
        dataset.augmenter.imgaug_transforms = dataset.augmenter.build_imgaug_transforms()
        print('noise_prob_and_min_max =', dataset.augmenter.aug_noise_prob_and_min_max)

    def on_noise_max_change(new_value: int):
        new_noise_max = round(new_value / 10, 1)
        noise_prob, (noise_min, _) = dataset.augmenter.aug_noise_prob_and_min_max
        dataset.augmenter.aug_noise_prob_and_min_max = (noise_prob, (noise_min, new_noise_max))
        dataset.augmenter.imgaug_transforms = dataset.augmenter.build_imgaug_transforms()
        print('noise_prob_and_min_max =', dataset.augmenter.aug_noise_prob_and_min_max)

    def on_resized_crop_prob_change(new_value: int):
        new_resized_crop_prob = round(new_value / 10, 1)
        _, (resized_crop_width, resized_crop_height) = dataset.augmenter.aug_resized_crop_prob_and_width_height
        dataset.augmenter.aug_resized_crop_prob_and_width_height = (new_resized_crop_prob, (resized_crop_width, resized_crop_height))
        dataset.augmenter.albumentations_transforms = dataset.augmenter.build_albumentations_transforms()
        print('resized_crop_prob_and_width_height =', dataset.augmenter.aug_resized_crop_prob_and_width_height)

    def on_resized_crop_width_change(new_value: int):
        new_resized_crop_width = new_value + 1
        resized_crop_prob, (_, resized_crop_height) = dataset.augmenter.aug_resized_crop_prob_and_width_height
        dataset.augmenter.aug_resized_crop_prob_and_width_height = (resized_crop_prob, (new_resized_crop_width, resized_crop_height))
        dataset.augmenter.albumentations_transforms = dataset.augmenter.build_albumentations_transforms()
        print('resized_crop_prob_and_width_height =', dataset.augmenter.aug_resized_crop_prob_and_width_height)

    def on_resized_crop_height_change(new_value: int):
        new_resized_crop_height = new_value + 1
        resized_crop_prob, (resized_crop_width, _) = dataset.augmenter.aug_resized_crop_prob_and_width_height
        dataset.augmenter.aug_resized_crop_prob_and_width_height = (resized_crop_prob, (resized_crop_width, new_resized_crop_height))
        dataset.augmenter.albumentations_transforms = dataset.augmenter.build_albumentations_transforms()
        print('resized_crop_prob_and_width_height =', dataset.augmenter.aug_resized_crop_prob_and_width_height)

    control_panel_1_window_name = 'Control Panel 1'
    cv2.namedWindow(control_panel_1_window_name, cv2.WINDOW_GUI_NORMAL)
    cv2.createTrackbar('HFlip Prob', control_panel_1_window_name, 5, 10, on_hflip_prob_change)
    cv2.createTrackbar('VFlip Prob', control_panel_1_window_name, 0, 10, on_vflip_prob_change)
    cv2.createTrackbar('Rotate90 Prob', control_panel_1_window_name, 0, 10, on_rotate90_prob_change)
    cv2.createTrackbar('Crop Prob', control_panel_1_window_name, 0, 10, on_crop_prob_change)
    cv2.createTrackbar('Crop Min', control_panel_1_window_name, 0, 10, on_crop_min_change)
    cv2.createTrackbar('Crop Max', control_panel_1_window_name, 10, 10, on_crop_max_change)
    cv2.createTrackbar('Zoom Prob', control_panel_1_window_name, 0, 10, on_zoom_prob_change)
    cv2.createTrackbar('Zoom Min', control_panel_1_window_name, 0, 20, on_zoom_min_change)
    cv2.createTrackbar('Zoom Max', control_panel_1_window_name, 20, 20, on_zoom_max_change)
    cv2.createTrackbar('Scale Prob', control_panel_1_window_name, 0, 10, on_scale_prob_change)
    cv2.createTrackbar('Scale Min', control_panel_1_window_name, 0, 20, on_scale_min_change)
    cv2.createTrackbar('Scale Max', control_panel_1_window_name, 20, 20, on_scale_max_change)
    cv2.createTrackbar('Translate Prob', control_panel_1_window_name, 0, 10, on_translate_prob_change)
    cv2.createTrackbar('Translate Min', control_panel_1_window_name, 0, 20, on_translate_min_change)
    cv2.createTrackbar('Translate Max', control_panel_1_window_name, 20, 20, on_translate_max_change)
    cv2.createTrackbar('Rotate Prob', control_panel_1_window_name, 0, 10, on_rotate_prob_change)
    cv2.createTrackbar('Rotate Min', control_panel_1_window_name, 0, 20, on_rotate_min_change)
    cv2.createTrackbar('Rotate Max', control_panel_1_window_name, 20, 20, on_rotate_max_change)
    cv2.createTrackbar('Shear Prob', control_panel_1_window_name, 0, 10, on_shear_prob_change)
    cv2.createTrackbar('Shear Min', control_panel_1_window_name, 0, 20, on_shear_min_change)
    cv2.createTrackbar('Shear Max', control_panel_1_window_name, 20, 20, on_shear_max_change)
    cv2.createTrackbar('Blur Prob', control_panel_1_window_name, 0, 10, on_blur_prob_change)
    cv2.createTrackbar('Blur Min', control_panel_1_window_name, 0, 10, on_blur_min_change)
    cv2.createTrackbar('Blur Max', control_panel_1_window_name, 10, 10, on_blur_max_change)
    cv2.createTrackbar('Sharpen Prob', control_panel_1_window_name, 0, 10, on_sharpen_prob_change)
    cv2.createTrackbar('Sharpen Min', control_panel_1_window_name, 0, 10, on_sharpen_min_change)
    cv2.createTrackbar('Sharpen Max', control_panel_1_window_name, 10, 10, on_sharpen_max_change)
    cv2.createTrackbar('Color Prob', control_panel_1_window_name, 0, 10, on_color_prob_change)
    cv2.createTrackbar('Color Min', control_panel_1_window_name, 0, 20, on_color_min_change)
    cv2.createTrackbar('Color Max', control_panel_1_window_name, 20, 20, on_color_max_change)
    cv2.createTrackbar('Brightness Prob', control_panel_1_window_name, 0, 10, on_brightness_prob_change)
    cv2.createTrackbar('Brightness Min', control_panel_1_window_name, 0, 20, on_brightness_min_change)
    cv2.createTrackbar('Brightness Max', control_panel_1_window_name, 20, 20, on_brightness_max_change)
    cv2.createTrackbar('Grayscale Prob', control_panel_1_window_name, 0, 10, on_grayscale_prob_change)
    cv2.createTrackbar('Grayscale Min', control_panel_1_window_name, 0, 10, on_grayscale_min_change)
    cv2.createTrackbar('Grayscale Max', control_panel_1_window_name, 10, 10, on_grayscale_max_change)
    cv2.createTrackbar('Contrast Prob', control_panel_1_window_name, 0, 10, on_contrast_prob_change)
    cv2.createTrackbar('Contrast Min', control_panel_1_window_name, 0, 20, on_contrast_min_change)
    cv2.createTrackbar('Contrast Max', control_panel_1_window_name, 20, 20, on_contrast_max_change)
    cv2.createTrackbar('Noise Prob', control_panel_1_window_name, 0, 10, on_noise_prob_change)
    cv2.createTrackbar('Noise Min', control_panel_1_window_name, 0, 10, on_noise_min_change)
    cv2.createTrackbar('Noise Max', control_panel_1_window_name, 10, 10, on_noise_max_change)

    control_panel_2_window_name = 'Control Panel 2'
    cv2.namedWindow(control_panel_2_window_name, cv2.WINDOW_GUI_NORMAL)
    cv2.createTrackbar('Resized Crop Prob', control_panel_2_window_name, 0, 10, on_resized_crop_prob_change)
    cv2.createTrackbar('Resized Crop Width', control_panel_2_window_name, 224, 1023, on_resized_crop_width_change)
    cv2.createTrackbar('Resized Crop Height', control_panel_2_window_name, 224, 1023, on_resized_crop_height_change)

    preview_window_name = 'Preview'
    cv2.namedWindow(preview_window_name, cv2.WINDOW_GUI_NORMAL)

    index = 0

    while True:
        item = dataset[index]

        if task_name == Task.Name.CLASSIFICATION:
            image = to_pil_image(item.processed_image)
        elif task_name == Task.Name.DETECTION:
            image = to_pil_image(item.processed_image)
            processed_bboxes = item.processed_bboxes

            draw = ImageDraw.Draw(image)
            for bbox in processed_bboxes:
                draw.rectangle(((bbox[0], bbox[1]), (bbox[2], bbox[3])), outline='green', width=2)
        elif task_name == Task.Name.INSTANCE_SEGMENTATION:
            image = to_pil_image(item.processed_image)
            processed_bboxes = item.processed_bboxes
            processed_masks = item.processed_masks

            draw = ImageDraw.Draw(image)
            flatten_palette = Palette.get_flatten_palette()

            for bbox in processed_bboxes:
                draw.rectangle(((bbox[0], bbox[1]), (bbox[2], bbox[3])), outline='green', width=2)

            for color, mask in enumerate(processed_masks, start=1):
                mask_image = to_pil_image(mask * color)
                mask_image.putpalette(flatten_palette)
                blended_image = Image.blend(image.convert('RGBA'), mask_image.convert('RGBA'), alpha=0.5).convert('RGB')
                image = Image.composite(blended_image, image, mask=to_pil_image(mask * 255).convert('1'))
        else:
            raise ValueError

        image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
        cv2.putText(image, item.path_to_image, (20, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1, cv2.LINE_AA)
        cv2.putText(image, str(image.shape), (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1, cv2.LINE_AA)
        cv2.imshow(preview_window_name, image)

        key = cv2.waitKey(0)
        if key == 27:
            break
        elif key == ord('f'):
            index = min(index + 1, len(dataset) - 1)
        elif key == ord('a'):
            index = max(index - 1, 0)

    cv2.destroyAllWindows()


if __name__ == '__main__':
    def main():
        parser = argparse.ArgumentParser()

        # region ===== Common arguments =====
        parser.add_argument('-d', '--data_dir', type=str, required=True, help='path to data directory')
        # endregion =========================

        subparsers = parser.add_subparsers(dest='task', help='task name')
        classification_subparser = subparsers.add_parser(Task.Name.CLASSIFICATION.value)
        detection_subparser = subparsers.add_parser(Task.Name.DETECTION.value)
        instance_segmentation_subparser = subparsers.add_parser(Task.Name.INSTANCE_SEGMENTATION.value)

        # region ===== Classification arguments =====
        # endregion =================================

        # region ===== Detection arguments =====
        # endregion ============================

        # region ===== Instance Segmentation arguments =====
        # endregion ========================================

        args = parser.parse_args()

        path_to_data_dir = args.data_dir
        task_name = Task.Name(args.task)

        print('Arguments:\n' + ' '.join(sys.argv[1:]))

        _augment(task_name, path_to_data_dir)

    main()
