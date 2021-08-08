import pickle
from ast import literal_eval
from dataclasses import dataclass, asdict
from distutils.util import strtobool
from typing import List, Optional, Tuple, Dict, Any, Union

from torch import Tensor

import mirle_vision
from .augmenter import Augmenter
from .task import Task


class RequiredType:
    pass


class LazyDefaultType:
    pass


REQUIRED = RequiredType()
LAZY_DEFAULT = LazyDefaultType()


@dataclass
class Config:

    VERSION: str = mirle_vision.__version__

    task_name: Task.Name = REQUIRED

    path_to_checkpoints_dir: str = REQUIRED
    path_to_data_dir: str = REQUIRED
    path_to_extra_data_dirs: Tuple[str, ...] = ()

    path_to_resuming_checkpoint: Optional[str] = None
    path_to_finetuning_checkpoint: Optional[str] = None

    num_workers: int = 2
    visible_devices: Optional[List[int]] = None
    
    needs_freeze_bn: bool = LAZY_DEFAULT

    image_resized_width: int = LAZY_DEFAULT
    image_resized_height: int = LAZY_DEFAULT
    image_min_side: int = LAZY_DEFAULT
    image_max_side: int = LAZY_DEFAULT
    image_side_divisor: int = LAZY_DEFAULT

    aug_strategy: Augmenter.Strategy = Augmenter.Strategy.ALL
    aug_hflip_prob: float = 0.5
    aug_vflip_prob: float = 0
    aug_rotate90_prob: float = 0
    aug_crop_prob_and_min_max: Tuple[float, Tuple[float, float]] = (0, (0, 1))
    aug_zoom_prob_and_min_max: Tuple[float, Tuple[float, float]] = (0, (-1, 1))
    aug_scale_prob_and_min_max: Tuple[float, Tuple[float, float]] = (0, (-1, 1))
    aug_translate_prob_and_min_max: Tuple[float, Tuple[float, float]] = (0, (-1, 1))
    aug_rotate_prob_and_min_max: Tuple[float, Tuple[float, float]] = (0, (-1, 1))
    aug_shear_prob_and_min_max: Tuple[float, Tuple[float, float]] = (0, (-1, 1))
    aug_blur_prob_and_min_max: Tuple[float, Tuple[float, float]] = (0, (0, 1))
    aug_sharpen_prob_and_min_max: Tuple[float, Tuple[float, float]] = (0, (0, 1))
    aug_color_prob_and_min_max: Tuple[float, Tuple[float, float]] = (0, (-1, 1))
    aug_brightness_prob_and_min_max: Tuple[float, Tuple[float, float]] = (0, (-1, 1))
    aug_grayscale_prob_and_min_max: Tuple[float, Tuple[float, float]] = (0, (0, 1))
    aug_contrast_prob_and_min_max: Tuple[float, Tuple[float, float]] = (0, (-1, 1))
    aug_noise_prob_and_min_max: Tuple[float, Tuple[float, float]] = (0, (0, 1))
    aug_resized_crop_prob_and_width_height: Tuple[float, Tuple[int, int]] = (0, (224, 224))

    batch_size: int = LAZY_DEFAULT
    learning_rate: float = LAZY_DEFAULT
    momentum: float = 0.9
    weight_decay: float = 0.0005
    clip_grad_base_and_max: Optional[str] = None
    step_lr_sizes: List[int] = LAZY_DEFAULT
    step_lr_gamma: float = 0.1
    warm_up_factor: float = 0.3333
    warm_up_num_iters: int = 500

    num_batches_to_display: int = 20
    num_epochs_to_validate: int = 1
    num_epochs_to_finish: int = LAZY_DEFAULT
    max_num_checkpoints: int = 6

    @staticmethod
    def parse_config_dict(
            task_name: str,
            path_to_checkpoints_dir: str, path_to_data_dir: str, path_to_extra_data_dirs: str = None,
            path_to_resuming_checkpoint: str = None, path_to_finetuning_checkpoint: str = None,
            num_workers: str = None, visible_devices: str = None,
            needs_freeze_bn: str = None,
            image_resized_width: str = None, image_resized_height: str = None,
            image_min_side: str = None, image_max_side: str = None,
            image_side_divisor: str = None,
            aug_strategy: str = None,
            aug_hflip_prob: str = None, aug_vflip_prob: str = None, aug_rotate90_prob: str = None,
            aug_crop_prob_and_min_max: str = None,
            aug_zoom_prob_and_min_max: str = None, aug_scale_prob_and_min_max: str = None,
            aug_translate_prob_and_min_max: str = None, aug_rotate_prob_and_min_max: str = None,
            aug_shear_prob_and_min_max: str = None, aug_blur_prob_and_min_max: str = None,
            aug_sharpen_prob_and_min_max: str = None, aug_color_prob_and_min_max: str = None,
            aug_brightness_prob_and_min_max: str = None, aug_grayscale_prob_and_min_max: str = None,
            aug_contrast_prob_and_min_max: str = None, aug_noise_prob_and_min_max: str = None,
            aug_resized_crop_prob_and_width_height: str = None,
            batch_size: str = None, learning_rate: str = None, momentum: str = None, weight_decay: str = None,
            clip_grad_base_and_max: str = None,
            step_lr_sizes: str = None, step_lr_gamma: str = None,
            warm_up_factor: str = None, warm_up_num_iters: str = None,
            num_batches_to_display: str = None, num_epochs_to_validate: str = None,
            num_epochs_to_finish: str = None, max_num_checkpoints: str = None
    ) -> Dict[str, Any]:
        assert task_name is not None
        assert path_to_checkpoints_dir is not None
        assert path_to_data_dir is not None

        task_name = Task.Name(task_name)

        if task_name == Task.Name.CLASSIFICATION:
            default_needs_freeze_bn = False
            default_image_resized_width = 224
            default_image_resized_height = 224
            default_image_min_side = -1
            default_image_max_side = -1
            default_image_side_divisor = 1
            default_batch_size = 8
            default_learning_rate = 0.008
            default_step_lr_sizes = [6, 8]
            default_num_epochs_to_finish = 10
        elif task_name == Task.Name.DETECTION:
            default_needs_freeze_bn = True
            default_image_resized_width = -1
            default_image_resized_height = -1
            default_image_min_side = 600
            default_image_max_side = 1000
            default_image_side_divisor = 32
            default_batch_size = 2
            default_learning_rate = 0.002
            default_step_lr_sizes = [10, 14]
            default_num_epochs_to_finish = 16
        elif task_name == Task.Name.INSTANCE_SEGMENTATION:
            default_needs_freeze_bn = True
            default_image_resized_width = -1
            default_image_resized_height = -1
            default_image_min_side = 800
            default_image_max_side = 1333
            default_image_side_divisor = 32
            default_batch_size = 1
            default_learning_rate = 0.001
            default_step_lr_sizes = [20, 28]
            default_num_epochs_to_finish = 32
        elif task_name == Task.Name.FINE_GRAINED_CLASSIFICATION:
            default_needs_freeze_bn = False
            default_image_resized_width = 448
            default_image_resized_height = 448
            default_image_min_side = -1
            default_image_max_side = -1
            default_image_side_divisor = 1
            default_batch_size = 8
            default_learning_rate = 0.008
            default_step_lr_sizes = [6, 8]
            default_num_epochs_to_finish = 10
        else:
            raise ValueError

        config_dict = {
            'task_name': task_name,
            'path_to_checkpoints_dir': path_to_checkpoints_dir,
            'path_to_data_dir': path_to_data_dir
        }

        if path_to_extra_data_dirs is not None:
            config_dict['path_to_extra_data_dirs'] = literal_eval(path_to_extra_data_dirs)

        if path_to_resuming_checkpoint is not None:
            config_dict['path_to_resuming_checkpoint'] = path_to_resuming_checkpoint
        if path_to_finetuning_checkpoint is not None:
            config_dict['path_to_finetuning_checkpoint'] = path_to_finetuning_checkpoint

        if num_workers is not None:
            config_dict['num_workers'] = int(num_workers)
        if visible_devices is not None:
            config_dict['visible_devices'] = literal_eval(visible_devices)
            
        config_dict['needs_freeze_bn'] = \
            bool(strtobool(needs_freeze_bn)) if needs_freeze_bn is not None else default_needs_freeze_bn

        config_dict['image_resized_width'] = \
            int(image_resized_width) if image_resized_width is not None else default_image_resized_width
        config_dict['image_resized_height'] = \
            int(image_resized_height) if image_resized_height is not None else default_image_resized_height
        config_dict['image_min_side'] = \
            int(image_min_side) if image_min_side is not None else default_image_min_side
        config_dict['image_max_side'] = \
            int(image_max_side) if image_max_side is not None else default_image_max_side
        config_dict['image_side_divisor'] = \
            int(image_side_divisor) if image_side_divisor is not None else default_image_side_divisor

        if aug_strategy is not None:
            config_dict['aug_strategy'] = Augmenter.Strategy(aug_strategy)
        if aug_hflip_prob is not None:
            config_dict['aug_hflip_prob'] = float(aug_hflip_prob)
        if aug_vflip_prob is not None:
            config_dict['aug_vflip_prob'] = float(aug_vflip_prob)
        if aug_rotate90_prob is not None:
            config_dict['aug_rotate90_prob'] = float(aug_rotate90_prob)
        if aug_crop_prob_and_min_max is not None:
            config_dict['aug_crop_prob_and_min_max'] = literal_eval(aug_crop_prob_and_min_max)
        if aug_zoom_prob_and_min_max is not None:
            config_dict['aug_zoom_prob_and_min_max'] = literal_eval(aug_zoom_prob_and_min_max)
        if aug_scale_prob_and_min_max is not None:
            config_dict['aug_scale_prob_and_min_max'] = literal_eval(aug_scale_prob_and_min_max)
        if aug_translate_prob_and_min_max is not None:
            config_dict['aug_translate_prob_and_min_max'] = literal_eval(aug_translate_prob_and_min_max)
        if aug_rotate_prob_and_min_max is not None:
            config_dict['aug_rotate_prob_and_min_max'] = literal_eval(aug_rotate_prob_and_min_max)
        if aug_shear_prob_and_min_max is not None:
            config_dict['aug_shear_prob_and_min_max'] = literal_eval(aug_shear_prob_and_min_max)
        if aug_blur_prob_and_min_max is not None:
            config_dict['aug_blur_prob_and_min_max'] = literal_eval(aug_blur_prob_and_min_max)
        if aug_sharpen_prob_and_min_max is not None:
            config_dict['aug_sharpen_prob_and_min_max'] = literal_eval(aug_sharpen_prob_and_min_max)
        if aug_color_prob_and_min_max is not None:
            config_dict['aug_color_prob_and_min_max'] = literal_eval(aug_color_prob_and_min_max)
        if aug_brightness_prob_and_min_max is not None:
            config_dict['aug_brightness_prob_and_min_max'] = literal_eval(aug_brightness_prob_and_min_max)
        if aug_grayscale_prob_and_min_max is not None:
            config_dict['aug_grayscale_prob_and_min_max'] = literal_eval(aug_grayscale_prob_and_min_max)
        if aug_contrast_prob_and_min_max is not None:
            config_dict['aug_contrast_prob_and_min_max'] = literal_eval(aug_contrast_prob_and_min_max)
        if aug_noise_prob_and_min_max is not None:
            config_dict['aug_noise_prob_and_min_max'] = literal_eval(aug_noise_prob_and_min_max)
        if aug_resized_crop_prob_and_width_height is not None:
            config_dict['aug_resized_crop_prob_and_width_height'] = literal_eval(aug_resized_crop_prob_and_width_height)

        config_dict['batch_size'] = int(batch_size) if batch_size is not None else default_batch_size
        config_dict['learning_rate'] = float(learning_rate) if learning_rate is not None else default_learning_rate
        if momentum is not None:
            config_dict['momentum'] = float(momentum)
        if weight_decay is not None:
            config_dict['weight_decay'] = float(weight_decay)
        if clip_grad_base_and_max is not None:
            config_dict['clip_grad_base_and_max'] = literal_eval(clip_grad_base_and_max)
        config_dict['step_lr_sizes'] = literal_eval(step_lr_sizes) \
            if step_lr_sizes is not None else default_step_lr_sizes
        if step_lr_gamma is not None:
            config_dict['step_lr_gamma'] = float(step_lr_gamma)
        if warm_up_factor is not None:
            config_dict['warm_up_factor'] = float(warm_up_factor)
        if warm_up_num_iters is not None:
            config_dict['warm_up_num_iters'] = int(warm_up_num_iters)

        if num_batches_to_display is not None:
            config_dict['num_batches_to_display'] = int(num_batches_to_display)
        if num_epochs_to_validate is not None:
            config_dict['num_epochs_to_validate'] = int(num_epochs_to_validate)
        config_dict['num_epochs_to_finish'] = \
            int(num_epochs_to_finish) if num_epochs_to_finish is not None else default_num_epochs_to_finish
        if max_num_checkpoints is not None:
            config_dict['max_num_checkpoints'] = int(max_num_checkpoints)

        return config_dict

    def describe(self) -> str:
        text = '\nConfig:\n'
        text += '\n'.join([f'\t{k} = {v}' for k, v in asdict(self).items()]) + '\n'
        return text

    def serialize(self, path_to_pickle_file: str):
        with open(path_to_pickle_file, 'wb') as f:
            pickle.dump(self, f)

    def to_hyper_param_dict(self) -> Dict[str, Union[int, float, str, bool, Tensor]]:
        hyper_param_dict = {
            'VERSION': self.VERSION,
            'num_workers': self.num_workers,
            'visible_devices': str(self.visible_devices),
            'needs_freeze_bn': self.needs_freeze_bn,
            'image_resized_width': self.image_resized_width,
            'image_resized_height': self.image_resized_height,
            'image_min_side': self.image_min_side,
            'image_max_side': self.image_max_side,
            'image_side_divisor': self.image_side_divisor,
            'aug_strategy': str(self.aug_strategy),
            'aug_hflip_prob': self.aug_hflip_prob,
            'aug_vflip_prob': self.aug_vflip_prob,
            'aug_rotate90_prob': self.aug_rotate90_prob,
            'aug_crop_prob_and_min_max': str(self.aug_crop_prob_and_min_max),
            'aug_zoom_prob_and_min_max': str(self.aug_zoom_prob_and_min_max),
            'aug_scale_prob_and_min_max': str(self.aug_scale_prob_and_min_max),
            'aug_translate_prob_and_min_max': str(self.aug_translate_prob_and_min_max),
            'aug_rotate_prob_and_min_max': str(self.aug_rotate_prob_and_min_max),
            'aug_shear_prob_and_min_max': str(self.aug_shear_prob_and_min_max),
            'aug_blur_prob_and_min_max': str(self.aug_blur_prob_and_min_max),
            'aug_sharpen_prob_and_min_max': str(self.aug_sharpen_prob_and_min_max),
            'aug_color_prob_and_min_max': str(self.aug_color_prob_and_min_max),
            'aug_brightness_prob_and_min_max': str(self.aug_brightness_prob_and_min_max),
            'aug_grayscale_prob_and_min_max': str(self.aug_grayscale_prob_and_min_max),
            'aug_contrast_prob_and_min_max': str(self.aug_contrast_prob_and_min_max),
            'aug_noise_prob_and_min_max': str(self.aug_noise_prob_and_min_max),
            'aug_resized_crop_prob_and_width_height': str(self.aug_resized_crop_prob_and_width_height),
            'batch_size': self.batch_size,
            'learning_rate': self.learning_rate,
            'momentum': self.momentum,
            'weight_decay': self.weight_decay,
            'clip_grad_base_and_max': str(self.clip_grad_base_and_max),
            'step_lr_sizes': str(self.step_lr_sizes),
            'step_lr_gamma': self.step_lr_gamma,
            'warm_up_factor': self.warm_up_factor,
            'warm_up_num_iters': self.warm_up_num_iters,
            'num_epochs_to_finish': self.num_epochs_to_finish
        }
        return hyper_param_dict

    @staticmethod
    def deserialize(path_to_pickle_file: str) -> 'Config':
        with open(path_to_pickle_file, 'rb') as f:
            config = pickle.load(f)
        return config
