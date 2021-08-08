from ast import literal_eval
from dataclasses import dataclass
from distutils.util import strtobool
from typing import Tuple, List, Dict, Any, Union

from torch import Tensor

from .algorithm import Algorithm
from .backbone import Backbone
from ... import config
from ...config import REQUIRED, LAZY_DEFAULT


@dataclass
class Config(config.Config):

    algorithm_name: Algorithm.Name = REQUIRED
    backbone_name: Backbone.Name = REQUIRED

    anchor_ratios: List[Tuple[int, int]] = LAZY_DEFAULT
    anchor_sizes: List[int] = LAZY_DEFAULT

    backbone_pretrained: bool = True
    backbone_num_frozen_levels: int = 2

    train_rpn_pre_nms_top_n: int = LAZY_DEFAULT
    train_rpn_post_nms_top_n: int = LAZY_DEFAULT

    eval_rpn_pre_nms_top_n: int = LAZY_DEFAULT
    eval_rpn_post_nms_top_n: int = LAZY_DEFAULT

    num_anchor_samples_per_batch: int = 256
    num_proposal_samples_per_batch: int = 128
    num_detections_per_image: int = 100

    anchor_smooth_l1_loss_beta: float = 1.0
    proposal_smooth_l1_loss_beta: float = 1.0

    proposal_nms_threshold: float = 0.7
    detection_nms_threshold: float = 0.5

    def to_hyper_param_dict(self) -> Dict[str, Union[int, float, str, bool, Tensor]]:
        hyper_param_dict = super().to_hyper_param_dict()
        hyper_param_dict.update({
            'algorithm_name': str(self.algorithm_name),
            'backbone_name': str(self.backbone_name),
            'anchor_ratios': str(self.anchor_ratios),
            'anchor_sizes': str(self.anchor_sizes),
            'backbone_pretrained': self.backbone_pretrained,
            'backbone_num_frozen_levels': self.backbone_num_frozen_levels,
            'train_rpn_pre_nms_top_n': self.train_rpn_pre_nms_top_n,
            'train_rpn_post_nms_top_n': self.train_rpn_post_nms_top_n,
            'eval_rpn_pre_nms_top_n': self.eval_rpn_pre_nms_top_n,
            'eval_rpn_post_nms_top_n': self.eval_rpn_post_nms_top_n,
            'num_anchor_samples_per_batch': self.num_anchor_samples_per_batch,
            'num_proposal_samples_per_batch': self.num_proposal_samples_per_batch,
            'num_detections_per_image': self.num_detections_per_image,
            'anchor_smooth_l1_loss_beta': self.anchor_smooth_l1_loss_beta,
            'proposal_smooth_l1_loss_beta': self.proposal_smooth_l1_loss_beta,
            'proposal_nms_threshold': self.proposal_nms_threshold,
            'detection_nms_threshold': self.detection_nms_threshold
        })
        return hyper_param_dict

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
            num_epochs_to_finish: str = None, max_num_checkpoints: str = None,
            algorithm_name: str = None, backbone_name: str = None,
            anchor_ratios: str = None, anchor_sizes: str = None,
            backbone_pretrained: str = None, backbone_num_frozen_levels: str = None,
            train_rpn_pre_nms_top_n: str = None, train_rpn_post_nms_top_n: str = None,
            eval_rpn_pre_nms_top_n: str = None, eval_rpn_post_nms_top_n: str = None,
            num_anchor_samples_per_batch: str = None, num_proposal_samples_per_batch: str = None,
            num_detections_per_image: str = None,
            anchor_smooth_l1_loss_beta: str = None, proposal_smooth_l1_loss_beta: str = None,
            proposal_nms_threshold: str = None, detection_nms_threshold: str = None
    ) -> Dict[str, Any]:
        config_dict = super(Config, Config).parse_config_dict(
            task_name,
            path_to_checkpoints_dir, path_to_data_dir, path_to_extra_data_dirs,
            path_to_resuming_checkpoint, path_to_finetuning_checkpoint,
            num_workers, visible_devices,
            needs_freeze_bn,
            image_resized_width, image_resized_height,
            image_min_side, image_max_side,
            image_side_divisor,
            aug_strategy,
            aug_hflip_prob, aug_vflip_prob, aug_rotate90_prob,
            aug_crop_prob_and_min_max,
            aug_zoom_prob_and_min_max, aug_scale_prob_and_min_max,
            aug_translate_prob_and_min_max, aug_rotate_prob_and_min_max,
            aug_shear_prob_and_min_max, aug_blur_prob_and_min_max,
            aug_sharpen_prob_and_min_max, aug_color_prob_and_min_max,
            aug_brightness_prob_and_min_max, aug_grayscale_prob_and_min_max,
            aug_contrast_prob_and_min_max, aug_noise_prob_and_min_max,
            aug_resized_crop_prob_and_width_height,
            batch_size, learning_rate, momentum, weight_decay,
            clip_grad_base_and_max,
            step_lr_sizes, step_lr_gamma,
            warm_up_factor, warm_up_num_iters,
            num_batches_to_display, num_epochs_to_validate,
            num_epochs_to_finish, max_num_checkpoints
        )

        assert algorithm_name is not None
        assert backbone_name is not None

        algorithm_name = Algorithm.Name(algorithm_name)
        backbone_name = Backbone.Name(backbone_name)

        if algorithm_name == Algorithm.Name.FASTER_RCNN:
            default_anchor_ratios = [(1, 2), (1, 1), (2, 1)]
            default_anchor_sizes = [128, 256, 512]
            default_train_rpn_pre_nms_top_n = 12000
            default_train_rpn_post_nms_top_n = 2000
            default_eval_rpn_pre_nms_top_n = 6000
            default_eval_rpn_post_nms_top_n = 1000
        elif algorithm_name == Algorithm.Name.FPN:
            default_anchor_ratios = [(1, 2), (1, 1), (2, 1)]
            default_anchor_sizes = [128]
            default_train_rpn_pre_nms_top_n = 2000
            default_train_rpn_post_nms_top_n = 2000
            default_eval_rpn_pre_nms_top_n = 1000
            default_eval_rpn_post_nms_top_n = 1000
        else:
            raise ValueError

        config_dict['algorithm_name'] = Algorithm.Name(algorithm_name)
        config_dict['backbone_name'] = Backbone.Name(backbone_name)

        config_dict['anchor_ratios'] = \
            literal_eval(anchor_ratios) if anchor_ratios is not None else default_anchor_ratios
        config_dict['anchor_sizes'] = \
            literal_eval(anchor_sizes) if anchor_sizes is not None else default_anchor_sizes

        if backbone_pretrained is not None:
            config_dict['backbone_pretrained'] = bool(strtobool(backbone_pretrained))
        if backbone_num_frozen_levels is not None:
            config_dict['backbone_num_frozen_levels'] = int(backbone_num_frozen_levels)

        config_dict['train_rpn_pre_nms_top_n'] = \
            int(train_rpn_pre_nms_top_n) if train_rpn_pre_nms_top_n is not None else default_train_rpn_pre_nms_top_n
        config_dict['train_rpn_post_nms_top_n'] = \
            int(train_rpn_post_nms_top_n) if train_rpn_post_nms_top_n is not None else default_train_rpn_post_nms_top_n

        config_dict['eval_rpn_pre_nms_top_n'] = \
            int(eval_rpn_pre_nms_top_n) if eval_rpn_pre_nms_top_n is not None else default_eval_rpn_pre_nms_top_n
        config_dict['eval_rpn_post_nms_top_n'] = \
            int(eval_rpn_post_nms_top_n) if eval_rpn_post_nms_top_n is not None else default_eval_rpn_post_nms_top_n

        if num_anchor_samples_per_batch is not None:
            config_dict['num_anchor_samples_per_batch'] = int(num_anchor_samples_per_batch)
        if num_proposal_samples_per_batch is not None:
            config_dict['num_proposal_samples_per_batch'] = int(num_proposal_samples_per_batch)
        if num_detections_per_image is not None:
            config_dict['num_detections_per_image'] = int(num_detections_per_image)

        if anchor_smooth_l1_loss_beta is not None:
            config_dict['anchor_smooth_l1_loss_beta'] = float(anchor_smooth_l1_loss_beta)
        if proposal_smooth_l1_loss_beta is not None:
            config_dict['proposal_smooth_l1_loss_beta'] = float(proposal_smooth_l1_loss_beta)

        if proposal_nms_threshold is not None:
            config_dict['proposal_nms_threshold'] = float(proposal_nms_threshold)
        if detection_nms_threshold is not None:
            config_dict['detection_nms_threshold'] = float(detection_nms_threshold)

        return config_dict
