import argparse
import os
import sys
import time
import traceback
import uuid
from multiprocessing import Event

import pkg_resources
import torch
from mirle_vision.lib.augmenter import Augmenter
from mirle_vision.lib.config import Config
from mirle_vision.lib.db import DB
from mirle_vision.lib.logger import Logger
from mirle_vision.lib.task import Task
from mirle_vision.lib.task.classification.algorithm import Algorithm as ClassificationAlgorithm
from mirle_vision.lib.task.classification.config import Config as ClassificationConfig
from mirle_vision.lib.task.detection.algorithm import Algorithm as DetectionAlgorithm
from mirle_vision.lib.task.detection.backbone import Backbone
from mirle_vision.lib.task.detection.config import Config as DetectionConfig
from mirle_vision.lib.task.fine_grained_classification.algorithm import Algorithm as FineGrainedClassificationAlgorithm
from mirle_vision.lib.task.fine_grained_classification.config import Config as FineGrainedClassificationConfig
from mirle_vision.lib.task.instance_segmentation.algorithm import Algorithm as InstanceSegmentationAlgorithm
from mirle_vision.lib.task.instance_segmentation.config import Config as InstanceSegmentationConfig
from mirle_vision.lib.util import Util


def _train(config: Config, terminator: Event):
    path_to_checkpoints_dir = config.path_to_checkpoints_dir
    logger = Logger.build(name=os.path.basename(path_to_checkpoints_dir),
                          path_to_log_file=os.path.join(path_to_checkpoints_dir, 'train.log'))
    logger.i(f'Created checkpoints directory: {path_to_checkpoints_dir}')

    logger.i(config.describe())
    config.serialize(path_to_pickle_file=os.path.join(path_to_checkpoints_dir, 'config.pkl'))

    logger.i('Packages:')
    for package in pkg_resources.working_set:
        logger.i(f'\t{package.project_name}, {package.version}')

    logger.i('Arguments:\n' + ' '.join(sys.argv[1:]))

    if config.task_name == Task.Name.CLASSIFICATION:
        from mirle_vision.lib.task.classification.db import DB as T
        db_class = T
    elif config.task_name == Task.Name.DETECTION:
        from mirle_vision.lib.task.detection.db import DB as T
        db_class = T
    elif config.task_name == Task.Name.INSTANCE_SEGMENTATION:
        from mirle_vision.lib.task.instance_segmentation.db import DB as T
        db_class = T
    elif config.task_name == Task.Name.FINE_GRAINED_CLASSIFICATION:
        from mirle_vision.lib.task.fine_grained_classification.db import DB as T
        db_class = T
    else:
        raise ValueError

    db = db_class(path_to_db=os.path.join(path_to_checkpoints_dir, 'summary.db'))
    db.insert_log_table(DB.Log(global_batch=0, status=DB.Log.Status.INITIALIZING, datetime=int(time.time()),
                               epoch=0, total_epoch=config.num_epochs_to_finish,
                               batch=0, total_batch=0,
                               avg_loss=-1,
                               learning_rate=-1, samples_per_sec=-1,
                               eta_hrs=-1))

    try:
        Util.setup_visible_devices(config.visible_devices)
        torch.multiprocessing.set_sharing_strategy('file_system')
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        device_count = 1 if not torch.cuda.is_available() else torch.cuda.device_count()

        # region ===== Setup trainer =====
        augmenter = Augmenter(config.aug_strategy,
                              config.aug_hflip_prob, config.aug_vflip_prob, config.aug_rotate90_prob,
                              config.aug_crop_prob_and_min_max,
                              config.aug_zoom_prob_and_min_max, config.aug_scale_prob_and_min_max,
                              config.aug_translate_prob_and_min_max, config.aug_rotate_prob_and_min_max,
                              config.aug_shear_prob_and_min_max,
                              config.aug_blur_prob_and_min_max, config.aug_sharpen_prob_and_min_max,
                              config.aug_color_prob_and_min_max, config.aug_brightness_prob_and_min_max,
                              config.aug_grayscale_prob_and_min_max, config.aug_contrast_prob_and_min_max,
                              config.aug_noise_prob_and_min_max,
                              config.aug_resized_crop_prob_and_width_height)

        if config.task_name == Task.Name.CLASSIFICATION:
            from mirle_vision.lib.task.classification.trainer import Trainer
            assert isinstance(config, ClassificationConfig)
        elif config.task_name == Task.Name.DETECTION:
            from mirle_vision.lib.task.detection.trainer import Trainer
            assert isinstance(config, DetectionConfig)
        elif config.task_name == Task.Name.INSTANCE_SEGMENTATION:
            from mirle_vision.lib.task.instance_segmentation.trainer import Trainer
            assert isinstance(config, InstanceSegmentationConfig)
        elif config.task_name == Task.Name.FINE_GRAINED_CLASSIFICATION:
            from mirle_vision.lib.task.fine_grained_classification.trainer import Trainer
            assert isinstance(config, FineGrainedClassificationConfig)
        else:
            raise ValueError('Invalid task name')

        trainer = Trainer(config, logger, augmenter, device, device_count, db, terminator, path_to_checkpoints_dir)
        # endregion ======================

        # region ===== Start training =====
        logger.i('Start training with {:s} (batch size: {:d})'.format('CPU' if device == torch.device('cpu') else
                                                                      '{:d} GPUs'.format(device_count),
                                                                      config.batch_size))

        time_checkpoint = time.time()
        trainer.train()
        elapsed_time = time.time() - time_checkpoint
        logger.i('Done! Elapsed {:.2f} hrs'.format(elapsed_time / 3600))
        # endregion =======================
    except Exception as e:
        logger.ex(f'Something went wrong')

        exception_class = e.__class__

        if exception_class == RuntimeError:
            exception_code = 'E01'
        elif exception_class == AssertionError:
            exception_code = 'E02'
        elif exception_class == ValueError:
            exception_code = 'E03'
        elif exception_class == FileNotFoundError:
            exception_code = 'E04'
        else:
            exception_code = 'E00'

        exception = DB.Log.Exception(code=exception_code,
                                     type=exception_class.__name__,
                                     message=str(e),
                                     traceback=traceback.format_exc())
        db.update_log_table_latest_exception(exception)
        raise e
    finally:
        db.close()


if __name__ == '__main__':
    def main():
        parser = argparse.ArgumentParser()

        # region ===== Common arguments =====
        parser.add_argument('-o', '--outputs_dir', type=str, required=True, help='path to outputs directory')
        parser.add_argument('-d', '--data_dir', type=str, required=True, help='path to data directory')
        parser.add_argument('-e', '--extra_data_dirs', type=str, help='path to extra data directory list')
        parser.add_argument('--resume_checkpoint', type=str)
        parser.add_argument('--finetune_checkpoint', type=str)
        parser.add_argument('--num_workers', type=str)
        parser.add_argument('--visible_devices', type=str)
        parser.add_argument('--needs_freeze_bn', type=str)
        parser.add_argument('--image_resized_width', type=str)
        parser.add_argument('--image_resized_height', type=str)
        parser.add_argument('--image_min_side', type=str)
        parser.add_argument('--image_max_side', type=str)
        parser.add_argument('--image_side_divisor', type=str)
        parser.add_argument('--aug_strategy', type=str, choices=Augmenter.OPTIONS)
        parser.add_argument('--aug_hflip_prob', type=str)
        parser.add_argument('--aug_vflip_prob', type=str)
        parser.add_argument('--aug_rotate90_prob', type=str)
        parser.add_argument('--aug_crop_prob_and_min_max', type=str)
        parser.add_argument('--aug_zoom_prob_and_min_max', type=str)
        parser.add_argument('--aug_scale_prob_and_min_max', type=str)
        parser.add_argument('--aug_translate_prob_and_min_max', type=str)
        parser.add_argument('--aug_rotate_prob_and_min_max', type=str)
        parser.add_argument('--aug_shear_prob_and_min_max', type=str)
        parser.add_argument('--aug_blur_prob_and_min_max', type=str)
        parser.add_argument('--aug_sharpen_prob_and_min_max', type=str)
        parser.add_argument('--aug_color_prob_and_min_max', type=str)
        parser.add_argument('--aug_brightness_prob_and_min_max', type=str)
        parser.add_argument('--aug_grayscale_prob_and_min_max', type=str)
        parser.add_argument('--aug_contrast_prob_and_min_max', type=str)
        parser.add_argument('--aug_noise_prob_and_min_max', type=str)
        parser.add_argument('--aug_resized_crop_prob_and_width_height', type=str)
        parser.add_argument('--batch_size', type=str)
        parser.add_argument('--learning_rate', type=str)
        parser.add_argument('--momentum', type=str)
        parser.add_argument('--weight_decay', type=str)
        parser.add_argument('--clip_grad_base_and_max', type=str)
        parser.add_argument('--step_lr_sizes', type=str)
        parser.add_argument('--step_lr_gamma', type=str)
        parser.add_argument('--warm_up_factor', type=str)
        parser.add_argument('--warm_up_num_iters', type=str)
        parser.add_argument('--num_batches_to_display', type=str)
        parser.add_argument('--num_epochs_to_validate', type=str)
        parser.add_argument('--num_epochs_to_finish', type=str)
        parser.add_argument('--max_num_checkpoints', type=str)
        # endregion =========================

        subparsers = parser.add_subparsers(dest='task', help='task name')
        classification_subparser = subparsers.add_parser(Task.Name.CLASSIFICATION.value)
        detection_subparser = subparsers.add_parser(Task.Name.DETECTION.value)
        instance_segmentation_subparser = subparsers.add_parser(Task.Name.INSTANCE_SEGMENTATION.value)
        fine_grained_classification_subparser = subparsers.add_parser(Task.Name.FINE_GRAINED_CLASSIFICATION.value)

        # region ===== Classification arguments =====
        classification_subparser.add_argument('--algorithm', type=str, choices=ClassificationAlgorithm.OPTIONS)
        classification_subparser.add_argument('--pretrained', type=str)
        classification_subparser.add_argument('--num_frozen_levels', type=str)
        classification_subparser.add_argument('--eval_center_crop_ratio', type=str)
        # endregion =================================

        # region ===== Detection arguments =====
        detection_subparser.add_argument('--algorithm', type=str, choices=DetectionAlgorithm.OPTIONS)
        detection_subparser.add_argument('--backbone', type=str, choices=Backbone.OPTIONS)
        detection_subparser.add_argument('--anchor_ratios', type=str)
        detection_subparser.add_argument('--anchor_sizes', type=str)
        detection_subparser.add_argument('--backbone_pretrained', type=str)
        detection_subparser.add_argument('--backbone_num_frozen_levels', type=str)
        detection_subparser.add_argument('--train_rpn_pre_nms_top_n', type=str)
        detection_subparser.add_argument('--train_rpn_post_nms_top_n', type=str)
        detection_subparser.add_argument('--eval_rpn_pre_nms_top_n', type=str)
        detection_subparser.add_argument('--eval_rpn_post_nms_top_n', type=str)
        detection_subparser.add_argument('--num_anchor_samples_per_batch', type=str)
        detection_subparser.add_argument('--num_proposal_samples_per_batch', type=str)
        detection_subparser.add_argument('--num_detections_per_image', type=str)
        detection_subparser.add_argument('--anchor_smooth_l1_loss_beta', type=str)
        detection_subparser.add_argument('--proposal_smooth_l1_loss_beta', type=str)
        detection_subparser.add_argument('--proposal_nms_threshold', type=str)
        detection_subparser.add_argument('--detection_nms_threshold', type=str)
        # endregion ============================

        # region ===== Instance Segmentation arguments =====
        instance_segmentation_subparser.add_argument('--algorithm', type=str, choices=InstanceSegmentationAlgorithm.OPTIONS)
        # endregion ========================================

        # region ===== Fine-Grained Classification arguments =====
        fine_grained_classification_subparser.add_argument('--algorithm', type=str, choices=FineGrainedClassificationAlgorithm.OPTIONS)
        fine_grained_classification_subparser.add_argument('--pretrained', type=str)
        fine_grained_classification_subparser.add_argument('--num_frozen_levels', type=str)
        fine_grained_classification_subparser.add_argument('--roi_resized_width', type=str)
        fine_grained_classification_subparser.add_argument('--roi_resized_height', type=str)
        # endregion ==============================================

        args = parser.parse_args()

        checkpoint_id = str(uuid.uuid4())
        path_to_checkpoints_dir = os.path.join(args.outputs_dir,
                                               args.task,
                                               args.algorithm if 'algorithm' in args else '',
                                               args.backbone if 'backbone' in args else '',
                                               'checkpoints-{:s}-{:s}'.format(time.strftime('%Y%m%d%H%M%S'),
                                                                              checkpoint_id))
        os.makedirs(path_to_checkpoints_dir)

        task_name = Task.Name(args.task)
        common_args = (
            task_name.value,
            path_to_checkpoints_dir,
            args.data_dir,
            args.extra_data_dirs,
            args.resume_checkpoint,
            args.finetune_checkpoint,
            args.num_workers,
            args.visible_devices,
            args.needs_freeze_bn,
            args.image_resized_width,
            args.image_resized_height,
            args.image_min_side,
            args.image_max_side,
            args.image_side_divisor,
            args.aug_strategy,
            args.aug_hflip_prob,
            args.aug_vflip_prob,
            args.aug_rotate90_prob,
            args.aug_crop_prob_and_min_max,
            args.aug_zoom_prob_and_min_max,
            args.aug_scale_prob_and_min_max,
            args.aug_translate_prob_and_min_max,
            args.aug_rotate_prob_and_min_max,
            args.aug_shear_prob_and_min_max,
            args.aug_blur_prob_and_min_max,
            args.aug_sharpen_prob_and_min_max,
            args.aug_color_prob_and_min_max,
            args.aug_brightness_prob_and_min_max,
            args.aug_grayscale_prob_and_min_max,
            args.aug_contrast_prob_and_min_max,
            args.aug_noise_prob_and_min_max,
            args.aug_resized_crop_prob_and_width_height,
            args.batch_size,
            args.learning_rate,
            args.momentum,
            args.weight_decay,
            args.clip_grad_base_and_max,
            args.step_lr_sizes,
            args.step_lr_gamma,
            args.warm_up_factor,
            args.warm_up_num_iters,
            args.num_batches_to_display,
            args.num_epochs_to_validate,
            args.num_epochs_to_finish,
            args.max_num_checkpoints
        )

        if task_name == Task.Name.CLASSIFICATION:
            config_dict = ClassificationConfig.parse_config_dict(
                *common_args,
                args.algorithm,
                args.pretrained,
                args.num_frozen_levels,
                args.eval_center_crop_ratio
            )
            config = ClassificationConfig(**config_dict)
        elif task_name == Task.Name.DETECTION:
            config_dict = DetectionConfig.parse_config_dict(
                *common_args,
                args.algorithm,
                args.backbone,
                args.anchor_ratios,
                args.anchor_sizes,
                args.backbone_pretrained,
                args.backbone_num_frozen_levels,
                args.train_rpn_pre_nms_top_n,
                args.train_rpn_post_nms_top_n,
                args.eval_rpn_pre_nms_top_n,
                args.eval_rpn_post_nms_top_n,
                args.num_anchor_samples_per_batch,
                args.num_proposal_samples_per_batch,
                args.num_detections_per_image,
                args.anchor_smooth_l1_loss_beta,
                args.proposal_smooth_l1_loss_beta,
                args.proposal_nms_threshold,
                args.detection_nms_threshold
            )
            config = DetectionConfig(**config_dict)
        elif task_name == Task.Name.INSTANCE_SEGMENTATION:
            config_dict = InstanceSegmentationConfig.parse_config_dict(
                *common_args,
                args.algorithm
            )
            config = InstanceSegmentationConfig(**config_dict)
        elif task_name == Task.Name.FINE_GRAINED_CLASSIFICATION:
            config_dict = FineGrainedClassificationConfig.parse_config_dict(
                *common_args,
                args.algorithm,
                args.pretrained,
                args.num_frozen_levels,
                args.roi_resized_width,
                args.roi_resized_height
            )
            config = FineGrainedClassificationConfig(**config_dict)
        else:
            raise ValueError('Invalid task name')

        terminator = Event()
        _train(config, terminator)

    main()
