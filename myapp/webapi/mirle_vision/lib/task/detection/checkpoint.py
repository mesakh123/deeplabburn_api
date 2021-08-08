from dataclasses import dataclass

import torch

from .algorithm import Algorithm
from .backbone import Backbone
from .model import Model
from ...checkpoint import Checkpoint as Base


@dataclass
class Checkpoint(Base):

    @staticmethod
    def save(checkpoint: 'Checkpoint', path_to_checkpoint: str):
        model = checkpoint.model
        algorithm: Algorithm = model.algorithm
        backbone: Backbone = algorithm.backbone

        checkpoint_dict = {
            'epoch': checkpoint.epoch,
            'optimizer': checkpoint.optimizer,
            'model_state_dict': model.state_dict(),
            'num_classes': model.num_classes,
            'preprocessor': model.preprocessor,
            'class_to_category_dict': model.class_to_category_dict,
            'category_to_class_dict': model.category_to_class_dict,
            'algorithm_class': algorithm.__class__,
            'algorithm_params': {
                'backbone_class': backbone.__class__,
                'backbone_params': {
                    'pretrained': backbone.pretrained,
                    'num_frozen_levels': backbone.num_frozen_levels
                },
                'anchor_ratios': algorithm.anchor_ratios,
                'anchor_sizes': algorithm.anchor_sizes,
                'train_rpn_pre_nms_top_n': algorithm.train_rpn_pre_nms_top_n,
                'train_rpn_post_nms_top_n': algorithm.train_rpn_post_nms_top_n,
                'eval_rpn_pre_nms_top_n': algorithm.eval_rpn_pre_nms_top_n,
                'eval_rpn_post_nms_top_n': algorithm.eval_rpn_post_nms_top_n,
                'num_anchor_samples_per_batch': algorithm.num_anchor_samples_per_batch,
                'num_proposal_samples_per_batch': algorithm.num_proposal_samples_per_batch,
                'num_detections_per_image': algorithm.num_detections_per_image,
                'anchor_smooth_l1_loss_beta': algorithm.anchor_smooth_l1_loss_beta,
                'proposal_smooth_l1_loss_beta': algorithm.proposal_smooth_l1_loss_beta,
                'proposal_nms_threshold': algorithm.proposal_nms_threshold,
                'detection_nms_threshold': algorithm.detection_nms_threshold
            }
        }
        torch.save(checkpoint_dict, path_to_checkpoint)

    @staticmethod
    def load(path_to_checkpoint: str, device: torch.device) -> 'Checkpoint':
        checkpoint_dict = torch.load(path_to_checkpoint, map_location=device)

        num_classes = checkpoint_dict['num_classes']

        backbone_class = checkpoint_dict['algorithm_params']['backbone_class']
        backbone_params = checkpoint_dict['algorithm_params']['backbone_params']
        backbone: Backbone = backbone_class(
            pretrained=backbone_params['pretrained'],
            num_frozen_levels=backbone_params['num_frozen_levels']
        )

        algorithm_class = checkpoint_dict['algorithm_class']
        algorithm_params = checkpoint_dict['algorithm_params']
        algorithm: Algorithm = algorithm_class(
            num_classes,
            backbone,
            anchor_ratios=algorithm_params['anchor_ratios'],
            anchor_sizes=algorithm_params['anchor_sizes'],
            train_rpn_pre_nms_top_n=algorithm_params['train_rpn_pre_nms_top_n'],
            train_rpn_post_nms_top_n=algorithm_params['train_rpn_post_nms_top_n'],
            eval_rpn_pre_nms_top_n=algorithm_params['eval_rpn_pre_nms_top_n'],
            eval_rpn_post_nms_top_n=algorithm_params['eval_rpn_post_nms_top_n'],
            num_anchor_samples_per_batch=algorithm_params['num_anchor_samples_per_batch'],
            num_proposal_samples_per_batch=algorithm_params['num_proposal_samples_per_batch'],
            num_detections_per_image=algorithm_params['num_detections_per_image'],
            anchor_smooth_l1_loss_beta=algorithm_params['anchor_smooth_l1_loss_beta'],
            proposal_smooth_l1_loss_beta=algorithm_params['proposal_smooth_l1_loss_beta'],
            proposal_nms_threshold=algorithm_params['proposal_nms_threshold'],
            detection_nms_threshold=algorithm_params['detection_nms_threshold']
        )

        model = Model(algorithm,
                      num_classes,
                      preprocessor=checkpoint_dict['preprocessor'],
                      class_to_category_dict=checkpoint_dict['class_to_category_dict'],
                      category_to_class_dict=checkpoint_dict['category_to_class_dict'])
        model.load_state_dict(checkpoint_dict['model_state_dict'])
        model.to(device)

        checkpoint = Checkpoint(epoch=checkpoint_dict['epoch'],
                                model=model,
                                optimizer=checkpoint_dict['optimizer'])
        return checkpoint
