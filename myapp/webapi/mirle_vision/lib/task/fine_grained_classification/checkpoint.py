from dataclasses import dataclass

import torch

from .algorithm import Algorithm
from .model import Model
from ...checkpoint import Checkpoint as Base


@dataclass
class Checkpoint(Base):

    @staticmethod
    def save(checkpoint: 'Checkpoint', path_to_checkpoint: str):
        model = checkpoint.model
        algorithm: Algorithm = model.algorithm

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
                'pretrained': algorithm.pretrained,
                'num_frozen_levels': algorithm.num_frozen_levels,
                'roi_resized_width': algorithm.roi_resized_width,
                'roi_resized_height': algorithm.roi_resized_height
            }
        }
        torch.save(checkpoint_dict, path_to_checkpoint)

    @staticmethod
    def load(path_to_checkpoint: str, device: torch.device) -> 'Checkpoint':
        checkpoint_dict = torch.load(path_to_checkpoint, map_location=device)

        num_classes = checkpoint_dict['num_classes']

        algorithm_class = checkpoint_dict['algorithm_class']
        algorithm_params = checkpoint_dict['algorithm_params']
        algorithm: Algorithm = algorithm_class(
            num_classes,
            pretrained=algorithm_params['pretrained'],
            num_frozen_levels=algorithm_params['num_frozen_levels'],
            roi_resized_width=algorithm_params['roi_resized_width'],
            roi_resized_height=algorithm_params['roi_resized_height']
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
