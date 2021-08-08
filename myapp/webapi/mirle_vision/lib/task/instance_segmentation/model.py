from typing import Union, Tuple, Dict

import torch
from torch.nn import functional as F

from .algorithm import Algorithm
from .preprocessor import Preprocessor
from ...extension.data_parallel import Bunch
from ...model import Model as Base


class Model(Base):

    def __init__(self, algorithm: Algorithm, num_classes: int, preprocessor: Preprocessor,
                 class_to_category_dict: Dict[int, str], category_to_class_dict: Dict[str, int]):
        super().__init__()
        self.algorithm = algorithm
        self.num_classes = num_classes
        self.preprocessor = preprocessor
        self.class_to_category_dict = class_to_category_dict
        self.category_to_class_dict = category_to_class_dict

    def forward(
            self,
            image_batch: Bunch,
            gt_bboxes_batch: Bunch = None, gt_masks_batch: Bunch = None, gt_classes_batch: Bunch = None
    ) -> Union[Tuple[Bunch, Bunch, Bunch, Bunch, Bunch],
               Tuple[Bunch, Bunch, Bunch, Bunch]]:
        batch_size = len(image_batch)

        padded_image_width = max([it.shape[2] for it in image_batch])
        padded_image_height = max([it.shape[1] for it in image_batch])

        padded_image_batch = []
        for image in image_batch:
            padded_image = F.pad(input=image, pad=[0, padded_image_width - image.shape[2], 0, padded_image_height - image.shape[1]])  # pad has format [left, right, top, bottom]
            padded_image_batch.append(padded_image)
        padded_image_batch = torch.stack(padded_image_batch, dim=0)

        if self.training:
            padded_gt_masks_width = max([it.shape[2] for it in gt_masks_batch])
            padded_gt_masks_height = max([it.shape[1] for it in gt_masks_batch])

            assert padded_gt_masks_width == padded_image_width
            assert padded_gt_masks_height == padded_image_height

            padded_gt_masks_batch = []
            for gt_masks in gt_masks_batch:
                padded_gt_mask = F.pad(input=gt_masks, pad=[0, padded_gt_masks_width - gt_masks.shape[2], 0, padded_gt_masks_height - gt_masks.shape[1]])  # pad has format [left, right, top, bottom]
                padded_gt_masks_batch.append(padded_gt_mask)

            (anchor_objectness_loss_batch, anchor_transformer_loss_batch,
             proposal_class_loss_batch, proposal_transformer_loss_batch,
             mask_loss_batch) = \
                self.algorithm.forward(padded_image_batch,
                                       gt_bboxes_batch, padded_gt_masks_batch, gt_classes_batch)

            anchor_objectness_loss_batch = Bunch([anchor_objectness_loss_batch])
            anchor_transformer_loss_batch = Bunch([anchor_transformer_loss_batch])
            proposal_class_loss_batch = Bunch([proposal_class_loss_batch])
            proposal_transformer_loss_batch = Bunch([proposal_transformer_loss_batch])
            mask_loss_batch = Bunch([mask_loss_batch])

            return (anchor_objectness_loss_batch, anchor_transformer_loss_batch,
                    proposal_class_loss_batch, proposal_transformer_loss_batch,
                    mask_loss_batch)
        else:
            detection_bboxes_batch, detection_classes_batch, detection_probs_batch, detection_masks_batch = \
                self.algorithm.forward(padded_image_batch)

            detection_bboxes_batch = Bunch(detection_bboxes_batch)
            detection_classes_batch = Bunch(detection_classes_batch)
            detection_probs_batch = Bunch(detection_probs_batch)
            detection_masks_batch = Bunch(detection_masks_batch)

            return detection_bboxes_batch, detection_classes_batch, detection_probs_batch, detection_masks_batch
