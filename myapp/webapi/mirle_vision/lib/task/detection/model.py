from typing import Union, Tuple, Dict

import torch
from torch.nn import functional as F

from .algorithm import Algorithm
from .preprocessor import Preprocessor
from ...extension.data_parallel import Bunch
from ...extension.functional import normalize_means_stds
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
            gt_bboxes_batch: Bunch = None, gt_classes_batch: Bunch = None
    ) -> Union[Tuple[Bunch, Bunch, Bunch, Bunch],
               Tuple[Bunch, Bunch, Bunch, Bunch, Bunch, Bunch]]:
        batch_size = len(image_batch)

        padded_image_width = max([it.shape[2] for it in image_batch])
        padded_image_height = max([it.shape[1] for it in image_batch])

        padded_image_batch = []
        for image in image_batch:
            padded_image = F.pad(input=image, pad=[0, padded_image_width - image.shape[2], 0, padded_image_height - image.shape[1]])  # pad has format [left, right, top, bottom]
            padded_image_batch.append(padded_image)
        padded_image_batch = torch.stack(padded_image_batch, dim=0)

        padded_image_batch = normalize_means_stds(padded_image_batch,
                                                  list(self.algorithm.backbone.normalization_means()),
                                                  list(self.algorithm.backbone.normalization_stds()))

        if self.training:
            (anchor_objectness_loss_batch, anchor_transformer_loss_batch,
             proposal_class_loss_batch, proposal_transformer_loss_batch) = \
                self.algorithm.forward(padded_image_batch,
                                       gt_bboxes_batch, gt_classes_batch)

            anchor_objectness_loss_batch = Bunch(anchor_objectness_loss_batch.unbind(dim=0))
            anchor_transformer_loss_batch = Bunch(anchor_transformer_loss_batch.unbind(dim=0))
            proposal_class_loss_batch = Bunch(proposal_class_loss_batch.unbind(dim=0))
            proposal_transformer_loss_batch = Bunch(proposal_transformer_loss_batch.unbind(dim=0))

            return (anchor_objectness_loss_batch, anchor_transformer_loss_batch,
                    proposal_class_loss_batch, proposal_transformer_loss_batch)
        else:
            anchor_bboxes_batch, proposal_bboxes_batch, proposal_probs_batch, \
                detection_bboxes_batch, detection_classes_batch, detection_probs_batch = \
                self.algorithm.forward(padded_image_batch)

            anchor_bboxes_batch = Bunch(anchor_bboxes_batch)
            proposal_bboxes_batch = Bunch(proposal_bboxes_batch)
            proposal_probs_batch = Bunch(proposal_probs_batch)
            detection_bboxes_batch = Bunch(detection_bboxes_batch)
            detection_classes_batch = Bunch(detection_classes_batch)
            detection_probs_batch = Bunch(detection_probs_batch)

            return (anchor_bboxes_batch, proposal_bboxes_batch, proposal_probs_batch,
                    detection_bboxes_batch, detection_classes_batch, detection_probs_batch)
