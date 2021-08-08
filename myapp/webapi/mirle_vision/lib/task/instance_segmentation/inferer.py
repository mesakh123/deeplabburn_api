from dataclasses import dataclass
from typing import List

import torch
from torch import Tensor
from torchvision.ops.boxes import remove_small_boxes

from .model import Model
from ...extension.data_parallel import BunchDataParallel, Bunch


class Inferer:

    @dataclass
    class Inference:
        final_detection_bboxes_batch: List[Tensor]
        final_detection_classes_batch: List[Tensor]
        final_detection_probs_batch: List[Tensor]
        final_detection_probmasks_batch: List[Tensor]

    def __init__(self, model: Model, device_ids: List[int] = None):
        super().__init__()
        self._model = BunchDataParallel(model, device_ids)

    @torch.no_grad()
    def infer(self, image_batch: List[Tensor],
              lower_prob_thresh: float, upper_prob_thresh: float) -> Inference:
        image_batch = Bunch(image_batch)

        (detection_bboxes_batch, detection_classes_batch, detection_probs_batch, detection_probmasks_batch) = \
            self._model.eval().forward(image_batch)

        final_detection_bboxes_batch = []
        final_detection_classes_batch = []
        final_detection_probs_batch = []
        final_detection_probmasks_batch = []
        for detection_bboxes, detection_classes, detection_probs, detection_probmasks in zip(detection_bboxes_batch,
                                                                                             detection_classes_batch,
                                                                                             detection_probs_batch,
                                                                                             detection_probmasks_batch):

            kept_mask = (detection_probs >= lower_prob_thresh) & (detection_probs <= upper_prob_thresh)
            final_detection_bboxes = detection_bboxes[kept_mask]
            final_detection_classes = detection_classes[kept_mask]
            final_detection_probs = detection_probs[kept_mask]
            final_detection_probmasks = detection_probmasks[kept_mask]

            kept_indices = remove_small_boxes(final_detection_bboxes, 1)
            final_detection_bboxes = final_detection_bboxes[kept_indices]
            final_detection_classes = final_detection_classes[kept_indices]
            final_detection_probs = final_detection_probs[kept_indices]
            final_detection_probmasks = final_detection_probmasks[kept_indices]

            final_detection_bboxes_batch.append(final_detection_bboxes)
            final_detection_classes_batch.append(final_detection_classes)
            final_detection_probs_batch.append(final_detection_probs)
            final_detection_probmasks_batch.append(final_detection_probmasks)

        inference = Inferer.Inference(
            final_detection_bboxes_batch,
            final_detection_classes_batch,
            final_detection_probs_batch,
            final_detection_probmasks_batch
        )
        return inference
