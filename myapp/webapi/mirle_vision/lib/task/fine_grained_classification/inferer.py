from dataclasses import dataclass
from typing import List

import torch
from torch import Tensor

from .model import Model
from ...extension.data_parallel import BunchDataParallel, Bunch


class Inferer:

    @dataclass
    class Inference:
        resized_roi1_batch: List[Tensor]
        resized_roi2_batch: List[Tensor]
        final_pred_class_batch: List[Tensor]
        final_pred_prob_batch: List[Tensor]

    def __init__(self, model: Model, device_ids: List[int] = None):
        super().__init__()
        self._model = BunchDataParallel(model, device_ids)

    @torch.no_grad()
    def infer(self, image_batch: List[Tensor],
              lower_prob_thresh: float, upper_prob_thresh: float) -> Inference:
        image_batch = Bunch(image_batch)

        pred_prob_batch, pred_class_batch, resized_roi1_batch, resized_roi2_batch = \
            self._model.eval().forward(image_batch)

        final_pred_class_batch = []
        for pred_prob, pred_class in zip(pred_prob_batch, pred_class_batch):
            if (pred_prob >= lower_prob_thresh) & (pred_prob <= upper_prob_thresh):
                final_pred_class = pred_class
            else:
                final_pred_class = torch.tensor(0).to(pred_class)
            final_pred_class_batch.append(final_pred_class)

        final_pred_prob_batch = pred_prob_batch

        inference = Inferer.Inference(resized_roi1_batch, resized_roi2_batch, final_pred_class_batch, final_pred_prob_batch)
        return inference
