from dataclasses import dataclass
from typing import List

import torch
from torch import Tensor

from .grad_cam_generator import GradCAMGenerator
from .model import Model
from ...extension.data_parallel import BunchDataParallel, Bunch
import torch.nn.functional as F


class Inferer:

    @dataclass
    class Inference:
        grad_cam_batch: List[Tensor]
        final_pred_class_batch: List[Tensor]
        final_pred_prob_batch: List[Tensor]

    def __init__(self, model: Model, device_ids: List[int] = None):
        super().__init__()
        self._model = BunchDataParallel(model, device_ids)
        self.grad_cam_generator = GradCAMGenerator(output_module_weight=model.algorithm.output_module_weight,
                                                   target_layer=model.algorithm.last_features_module)

    @torch.no_grad()
    def infer(self, image_batch: List[Tensor],
              lower_prob_thresh: float, upper_prob_thresh: float) -> Inference:
        image_batch = Bunch(image_batch)

        pred_prob_batch, pred_class_batch = \
            self._model.eval().forward(image_batch)

        final_pred_class_batch = []
        for pred_prob, pred_class in zip(pred_prob_batch, pred_class_batch):
            if (pred_prob >= lower_prob_thresh) & (pred_prob <= upper_prob_thresh):
                final_pred_class = pred_class
            else:
                final_pred_class = torch.tensor(0).to(pred_class)
            final_pred_class_batch.append(final_pred_class)

        final_pred_prob_batch = pred_prob_batch

        grad_cam_batch = \
            self.grad_cam_generator.generate_grad_cam_batch(target_class_batch=[it.item() for it in pred_class_batch])

        for b in range(len(grad_cam_batch)):
            grad_cam = grad_cam_batch[b]
            image = image_batch[b]
            grad_cam_batch[b] = F.interpolate(input=grad_cam.view(1, 1, *grad_cam.shape),
                                              size=(image.shape[1], image.shape[2]),
                                              mode='bilinear',
                                              align_corners=True).view(image.shape[1], image.shape[2])

        inference = Inferer.Inference(grad_cam_batch, final_pred_class_batch, final_pred_prob_batch)
        return inference
