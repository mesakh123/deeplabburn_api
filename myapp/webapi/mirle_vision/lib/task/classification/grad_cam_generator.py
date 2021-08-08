from typing import List, Tuple

import torch
from torch import Tensor, nn


class GradCAMGenerator:

    def __init__(self, output_module_weight: Tensor, target_layer: nn.Module):
        super().__init__()
        self.output_module_weight = output_module_weight
        self.device_index_to_target_output_batch_dict = {}
        target_layer.register_forward_hook(self._target_hook)

    def _target_hook(self, module: nn.Module, inputs_batch: Tuple[Tensor], output_batch: Tensor):
        self.device_index_to_target_output_batch_dict[output_batch.device.index] = output_batch

    def generate_grad_cam_batch(self, target_class_batch: List[int]) -> List[Tensor]:
        if len(self.device_index_to_target_output_batch_dict) == 0:
            raise ValueError('No output at the target layer, run `forward` before generating Grad-CAM.')

        target_output_batch = [self.device_index_to_target_output_batch_dict[k].to(self.output_module_weight.device)
                               for k in sorted(self.device_index_to_target_output_batch_dict.keys())]
        target_output_batch = torch.cat(target_output_batch, dim=0)

        batch_size = target_output_batch.shape[0]
        if len(target_class_batch) != batch_size:
            raise ValueError('Inconsistent batch size')

        grad_cam_batch = []

        for b in range(batch_size):
            target_class = target_class_batch[b]
            target_output = target_output_batch[b]

            c, h, w = target_output.shape

            # `output_module_weight` has shape of (#classes, N)
            grad_cam = self.output_module_weight[target_class].matmul(target_output.view(c, h * w))
            grad_cam = grad_cam.view(h, w)
            grad_cam -= grad_cam.min()
            grad_cam /= grad_cam.max()
            grad_cam_batch.append(grad_cam)

        return grad_cam_batch
