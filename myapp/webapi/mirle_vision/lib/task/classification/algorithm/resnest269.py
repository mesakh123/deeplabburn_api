from typing import Union, Tuple

import resnest.torch.resnest
from torch import nn, Tensor
from torch.nn import functional as F

from . import Algorithm


class ResNeSt269(Algorithm):

    def __init__(self, num_classes: int,
                 pretrained: bool, num_frozen_levels: int,
                 eval_center_crop_ratio: float):
        super().__init__(num_classes,
                         pretrained, num_frozen_levels,
                         eval_center_crop_ratio)

    def _build_net(self) -> nn.Module:
        resnest269 = resnest.torch.resnest269(pretrained=self.pretrained)
        resnest269.fc = nn.Linear(in_features=resnest269.fc.in_features, out_features=self.num_classes)

        # list(resnest269.children()) consists of following modules
        #   [0] = Sequential(Conv2d...), [1] = BatchNorm2d, [2] = ReLU,
        #   [3] = MaxPool2d, [4] = Sequential(Bottleneck...),
        #   [5] = Sequential(Bottleneck...),
        #   [6] = Sequential(Bottleneck...),
        #   [7] = Sequential(Bottleneck...),
        #   [8] = GlobalAvgPool2d, [9] = Linear
        children = list(resnest269.children())

        conv1 = nn.Sequential(*children[:3])
        conv2 = nn.Sequential(*children[3:5])
        conv3 = children[5]
        conv4 = children[6]
        conv5 = children[7]

        modules = [conv1, conv2, conv3, conv4, conv5]
        assert 0 <= self.num_frozen_levels <= len(modules)

        freezing_modules = modules[:self.num_frozen_levels]

        for module in freezing_modules:
            module.eval()
            for parameter in module.parameters():
                parameter.requires_grad = False

        return resnest269

    def forward(self,
                padded_image_batch: Tensor,
                gt_classes_batch: Tensor = None) -> Union[Tensor,
                                                          Tuple[Tensor, Tensor]]:
        batch_size, _, padded_image_height, padded_image_width = padded_image_batch.shape
        logit_batch = self.net.forward(padded_image_batch)

        if self.training:
            loss_batch = self.loss(logit_batch, gt_classes_batch)
            return loss_batch
        else:
            pred_prob_batch, pred_class_batch = F.softmax(input=logit_batch, dim=1).max(dim=1)
            return pred_prob_batch, pred_class_batch

    def loss(self, logit_batch: Tensor, gt_classes_batch: Tensor) -> Tensor:
        loss_batch = F.cross_entropy(input=logit_batch, target=gt_classes_batch, reduction='none')
        return loss_batch

    def remove_output_module(self):
        del self.net.fc

    @property
    def output_module_weight(self) -> Tensor:
        return self.net.fc.weight.detach()

    @property
    def last_features_module(self) -> nn.Module:
        return self.net.layer4

    @staticmethod
    def normalization_means() -> Tuple[float, float, float]:
        return 0.485, 0.456, 0.406

    @staticmethod
    def normalization_stds() -> Tuple[float, float, float]:
        return 0.229, 0.224, 0.225
