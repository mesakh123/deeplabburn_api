from typing import Union, Tuple

import torchvision
from torch import nn, Tensor
from torch.nn import functional as F

from . import Algorithm


class Inception_v3(Algorithm):

    def __init__(self, num_classes: int,
                 pretrained: bool, num_frozen_levels: int,
                 eval_center_crop_ratio: float):
        super().__init__(num_classes,
                         pretrained, num_frozen_levels,
                         eval_center_crop_ratio)

    def _build_net(self) -> nn.Module:
        inception_v3 = torchvision.models.inception_v3(pretrained=self.pretrained, aux_logits=False)
        inception_v3.fc = nn.Linear(in_features=inception_v3.fc.in_features, out_features=self.num_classes)

        conv1 = nn.Sequential(
            inception_v3.Conv2d_1a_3x3,
            inception_v3.Conv2d_2a_3x3,
            inception_v3.Conv2d_2b_3x3
        )
        conv2 = nn.Sequential(
            inception_v3.maxpool1,
            inception_v3.Conv2d_3b_1x1,
            inception_v3.Conv2d_4a_3x3
        )
        conv3 = nn.Sequential(
            inception_v3.maxpool2,
            inception_v3.Mixed_5b,
            inception_v3.Mixed_5c,
            inception_v3.Mixed_5d
        )
        conv4 = nn.Sequential(
            inception_v3.Mixed_6a,
            inception_v3.Mixed_6b,
            inception_v3.Mixed_6c,
            inception_v3.Mixed_6d,
            inception_v3.Mixed_6e
        )
        conv5 = nn.Sequential(
            inception_v3.Mixed_7a,
            inception_v3.Mixed_7b,
            inception_v3.Mixed_7c
        )

        modules = [conv1, conv2, conv3, conv4, conv5]
        assert 0 <= self.num_frozen_levels <= len(modules)

        freezing_modules = modules[:self.num_frozen_levels]

        for module in freezing_modules:
            module.eval()
            for parameter in module.parameters():
                parameter.requires_grad = False

        return inception_v3

    def forward(
            self, padded_image_batch: Tensor, gt_classes_batch: Tensor = None
    ) -> Union[Tensor,
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
        return self.net.Mixed_7c

    @staticmethod
    def normalization_means() -> Tuple[float, float, float]:
        return 0.485, 0.456, 0.406

    @staticmethod
    def normalization_stds() -> Tuple[float, float, float]:
        return 0.229, 0.224, 0.225
