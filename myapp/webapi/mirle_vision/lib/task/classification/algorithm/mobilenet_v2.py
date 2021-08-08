from typing import Union, Tuple

import torchvision
from torch import nn, Tensor
from torch.nn import functional as F

from . import Algorithm


class MobileNet_v2(Algorithm):

    def __init__(self, num_classes: int,
                 pretrained: bool, num_frozen_levels: int,
                 eval_center_crop_ratio: float):
        super().__init__(num_classes,
                         pretrained, num_frozen_levels,
                         eval_center_crop_ratio)

    def _build_net(self) -> nn.Module:
        mobilenet_v2 = torchvision.models.mobilenet_v2(pretrained=self.pretrained)
        mobilenet_v2.classifier = nn.Sequential(
            mobilenet_v2.classifier[0],
            nn.Linear(in_features=mobilenet_v2.classifier[1].in_features, out_features=self.num_classes)
        )

        # x = torch.randn(1, 3, 224, 224)
        # x = mobilenet_v2.features[0](x)           # (1, 32, 112, 112)
        # x = mobilenet_v2.features[1](x)           # (1, 16, 112, 112)
        # x = mobilenet_v2.features[2](x)           # (1, 24, 56, 56)
        # x = mobilenet_v2.features[3](x)           # (1, 24, 56, 56)
        # x = mobilenet_v2.features[4](x)           # (1, 32, 28, 28)
        # x = mobilenet_v2.features[5](x)           # (1, 32, 28, 28)
        # x = mobilenet_v2.features[6](x)           # (1, 32, 28, 28)
        # x = mobilenet_v2.features[7](x)           # (1, 64, 14, 14)
        # x = mobilenet_v2.features[8](x)           # (1, 64, 14, 14)
        # x = mobilenet_v2.features[9](x)           # (1, 64, 14, 14)
        # x = mobilenet_v2.features[10](x)          # (1, 64, 14, 14)
        # x = mobilenet_v2.features[11](x)          # (1, 96, 14, 14)
        # x = mobilenet_v2.features[12](x)          # (1, 96, 14, 14)
        # x = mobilenet_v2.features[13](x)          # (1, 96, 14, 14)
        # x = mobilenet_v2.features[14](x)          # (1, 160, 7, 7)
        # x = mobilenet_v2.features[15](x)          # (1, 160, 7, 7)
        # x = mobilenet_v2.features[16](x)          # (1, 160, 7, 7)
        # x = mobilenet_v2.features[17](x)          # (1, 320, 7, 7)
        # x = mobilenet_v2.features[18](x)          # (1, 1280, 7, 7)
        children = mobilenet_v2.features

        conv1 = children[:2]
        conv2 = children[2:4]
        conv3 = children[4:7]
        conv4 = children[7:14]
        conv5 = children[14:]

        modules = [conv1, conv2, conv3, conv4, conv5]
        assert 0 <= self.num_frozen_levels <= len(modules)

        freezing_modules = modules[:self.num_frozen_levels]

        for module in freezing_modules:
            module.eval()
            for parameter in module.parameters():
                parameter.requires_grad = False

        return mobilenet_v2

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
        del self.net.classifier

    @property
    def output_module_weight(self) -> Tensor:
        return self.net.classifier[1].weight.detach()

    @property
    def last_features_module(self) -> nn.Module:
        return self.net.features[-1]

    @staticmethod
    def normalization_means() -> Tuple[float, float, float]:
        return 0.485, 0.456, 0.406

    @staticmethod
    def normalization_stds() -> Tuple[float, float, float]:
        return 0.229, 0.224, 0.225
