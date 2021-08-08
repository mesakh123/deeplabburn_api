from typing import Union, Tuple

from efficientnet_pytorch import EfficientNet
from torch import nn, Tensor
from torch.nn import functional as F

from . import Algorithm


class EfficientNet_B0(Algorithm):

    def __init__(self, num_classes: int,
                 pretrained: bool, num_frozen_levels: int,
                 eval_center_crop_ratio: float):
        super().__init__(num_classes,
                         pretrained, num_frozen_levels,
                         eval_center_crop_ratio)

    def _build_net(self) -> nn.Module:
        if self.pretrained:
            efficientnet_b0 = EfficientNet.from_pretrained(model_name='efficientnet-b0')
        else:
            efficientnet_b0 = EfficientNet.from_name(model_name='efficientnet-b0')

        efficientnet_b0._fc = nn.Linear(in_features=efficientnet_b0._fc.in_features, out_features=self.num_classes)

        # x = torch.randn(1, 3, 600, 1000)
        # x = efficientnet_b0._conv_stem(x)        # (1, 32, 300, 500)
        # x = efficientnet_b0._bn0(x)              # (1, 32, 300, 500)
        # x = efficientnet_b0._swish(x)            # (1, 32, 300, 500)
        # x = efficientnet_b0._blocks[0](x, ...)   # (1, 16, 300, 500)
        # x = efficientnet_b0._blocks[1](x, ...)   # [1, 24, 150, 250)
        # x = efficientnet_b0._blocks[2](x, ...)   # (1, 24, 150, 250)
        # x = efficientnet_b0._blocks[3](x, ...)   # (1, 40, 75, 125)
        # x = efficientnet_b0._blocks[4](x, ...)   # (1, 40, 75, 125)
        # x = efficientnet_b0._blocks[5](x, ...)   # (1, 80, 37, 62)
        # x = efficientnet_b0._blocks[6](x, ...)   # (1, 80, 37, 62)
        # x = efficientnet_b0._blocks[7](x, ...)   # (1, 80, 37, 62)
        # x = efficientnet_b0._blocks[8](x, ...)   # (1, 112, 37, 62)
        # x = efficientnet_b0._blocks[9](x, ...)   # (1, 112, 37, 62)
        # x = efficientnet_b0._blocks[10](x, ...)  # (1, 112, 37, 62)
        # x = efficientnet_b0._blocks[11](x, ...)  # (1, 192, 18, 31)
        # x = efficientnet_b0._blocks[12](x, ...)  # (1, 192, 18, 31)
        # x = efficientnet_b0._blocks[13](x, ...)  # (1, 192, 18, 31)
        # x = efficientnet_b0._blocks[14](x, ...)  # (1, 192, 18, 31)
        # x = efficientnet_b0._blocks[15](x, ...)  # (1, 320, 18, 31)
        # x = efficientnet_b0._conv_head(x)        # (1, 1280, 18, 31)
        # x = efficientnet_b0._bn1(x)              # (1, 1280, 18, 31)
        # x = efficientnet_b0._swish(x)            # (1, 1280, 18, 31)

        conv1 = nn.ModuleList([efficientnet_b0._conv_stem] +
                              list(efficientnet_b0._blocks[:1]))
        conv2 = efficientnet_b0._blocks[1:3]
        conv3 = efficientnet_b0._blocks[3:5]
        conv4 = efficientnet_b0._blocks[5:11]
        conv5 = nn.ModuleList(list(efficientnet_b0._blocks[11:]) +
                              [efficientnet_b0._conv_head])

        modules = [conv1, conv2, conv3, conv4, conv5]
        assert 0 <= self.num_frozen_levels <= len(modules)

        freezing_modules = modules[:self.num_frozen_levels]

        for module in freezing_modules:
            module.eval()
            for parameter in module.parameters():
                parameter.requires_grad = False

        return efficientnet_b0

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
        del self.net._fc

    @property
    def output_module_weight(self) -> Tensor:
        return self.net._fc.weight.detach()

    @property
    def last_features_module(self) -> nn.Module:
        return self.net._conv_head

    @staticmethod
    def normalization_means() -> Tuple[float, float, float]:
        return 0.485, 0.456, 0.406

    @staticmethod
    def normalization_stds() -> Tuple[float, float, float]:
        return 0.229, 0.224, 0.225
