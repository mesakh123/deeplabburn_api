from typing import Union, Tuple

import torch
import torchvision
from torch import nn, Tensor
from torch.nn import functional as F

from . import Algorithm
from ....extension.functional import crop_and_resize


class RACNN(Algorithm):

    def __init__(self, num_classes: int,
                 pretrained: bool, num_frozen_levels: int,
                 roi_resized_width: int, roi_resized_height: int):
        super().__init__(num_classes,
                         pretrained, num_frozen_levels,
                         roi_resized_width, roi_resized_height)
        self.backbone1, num_out_channels1 = self._build_backbone()
        self.backbone2, num_out_channels2 = self._build_backbone()
        self.backbone3, num_out_channels3 = self._build_backbone()

        self.classifier1 = nn.Linear(in_features=num_out_channels1, out_features=num_classes)
        self.classifier2 = nn.Linear(in_features=num_out_channels2, out_features=num_classes)
        self.classifier3 = nn.Linear(in_features=num_out_channels3, out_features=num_classes)

        # attention proposal network outputs center x, y and square length in range [0, 1]
        self.apn1 = nn.Sequential(
            nn.Linear(in_features=num_out_channels1 * 14 * 14, out_features=1024),
            nn.Tanh(),
            nn.Linear(in_features=1024, out_features=3),
            nn.Sigmoid()
        )
        self.apn2 = nn.Sequential(
            nn.Linear(in_features=num_out_channels2 * 7 * 7, out_features=1024),
            nn.Tanh(),
            nn.Linear(in_features=1024, out_features=3),
            nn.Sigmoid()
        )

        self.needs_ensemble_prediction = False

    def _build_backbone(self) -> Tuple[nn.Module, int]:
        # resnet50 = torchvision.models.resnet50(pretrained=self.pretrained)
        resnet50 = torchvision.models.resnet18(pretrained=self.pretrained)  # TODO

        # list(resnet50.children()) consists of following modules
        #   [0] = Conv2d, [1] = BatchNorm2d, [2] = ReLU,
        #   [3] = MaxPool2d, [4] = Sequential(Bottleneck...),
        #   [5] = Sequential(Bottleneck...),
        #   [6] = Sequential(Bottleneck...),
        #   [7] = Sequential(Bottleneck...),
        #   [8] = AvgPool2d, [9] = Linear
        children = list(resnet50.children())

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

        backbone = nn.Sequential(
            conv1,
            conv2,
            conv3,
            conv4,
            conv5
        )
        # num_out_channels = 2048
        num_out_channels = 512  # TODO
        return backbone, num_out_channels

    def forward(self,
                padded_image_batch: Tensor,
                gt_classes_batch: Tensor = None) -> Union[Tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor],
                                                          Tuple[Tensor, Tensor, Tensor, Tensor]]:
        batch_size, _, padded_image_height, padded_image_width = padded_image_batch.shape

        # region ===== First stage =====
        features1_batch = self.backbone1.forward(padded_image_batch)
        pools1_batch = F.adaptive_avg_pool2d(input=features1_batch, output_size=(1, 1))
        logit1_batch = self.classifier1.forward(pools1_batch.flatten(start_dim=1))

        # TODO3
        if not self.apn1.training:
            features1_batch = features1_batch.detach()

        attention1_batch = self.apn1.forward(features1_batch.flatten(start_dim=1))
        center_x_norm_batch = attention1_batch[:, 0]
        center_y_norm_batch = attention1_batch[:, 1]
        square_l_norm_batch = attention1_batch[:, 2].clamp(min=0.6)

        resized_roi1_batch = crop_and_resize(padded_image_batch,
                                             center_x_norm_batch,
                                             center_y_norm_batch,
                                             square_l_norm_batch,
                                             square_l_norm_batch,
                                             resized_width=self.roi_resized_width,
                                             resized_height=self.roi_resized_height)
        # endregion ====================

        # region ===== Second stage =====
        features2_batch = self.backbone2.forward(resized_roi1_batch)
        pools2_batch = F.adaptive_avg_pool2d(input=features2_batch, output_size=(1, 1))
        logit2_batch = self.classifier2.forward(pools2_batch.flatten(start_dim=1))

        # TODO3
        if not self.apn2.training:
            features2_batch = features2_batch.detach()

        attention2_batch = self.apn2.forward(features2_batch.flatten(start_dim=1))
        center_x_norm_batch = attention2_batch[:, 0]
        center_y_norm_batch = attention2_batch[:, 1]
        square_l_norm_batch = attention2_batch[:, 2].clamp(min=0.6)

        resized_roi2_batch = crop_and_resize(resized_roi1_batch,
                                             center_x_norm_batch,
                                             center_y_norm_batch,
                                             square_l_norm_batch,
                                             square_l_norm_batch,
                                             resized_width=self.roi_resized_width,
                                             resized_height=self.roi_resized_height)
        # endregion =====================

        # region ===== Third stage =====
        features3_batch = self.backbone3.forward(resized_roi2_batch)
        pools3_batch = F.adaptive_avg_pool2d(input=features3_batch, output_size=(1, 1))
        logit3_batch = self.classifier3.forward(pools3_batch.flatten(start_dim=1))
        # endregion ====================

        if self.training:
            entropy_loss1_batch, entropy_loss2_batch, entropy_loss3_batch, rank_loss1_batch, rank_loss2_batch = \
                self.loss(logit1_batch, logit2_batch, logit3_batch, gt_classes_batch)
            return entropy_loss1_batch, entropy_loss2_batch, entropy_loss3_batch, rank_loss1_batch, rank_loss2_batch, resized_roi1_batch, resized_roi2_batch
        else:
            if self.needs_ensemble_prediction:
                logit_batch = (logit1_batch + logit2_batch + logit3_batch).mean(dim=0, keepdim=True)
            else:
                logit_batch = logit2_batch
            pred_prob_batch, pred_class_batch = F.softmax(input=logit_batch, dim=1).max(dim=1)
            return pred_prob_batch, pred_class_batch, resized_roi1_batch, resized_roi2_batch

    def loss(self, logit1_batch: Tensor, logit2_batch: Tensor, logit3_batch: Tensor,
             gt_classes_batch: Tensor) -> Tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
        entropy_loss1_batch = F.cross_entropy(input=logit1_batch, target=gt_classes_batch, reduction='none')
        entropy_loss2_batch = F.cross_entropy(input=logit2_batch, target=gt_classes_batch, reduction='none')
        entropy_loss3_batch = F.cross_entropy(input=logit3_batch, target=gt_classes_batch, reduction='none')

        prob1_batch = F.softmax(input=logit1_batch, dim=1)
        prob2_batch = F.softmax(input=logit2_batch, dim=1)
        prob3_batch = F.softmax(input=logit3_batch, dim=1)

        target_prob1_batch = prob1_batch[torch.arange(start=0, end=prob1_batch.shape[0]), gt_classes_batch.unbind(dim=0)]
        target_prob2_batch = prob2_batch[torch.arange(start=0, end=prob2_batch.shape[0]), gt_classes_batch.unbind(dim=0)]
        target_prob3_batch = prob3_batch[torch.arange(start=0, end=prob3_batch.shape[0]), gt_classes_batch.unbind(dim=0)]

        # specify target to -1 to encourage that `input2` be greater that `input1`
        rank_loss1_batch = F.margin_ranking_loss(input1=target_prob1_batch, input2=target_prob2_batch, margin=0.05,
                                                 target=torch.ones_like(target_prob1_batch) * -1,
                                                 reduction='none')
        rank_loss2_batch = F.margin_ranking_loss(input1=target_prob2_batch, input2=target_prob3_batch, margin=0.05,
                                                 target=torch.ones_like(target_prob1_batch) * -1,
                                                 reduction='none')

        return entropy_loss1_batch, entropy_loss2_batch, entropy_loss3_batch, rank_loss1_batch, rank_loss2_batch

    def remove_output_module(self):
        del self.net.fc

    @staticmethod
    def normalization_means() -> Tuple[float, float, float]:
        return 0.485, 0.456, 0.406

    @staticmethod
    def normalization_stds() -> Tuple[float, float, float]:
        return 0.229, 0.224, 0.225
