from typing import Union, Tuple

import torchvision
from graphviz import Digraph
from torch import nn, Tensor
from torch.nn import functional as F

from . import Algorithm


class ResNet101(Algorithm):

    def __init__(self, num_classes: int,
                 pretrained: bool, num_frozen_levels: int,
                 eval_center_crop_ratio: float):
        super().__init__(num_classes,
                         pretrained, num_frozen_levels,
                         eval_center_crop_ratio)

    def _build_net(self) -> nn.Module:
        resnet101 = torchvision.models.resnet101(pretrained=self.pretrained)
        resnet101.fc = nn.Linear(in_features=resnet101.fc.in_features, out_features=self.num_classes)

        # list(resnet101.children()) consists of following modules
        #   [0] = Conv2d, [1] = BatchNorm2d, [2] = ReLU,
        #   [3] = MaxPool2d, [4] = Sequential(Bottleneck...),
        #   [5] = Sequential(Bottleneck...),
        #   [6] = Sequential(Bottleneck...),
        #   [7] = Sequential(Bottleneck...),
        #   [8] = AvgPool2d, [9] = Linear
        children = list(resnet101.children())

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

        return resnet101

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

    def make_graph(self) -> Tuple[Digraph, str, str]:
        graph = Digraph()

        # x = self.conv1(x)
        name = 'net.conv1'
        label = [name]
        for key, param in dict(self.net.conv1.named_parameters()).items():
            label.append(f'{key} {str(tuple(param.shape))}')
        label = '\n'.join(label)
        graph.node(name=name, label=label, shape='box', style='filled', fillcolor='#ffffff')
        graph.edge('net.conv1', 'net.bn1')

        # x = self.bn1(x)
        name = 'net.bn1'
        label = [name]
        for key, param in dict(self.net.bn1.named_parameters()).items():
            label.append(f'{key} {str(tuple(param.shape))}')
        label = '\n'.join(label)
        graph.node(name=name, label=label, shape='box', style='filled', fillcolor='#ffffff')
        graph.edge('net.bn1', 'net.relu')

        # x = self.relu(x)
        graph.edge('net.relu', 'net.maxpool')

        # x = self.maxpool(x)
        graph.edge('net.maxpool', 'net.layer1.0.conv1')

        # x = self.layer1(x)
        with graph.subgraph(name='cluster_layer1') as c:
            c.attr(label='net.layer1', style='filled', color='orange')

            name = 'net.layer1.0.conv1'
            label = [name]
            for key, param in dict(self.net.layer1[0].conv1.named_parameters()).items():
                label.append(f'{key} {str(tuple(param.shape))}')
            label = '\n'.join(label)
            c.node(name=name, label=label, shape='box', style='filled', fillcolor='#ffffff')
            c.edge('net.layer1.0.conv1', 'net.layer1.2.conv3', style='dashed')

            name = 'net.layer1.2.conv3'
            label = [name]
            for key, param in dict(self.net.layer1[2].conv3.named_parameters()).items():
                label.append(f'{key} {str(tuple(param.shape))}')
            label = '\n'.join(label)
            c.node(name=name, label=label, shape='box', style='filled', fillcolor='#ffffff')
        graph.edge('net.layer1.2.conv3', 'net.layer2.0.conv1')

        # x = self.layer2(x)
        with graph.subgraph(name='cluster_layer2') as c:
            c.attr(label='net.layer2', style='filled', color='orange')

            name = 'net.layer2.0.conv1'
            label = [name]
            for key, param in dict(self.net.layer2[0].conv1.named_parameters()).items():
                label.append(f'{key} {str(tuple(param.shape))}')
            label = '\n'.join(label)
            c.node(name=name, label=label, shape='box', style='filled', fillcolor='#ffffff')
            c.edge('net.layer2.0.conv1', 'net.layer2.3.conv3', style='dashed')

            name = 'net.layer2.3.conv3'
            label = [name]
            for key, param in dict(self.net.layer2[3].conv3.named_parameters()).items():
                label.append(f'{key} {str(tuple(param.shape))}')
            label = '\n'.join(label)
            c.node(name=name, label=label, shape='box', style='filled', fillcolor='#ffffff')
        graph.edge('net.layer2.3.conv3', 'net.layer3.0.conv1')

        # x = self.layer3(x)
        with graph.subgraph(name='cluster_layer3') as c:
            c.attr(label='net.layer3', style='filled', color='orange')

            name = 'net.layer3.0.conv1'
            label = [name]
            for key, param in dict(self.net.layer3[0].conv1.named_parameters()).items():
                label.append(f'{key} {str(tuple(param.shape))}')
            label = '\n'.join(label)
            c.node(name=name, label=label, shape='box', style='filled', fillcolor='#ffffff')
            c.edge('net.layer3.0.conv1', 'net.layer3.22.conv3', style='dashed')

            name = 'net.layer3.22.conv3'
            label = [name]
            for key, param in dict(self.net.layer3[22].conv3.named_parameters()).items():
                label.append(f'{key} {str(tuple(param.shape))}')
            label = '\n'.join(label)
            c.node(name=name, label=label, shape='box', style='filled', fillcolor='#ffffff')
        graph.edge('net.layer3.22.conv3', 'net.layer4.0.conv1')

        # x = self.layer4(x)
        with graph.subgraph(name='cluster_layer4') as c:
            c.attr(label='net.layer4', style='filled', color='orange')

            name = 'net.layer4.0.conv1'
            label = [name]
            for key, param in dict(self.net.layer4[0].conv1.named_parameters()).items():
                label.append(f'{key} {str(tuple(param.shape))}')
            label = '\n'.join(label)
            c.node(name=name, label=label, shape='box', style='filled', fillcolor='#ffffff')
            c.edge('net.layer4.0.conv1', 'net.layer4.2.conv3', style='dashed')

            name = 'net.layer4.2.conv3'
            label = [name]
            for key, param in dict(self.net.layer4[2].conv3.named_parameters()).items():
                label.append(f'{key} {str(tuple(param.shape))}')
            label = '\n'.join(label)
            c.node(name=name, label=label, shape='box', style='filled', fillcolor='#ffffff')
        graph.edge('net.layer4.2.conv3', 'net.avgpool')

        # x = self.avgpool(x)
        graph.edge('net.avgpool', 'net.flatten')

        # x = torch.flatten(x, 1)
        graph.edge('net.flatten', 'net.fc')

        # x = self.fc(x)
        name = 'net.fc'
        label = [name]
        for key, param in dict(self.net.fc.named_parameters()).items():
            label.append(f'{key} {str(tuple(param.shape))}')
        label = '\n'.join(label)
        graph.node(name=name, label=label, shape='box', style='filled', fillcolor='#ffffff')

        input_node_name = 'net.conv1'
        output_node_name = 'net.fc'
        return graph, input_node_name, output_node_name

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
