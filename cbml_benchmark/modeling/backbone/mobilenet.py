from __future__ import absolute_import, division, print_function

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from cbml_benchmark.modeling import registry


@registry.BACKBONES.register('mobilenet_v3_small')
class MobileNetV3Small(nn.Module):

    def __init__(self):
        super(MobileNetV3Small, self).__init__()
        self.model = models.mobilenet_v3_small(pretrained=True)

        # freeze BatchNorm during training, same convention as the ResNet backbones
        for module in filter(lambda m: type(m) == nn.BatchNorm2d, self.model.modules()):
            module.eval()
            module.train = lambda _: None

    def forward(self, x):
        x = self.model.features(x)
        x = F.adaptive_avg_pool2d(x, (1, 1))
        x = x.view(x.size(0), -1)
        # output dim: 576 (no classifier head)
        return x

    def load_param(self, model_path):
        param_dict = torch.load(model_path, weights_only=False)
        for i in param_dict:
            if i.startswith('classifier.'):
                continue
            self.model.state_dict()[i].copy_(param_dict[i])


@registry.BACKBONES.register('mobilenet_v2')
class MobileNetV2(nn.Module):

    def __init__(self):
        super(MobileNetV2, self).__init__()
        self.model = models.mobilenet_v2(pretrained=True)

        for module in filter(lambda m: type(m) == nn.BatchNorm2d, self.model.modules()):
            module.eval()
            module.train = lambda _: None

    def forward(self, x):
        x = self.model.features(x)
        x = F.adaptive_avg_pool2d(x, (1, 1))
        x = x.view(x.size(0), -1)
        # output dim: 1280
        return x

    def load_param(self, model_path):
        param_dict = torch.load(model_path, weights_only=False)
        for i in param_dict:
            if i.startswith('classifier.'):
                continue
            self.model.state_dict()[i].copy_(param_dict[i])