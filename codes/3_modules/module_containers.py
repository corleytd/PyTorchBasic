# -*- coding: utf-8 -*-

'''
@Author   :   Corley Tang
@Contact  :   cutercorleytd@gmail.com
@Github   :   https://github.com/corleytd
@Time     :   2022-11-07 11:19
@Project  :   PyTorchBasic-module_containers
'''

from collections import OrderedDict

import torch
import torchvision
from torch import nn


# 1.Sequential
class LeNetSequential(nn.Module):
    def __init__(self, classes):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 6, 5),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(6, 16, 5),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.classifier = nn.Sequential(
            nn.Linear(16 * 5 * 5, 120),
            nn.ReLU(),
            nn.Linear(120, 84),
            nn.ReLU(),
            nn.Linear(84, classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size()[0], -1)
        x = self.classifier(x)
        return x


class LeNetSequentialOrderedDict(nn.Module):
    def __init__(self, classes):
        super().__init__()
        self.features = nn.Sequential(OrderedDict({
            'conv1': nn.Conv2d(3, 6, 5),
            'relu1': nn.ReLU(),
            'pool1': nn.MaxPool2d(kernel_size=2, stride=2),
            'conv2': nn.Conv2d(6, 16, 5),
            'relu2': nn.ReLU(),
            'pool2': nn.MaxPool2d(kernel_size=2, stride=2)
        })
        )
        self.classifier = nn.Sequential(OrderedDict({
            'fc1': nn.Linear(16 * 5 * 5, 120),
            'relu3': nn.ReLU(),
            'fc2': nn.Linear(120, 84),
            'relu4': nn.ReLU(),
            'fc3': nn.Linear(84, classes)
        })
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size()[0], -1)
        x = self.classifier(x)
        return x


# lenet = LeNetSequential(classes=2)
# lenet = LeNetSequentialOrderedDict(classes=2)
# fake_img = torch.randn((4, 3, 32, 32), dtype=torch.float32)
# output = lenet(fake_img)
# print(lenet)
# print(output)


# 2.ModuleList
class ModuleList(nn.ModuleList):
    def __init__(self):
        super().__init__()
        self.linears = nn.ModuleList([nn.Linear(10, 10) for _ in range(20)])

    def forward(self, x):
        for linear in self.linears:
            x = linear(x)
        return x


# net = ModuleList()
# fake_data = torch.ones((10, 10))
# output = net(fake_data)
# print(net)
# print(output)

# 3.ModuleDict
class ModuleDict(nn.Module):
    def __init__(self):
        super().__init__()
        self.choices = nn.ModuleDict({
            'conv': nn.Conv2d(10, 10, 3),
            'pool': nn.MaxPool2d(3)
        })
        self.activations = nn.ModuleDict({
            'relu': nn.ReLU(),
            'prelu': nn.PReLU()
        })

    def forward(self, x, choice, act):
        x = self.choices[choice](x)
        x = self.activations[act](x)
        return x


# net = ModuleDict()
# fake_img = torch.randn((4, 10, 32, 32))
# output = net(fake_img, 'conv', 'relu')
# print(output)
# output = net(fake_img, 'conv', 'prelu')
# print(output)

# 4.AlexNet
class AlexNet(nn.Module):

    def __init__(self, num_classes: int = 1000) -> None:
        super(AlexNet, self).__init__()
        self.features = nn.Sequential(OrderedDict({
            'conv1': nn.Conv2d(3, 64, kernel_size=11, stride=4, padding=2),
            'relu1': nn.ReLU(inplace=True),
            'pool1': nn.MaxPool2d(kernel_size=3, stride=2),
            'conv2': nn.Conv2d(64, 192, kernel_size=5, padding=2),
            'relu2': nn.ReLU(inplace=True),
            'pool2': nn.MaxPool2d(kernel_size=3, stride=2),
            'conv3': nn.Conv2d(192, 384, kernel_size=3, padding=1),
            'relu3': nn.ReLU(inplace=True),
            'conv4': nn.Conv2d(384, 256, kernel_size=3, padding=1),
            'relu4': nn.ReLU(inplace=True),
            'conv5': nn.Conv2d(256, 256, kernel_size=3, padding=1),
            'relu5': nn.ReLU(inplace=True),
            'pool3': nn.MaxPool2d(kernel_size=3, stride=2)
        }))
        self.avgpool = nn.AdaptiveAvgPool2d((6, 6))
        self.classifier = nn.Sequential(
            nn.Dropout(),
            nn.Linear(256 * 6 * 6, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x


alexnet = torchvision.models.AlexNet()
print(alexnet._modules['features']._modules.keys())
alexnet = AlexNet()
print(alexnet._modules['features']._modules.keys())
