# -*- coding: utf-8 -*-

'''
@Author   :   Corley Tang
@Contact  :   cutercorleytd@gmail.com
@Github   :   https://github.com/corleytd
@Time     :   2022-10-31 22:19
@Project  :   PyTorchBasic-lenet
'''

import torch
import torch.nn.functional as F
from torch import nn


def conv_input(ci, co):
    return nn.Conv2d(ci, co, kernel_size=7, stride=1, padding=3)


def conv(ci, co):
    return nn.Conv2d(ci, co, kernel_size=3, stride=1, padding=1)


def conv_half_reduce(ci, co):
    return nn.Conv2d(ci, co, kernel_size=3, stride=2, padding=1)


class LeNet(nn.Module):
    def __init__(self, classes):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, classes)

    def forward(self, x):
        out = F.relu(self.conv1(x))
        out = F.max_pool2d(out, 2)
        out = F.relu(self.conv2(out))
        out = F.max_pool2d(out, 2)
        out = out.view(out.size(0), -1)
        out = F.relu(self.fc1(out))
        out = F.relu(self.fc2(out))
        out = self.fc3(out)
        return out

    def initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_normal_(m.weight.data)
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight.data, 0, 0.1)
                m.bias.data.zero_()


class CatDogLeNet(nn.Module):
    def __init__(self, classes):
        super().__init__()
        self.conv1 = conv_input(3, 16)
        self.bn1 = nn.BatchNorm2d(16)
        self.conv2_1 = conv_half_reduce(16, 32)
        self.bn2_1 = nn.BatchNorm2d(32)
        self.conv2_2 = conv(32, 32)
        self.bn2_2 = nn.BatchNorm2d(32)
        self.conv3_1 = conv_half_reduce(32, 64)
        self.bn3_1 = nn.BatchNorm2d(64)
        self.conv3_2 = conv(64, 64)
        self.bn3_2 = nn.BatchNorm2d(64)
        self.conv4_1 = conv_half_reduce(64, 128)
        self.bn4_1 = nn.BatchNorm2d(128)
        self.conv4_2 = conv(128, 128)
        self.bn4_2 = nn.BatchNorm2d(128)

        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.bn5_avg = nn.BatchNorm2d(128)

        self.max_pool = nn.AdaptiveMaxPool2d((1, 1))
        self.bn5_max = nn.BatchNorm2d(128)

        self.dropout1 = nn.Dropout(0.25)
        self.fc1 = nn.Linear(256, 128)
        self.dropout2 = nn.Dropout(0.5)
        self.fc2 = nn.Linear(128, classes)

    def forward(self, x):
        out = self.conv1(x)
        out = F.relu(self.bn1(out))
        out = self.conv2_1(out)
        out = F.relu(self.bn2_1(out))
        out = self.conv2_2(out)
        out = F.relu(self.bn2_2(out))
        out = self.conv3_1(out)
        out = F.relu(self.bn3_1(out))
        out = self.conv3_2(out)
        out = F.relu(self.bn3_2(out))
        out = self.conv4_1(out)
        out = F.relu(self.bn4_1(out))
        out = self.conv4_2(out)
        out = F.relu(self.bn4_2(out))

        out1 = self.avg_pool(out)
        out1 = self.bn5_avg(out1)
        out2 = self.max_pool(out)
        out2 = self.bn5_max(out2)
        out = torch.cat((out1, out2), dim=1)
        out = out.view(out.size(0), -1)

        out = self.dropout1(out)
        out = F.relu(self.fc1(out))
        out = self.dropout2(out)
        out = self.fc2(out)

        return out

    def initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_normal_(m.weight.data)
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight.data, 0, 0.1)
                m.bias.data.zero_()


class BNLeNet(nn.Module):
    def __init__(self, classes):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.bn1 = nn.BatchNorm2d(num_features=6)

        self.conv2 = nn.Conv2d(6, 16, 5)
        self.bn2 = nn.BatchNorm2d(num_features=16)

        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.bn3 = nn.BatchNorm1d(num_features=120)

        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, classes)

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = F.relu(out)

        out = F.max_pool2d(out, 2)

        out = self.conv2(out)
        out = self.bn2(out)
        out = F.relu(out)

        out = F.max_pool2d(out, 2)

        out = out.view(out.size(0), -1)

        out = self.fc1(out)
        out = self.bn3(out)
        out = F.relu(out)

        out = F.relu(self.fc2(out))
        out = self.fc3(out)

        return out
