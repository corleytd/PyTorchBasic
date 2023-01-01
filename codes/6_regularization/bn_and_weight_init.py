# -*- coding: utf-8 -*-

'''
@Author   :   Corley Tang
@Contact  :   cutercorleytd@gmail.com
@Github   :   https://github.com/corleytd
@Time     :   2022-12-10 21:53
@Project  :   PyTorchBasic-bn_and_weight_init
'''

import torch
from torch import nn

from tools.common_tools import set_seed

set_seed()


class MLP(nn.Module):
    def __init__(self, neural_num, layers=100):
        super().__init__()
        self.linears = nn.ModuleList([nn.Linear(neural_num, neural_num, bias=False) for _ in range(layers)])
        self.bns = nn.ModuleList([nn.BatchNorm1d(neural_num) for _ in range(layers)])

    def forward(self, x, use_bn=False):
        for idx, (linear, bn) in enumerate(zip(self.linears, self.bns), 1):
            x = linear(x)
            if use_bn:
                x = bn(x)
            x = torch.relu(x)

            if torch.isnan(x.std()):
                print(f'output is nan in layer {idx}')
                break

            print(f'layer {idx}: std {x.std().item()}')

        return x

    def initialize(self, method='normal'):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                if method == 'normal':
                    nn.init.normal_(module.weight.data)  # mean=0, std=1
                elif method == 'kaiming':
                    nn.init.kaiming_normal_(module.weight.data)


# 超参数
neural_num = 256
layer_num = 100
batch_size = 16

# 构建模型
net = MLP(neural_num, layer_num)

# 构造输入
inputs = torch.randn(batch_size, neural_num)

flag = 4

# 1.不初始化
if flag == 0:
    output = net(inputs)
    print(output)

# 2.使用正态分布初始化
elif flag == 1:
    net.initialize()
    output = net(inputs)
    print(output)

# 3.使用凯明方法初始化
elif flag == 2:
    net.initialize(method='kaiming')
    output = net(inputs)
    print(output)

# 4.加入BN层
elif flag == 3:
    net.initialize(method='kaiming')
    output = net(inputs, use_bn=True)
    print(output)

# 5.只使用BN层，不进行初始化
elif flag == 4:
    output = net(inputs, use_bn=True)
    print(output)
