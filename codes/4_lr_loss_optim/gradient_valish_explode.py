# -*- coding: utf-8 -*-

'''
@Author   :   Corley Tang
@Contact  :   cutercorleytd@gmail.com
@Github   :   https://github.com/corleytd
@Time     :   2022-11-12 21:59
@Project  :   PyTorchBasic-gradient_valish_explode
'''

import torch
from torch import nn

from tools.common_tools import set_seed

set_seed()


class MLP(nn.Module):
    def __init__(self, neural_num, layers):
        super().__init__()
        self.layers = nn.ModuleList([nn.Linear(neural_num, neural_num, bias=False) for _ in range(layers)])
        self.neural_num = neural_num

    def forward(self, x):
        for i, linear in enumerate(self.layers, 1):
            x = linear(x)
            # x = torch.tanh(x)
            x = torch.relu(x)

            print(f'layer: {i}, std: {x.std()}')
            if torch.isnan(x.std()):
                print(f'output is nan in {i} layer')
                break

        return x

    def initalize(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                # nn.init.normal_(m.weight.data) # normal: mean=0, std=1

                # nn.init.normal_(m.weight.data, std=1/pow(self.neural_num, 0.5)) # normal: mean=0, std=1/sqr(n)

                # a = pow(6 / (self.neural_num + self.neural_num), 0.5)
                # tanh_gain = nn.init.calculate_gain('tanh')
                # a *= tanh_gain
                # nn.init.uniform_(m.weight.data, -a, a)

                # nn.init.xavier_uniform_(m.weight.data, gain=nn.init.calculate_gain('tanh'))

                # nn.init.normal_(m.weight.data, std=pow(2 / self.neural_num, 0.5))

                nn.init.kaiming_normal_(m.weight.data)


flag = 1
if flag == 0:
    layer_num = 100
    neural_num = 256
    batch_size = 16

    mlp = MLP(neural_num, layer_num)
    mlp.initalize()

    inputs = torch.randn((batch_size, neural_num))

    outputs = mlp(inputs)
    print(outputs)
else:
    x = torch.randn(10000)
    out = torch.tanh(x)

    gain = x.std() / out.std()
    print(f'gain: {gain}')

    torch_gain = nn.init.calculate_gain('tanh')
    print(f'gain in torch: {torch_gain}')
