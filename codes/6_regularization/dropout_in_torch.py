# -*- coding: utf-8 -*-

'''
@Author   :   Corley Tang
@Contact  :   cutercorleytd@gmail.com
@Github   :   https://github.com/corleytd
@Time     :   2022-12-10 19:09
@Project  :   PyTorchBasic-dropout_in_torch
'''

import torch
from torch import nn

from tools.common_tools import set_seed

set_seed()

# 超参数
input_size = 10000

# 1.生成数据
data = torch.ones(input_size)


# 2.定义模型
class Net(nn.Module):
    def __init__(self, neural_num, dropout=0.5):
        super().__init__()
        self.linears = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(neural_num, 1, bias=False),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.linears(x)


net = Net(input_size, dropout=0.5)
net.linears[1].weight.detach().fill_(1)

# 3.前向传播
net.train()
output = net(data)
print(f'output in train: {output.item()}')

net.eval()
output = net(data)
print(f'output in eval: {output.item()}')
