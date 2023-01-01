# -*- coding: utf-8 -*-

'''
@Author   :   Corley Tang
@Contact  :   cutercorleytd@gmail.com
@Github   :   https://github.com/corleytd
@Time     :   2022-12-21 16:59
@Project  :   PyTorchBasic-multi_gpus
本实验在RTX 2080 Ti*4上实现
'''

import os

import numpy as np
import torch
from torch import nn

from tools.common_tools import get_gpu_memory

flag = 1


class FooNet(nn.Module):
    def __init__(self, neural_num, layers=3):
        super().__init__()
        self.linears = nn.ModuleList([nn.Linear(neural_num, neural_num, bias=False) for _ in range(layers)])

    def forward(self, x):
        print(f'batch size in forward: {x.size()[0]}\n')

        for linear in self.linears:
            x = linear(x)
            x = torch.relu(x)

        return x


# 1.手动选择GPU
if flag % 2 == 0:
    gpus = [0, 2]
    gpus = ','.join(map(str, gpus))
    os.environ.setdefault('CUDA_VISIBLE_DEVICES', gpus)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('torch.cuda.device_count', torch.cuda.device_count())

'''
torch.cuda.device_count 2
'''

# 2.根据显存情况自动选择主GPU
if flag % 2 == 1:
    gpu_memory = get_gpu_memory()
    if gpu_memory:
        print(f'gpufree memory: {gpu_memory}')
        gpu_list = np.argsort(gpu_memory)[::-1]
        gpus = ','.join(map(str, gpu_list))
        os.environ.setdefault('CUDA_VISIBLE_DEVICES', gpus)
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print('torch.cuda.device_count', torch.cuda.device_count())

'''
gpufree memory: [11016, 9541, 11016, 9541]
torch.cuda.device_count 4
'''

# 3.并行训练模型
# if flag == 0:
if flag == 1:
    batch_size = 16
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 数据
    inputs = torch.randn(batch_size, 3)
    inputs = inputs.to(device)

    # 定义模型
    model = FooNet(neural_num=3)
    model = nn.DataParallel(model)
    model.to(device)

    # 训练
    for epoch in range(3):
        output = model(inputs)
        print(f'model output.shape: {output.shape}')

    print(f'\nCUDA_VISIBLE_DEVICES: {os.environ["CUDA_VISIBLE_DEVICES"]}')
    print(f'device_count: {torch.cuda.device_count()}')

'''
flag == 0:
torch.cuda.device_count 2
batch size in forward: 8
batch size in forward: 8
model output.shape: torch.Size([16, 3])
batch size in forward: 8
batch size in forward: 8
model output.shape: torch.Size([16, 3])
batch size in forward: 8
batch size in forward: 8
model output.shape: torch.Size([16, 3])

CUDA_VISIBLE_DEVICES: 0,2
device_count: 2


flag == 1:
gpufree memory: [11019, 11019, 11019, 11019]
torch.cuda.device_count 4
batch size in forward: 4batch size in forward: 4

batch size in forward: 4batch size in forward: 4

model output.shape: torch.Size([16, 3])
batch size in forward: 4
batch size in forward: 4
batch size in forward: 4
batch size in forward: 4
model output.shape: torch.Size([16, 3])
batch size in forward: 4
batch size in forward: 4
batch size in forward: 4
batch size in forward: 4
model output.shape: torch.Size([16, 3])

CUDA_VISIBLE_DEVICES: 3,2,1,0
device_count: 4
'''
