# -*- coding: utf-8 -*-

'''
@Author   :   Corley Tang
@Contact  :   cutercorleytd@gmail.com
@Github   :   https://github.com/corleytd
@Time     :   2022-12-21 11:20
@Project  :   PyTorchBasic-cuda_use
'''

import torch
from torch import nn

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

flag = 4

# 1.张量使用GPU
if flag == 0:
    x_cpu = torch.ones(3, 3)
    print(f'x_cpu: device:{x_cpu.device} is_cuda: {x_cpu.is_cuda} id: {id(x_cpu)}')

    # x_gpu = x_cpu.cuda()  # 已弃用
    x_gpu = x_cpu.to(device)
    print(f'x_gpu: device:{x_gpu.device} is_cuda: {x_gpu.is_cuda} id: {id(x_gpu)}')

# 2.模型使用GPU
if flag == 1:
    net = nn.Sequential(nn.Linear(3, 3))
    print(f'id: {id(net)} is_cuda: {next(net.parameters()).is_cuda}')

    net.to(device)
    print(f'id: {id(net)} is_cuda: {next(net.parameters()).is_cuda}')

# 3.GPU上前向传播
if flag == 2:
    data = torch.ones(3, 3)

    net = nn.Sequential(nn.Linear(3, 3))
    net.to(device)

    # output = net(data)  # RuntimeError
    data = data.to(device)
    output = net(data)

    print(f'output is_cuda: {output.is_cuda}')

# 4.指定GPU
if flag == 3:
    gpu_id = 0
    device = torch.device(f'cuda:{gpu_id}' if torch.cuda.is_available() else 'cpu')

    data = torch.ones((3, 3), device=device)

    print(f'data: device: {data.device} is_cuda: {data.is_cuda} id: {id(data)}')

# 5.获取GPU数量、名称
if flag == 4:
    device_count = torch.cuda.device_count()
    print(f'device_count: {device_count}')

    device_name = torch.cuda.get_device_name(0)
    print(f'device_name: {device_name}')
