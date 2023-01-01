# -*- coding: utf-8 -*-

'''
@Author   :   Corley Tang
@Contact  :   cutercorleytd@gmail.com
@Github   :   https://github.com/corleytd
@Time     :   2022-11-18 22:19
@Project  :   PyTorchBasic-optimizer_methods
'''

import torch
from torch import optim

from tools.common_tools import set_seed

set_seed()

# 构造权重数据
weight = torch.randn((3, 4), requires_grad=True)
weight.grad = torch.ones_like(weight)

# 构造优化器
optimizer = optim.SGD([weight], lr=0.1, momentum=0.9)

flag = 2

# 1.step方法
if flag == 0:
    print('weight before step', weight)
    optimizer.step()
    print('weight after step', weight)

# 2.zero_grad方法
if flag == 1:
    print('weight before step', weight)
    optimizer.step()
    print('weight after step', weight)

    print(f'weight in optimizer: {id(optimizer.param_groups[0]["params"][0])}\nweight in weight: {id(weight)}')

    print(f'weight.grad is {weight.grad}\n')
    optimizer.zero_grad()
    print(f'after optimzer.zero_grad(), weight.grad is {weight.grad}\n')

# 3.add_param_group方法
if flag == 2:
    print(f'optimizer.param_groups is\n{optimizer.param_groups}')

    w2 = torch.rand((3, 3), requires_grad=True)
    w3 = torch.rand((4, 4), requires_grad=True)
    optimizer.add_param_group({'params': w2, 'lr': 1e-3, 'momentum': 0.8})
    optimizer.add_param_group({'params': w3, 'lr': 3e-2, 'momentum': 0.7})
    print(f'optimizer.param_groups is\n{optimizer.param_groups}')

# 4.state_dict方法
if flag == 3:
    opt_state_dict = optimizer.state_dict()
    print(f'optimizer.state_dict before step:\n{opt_state_dict}')

    for _ in range(10):
        optimizer.step()

    print(f'optimizer.state_dict after step:\n{optimizer.state_dict()}')
    torch.save(optimizer.state_dict(), 'optimizer_state_dict.pkl')

# 5.load_state_dict方法
if flag == 4:
    state_dict = torch.load('optimizer_state_dict.pkl')

    print(f'optimizer.state_dict before step:\n{optimizer.state_dict()}')
    optimizer.load_state_dict(state_dict)
    print(f'optimizer.state_dict after step:\n{optimizer.state_dict()}')
