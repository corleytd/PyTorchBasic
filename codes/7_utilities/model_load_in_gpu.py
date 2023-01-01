# -*- coding: utf-8 -*-

'''
@Author   :   Corley Tang
@Contact  :   cutercorleytd@gmail.com
@Github   :   https://github.com/corleytd
@Time     :   2022-12-21 18:31
@Project  :   PyTorchBasic-model_load_in_gpu
本实验在RTX 2080 Ti*4上实现
'''

import os
import sys
from collections import OrderedDict

import torch
from torch import nn

from tools.common_tools import set_seed

flag = 1
set_seed()


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


# 1.加载GPU模型到CPU
if flag == 0:
    gpus = [0, 2]
    gpus = ','.join(map(str, gpus))
    os.environ.setdefault('CUDA_VISIBLE_DEVICES', gpus)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = FooNet(neural_num=3, layers=3)
    model.to(device)

    # 保存
    state_dict = model.state_dict()
    state_dict_path = 'output/model_in_gpu_0_2.pkl'
    torch.save(state_dict, state_dict_path)

    # 加载到GPU
    state_dict_in_gpu = torch.load(state_dict_path)
    print(state_dict_in_gpu)
    state_dict_in_cpu = torch.load(state_dict_path, map_location='cpu')
    print(state_dict_in_cpu)

'''
OrderedDict([('linears.0.weight', tensor([[ 0.4414,  0.4792, -0.1353],
        [ 0.5304, -0.1265,  0.1165],
        [-0.2811,  0.3391,  0.5090]], device='cuda:0')), ('linears.1.weight', tensor([[-0.4236,  0.5018,  0.1081],
        [ 0.4266,  0.0782,  0.2784],
        [-0.0815,  0.4451,  0.0853]], device='cuda:0')), ('linears.2.weight', tensor([[-0.2695,  0.1472, -0.2660],
        [-0.0677, -0.2345,  0.3830],
        [-0.4557, -0.2662, -0.1630]], device='cuda:0'))])
OrderedDict([('linears.0.weight', tensor([[ 0.4414,  0.4792, -0.1353],
        [ 0.5304, -0.1265,  0.1165],
        [-0.2811,  0.3391,  0.5090]])), ('linears.1.weight', tensor([[-0.4236,  0.5018,  0.1081],
        [ 0.4266,  0.0782,  0.2784],
        [-0.0815,  0.4451,  0.0853]])), ('linears.2.weight', tensor([[-0.2695,  0.1472, -0.2660],
        [-0.0677, -0.2345,  0.3830],
        [-0.4557, -0.2662, -0.1630]]))])
'''

# 2.多GPU场景保存模型
if flag == 1:
    if torch.cuda.device_count() < 2:
        print('GPU数量不足，青岛多GPU环境下运行')
        sys.exit(0)

    gpus = [0, 1, 2, 3]
    gpus = ','.join(map(str, gpus))
    os.environ.setdefault('CUDA_VISIBLE_DEVICES', gpus)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = FooNet(neural_num=3, layers=3)
    model = nn.DataParallel(model)
    model.to(device)

    # 保存
    state_dict = model.state_dict()
    state_dict_path = 'output/model_in_multi_gpu.pkl'
    torch.save(state_dict, state_dict_path)

# 3.多GPU场景加载模型
if flag == 1:
    model = FooNet(neural_num=3, layers=3)

    state_dict_path = 'output/model_in_multi_gpu.pkl'
    state_dict = torch.load(state_dict_path, map_location='cpu')
    print(f'state dict loaded:\n{state_dict}')

    # model.load_state_dict(state_dict)  # RuntimeError

    # 修正state_dict的key，去掉module.
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        new_state_dict[k.replace('module.', '')] = v
    print(f'new state dict:\n{new_state_dict}')

    model.load_state_dict(new_state_dict)

'''
state dict loaded:
OrderedDict([('module.linears.0.weight', tensor([[ 0.4414,  0.4792, -0.1353],
        [ 0.5304, -0.1265,  0.1165],
        [-0.2811,  0.3391,  0.5090]])), ('module.linears.1.weight', tensor([[-0.4236,  0.5018,  0.1081],
        [ 0.4266,  0.0782,  0.2784],
        [-0.0815,  0.4451,  0.0853]])), ('module.linears.2.weight', tensor([[-0.2695,  0.1472, -0.2660],
        [-0.0677, -0.2345,  0.3830],
        [-0.4557, -0.2662, -0.1630]]))])
new state dict:
OrderedDict([('linears.0.weight', tensor([[ 0.4414,  0.4792, -0.1353],
        [ 0.5304, -0.1265,  0.1165],
        [-0.2811,  0.3391,  0.5090]])), ('linears.1.weight', tensor([[-0.4236,  0.5018,  0.1081],
        [ 0.4266,  0.0782,  0.2784],
        [-0.0815,  0.4451,  0.0853]])), ('linears.2.weight', tensor([[-0.2695,  0.1472, -0.2660],
        [-0.0677, -0.2345,  0.3830],
        [-0.4557, -0.2662, -0.1630]]))])
'''
