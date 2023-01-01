# -*- coding: utf-8 -*-

'''
@Author   :   Corley Tang
@Contact  :   cutercorleytd@gmail.com
@Github   :   https://github.com/corleytd
@Time     :   2022-11-24 15:49
@Project  :   PyTorchBasic-momentum
'''

import numpy as np
import torch
from matplotlib import pyplot as plt
from torch import optim


def exp_w_func(beta, time_list):
    return [(1 - beta) * pow(beta, exp) for exp in time_list]


def pow_func(x):
    return 4 * torch.pow(x, 2)


beta = 0.9
num_point = 100
time_list = np.arange(num_point).tolist()

flag = 2

# 指数加权平均
if flag == 0:
    weights = exp_w_func(beta, time_list)

    plt.plot(time_list, weights, '-ro', label=f'Beta: {beta}\ny = (1-B)*B^t')
    plt.xlabel('time')
    plt.ylabel('weight')
    plt.legend()
    plt.title('Exponentially Weighted Average')
    plt.show()

    print(np.sum(weights))

# 多个权重对比
if flag == 1:
    beta_list = [0.98, 0.95, 0.9, 0.8]
    weight_list = [exp_w_func(beta, time_list) for beta in beta_list]
    for idx, weight in enumerate(weight_list):
        plt.plot(time_list, weight, label=f'Beta: {beta_list[idx]}')
        plt.xlabel('time')
        plt.ylabel('weight')

    plt.legend()
    plt.show()

# momentum
if flag == 2:
    iteration = 100
    m = 0.63  # 0 0.9 0.63
    lr_list = [0.01, 0.03]

    momentum_list = []
    losses = [[] for _ in range(len(lr_list))]
    iters = []

    for idx, lr in enumerate(lr_list):
        x = torch.tensor([2.], requires_grad=True)

        momentum = 0 if lr == 0.03 else m
        momentum_list.append(momentum)

        optimizer = optim.SGD([x], lr=lr, momentum=momentum)

        for ite in range(iteration):
            y = pow_func(x)
            y.backward()

            optimizer.step()
            optimizer.zero_grad()

            losses[idx].append(y.item())

    for i, loss in enumerate(losses):
        plt.plot(range(len(loss)), loss, label=f'LR:{lr_list[i]:.4f} M:{momentum_list[i]}')

    plt.xlabel('Iteration')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()
