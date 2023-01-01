# -*- coding: utf-8 -*-

'''
@Author   :   Corley Tang
@Contact  :   cutercorleytd@gmail.com
@Github   :   https://github.com/corleytd
@Time     :   2022-11-24 14:38
@Project  :   PyTorchBasic-learning_rate
'''

import matplotlib.pyplot as plt
import numpy as np
import torch


def func(x_t):
    '''
    y=2*x^2 dy/dx=8x
    '''
    return 4 * torch.pow(x_t, 2)


# 初始化自变量x
x = torch.tensor([2.], requires_grad=True)

flag = 2

# 绘制函数图像
if flag == 0:
    x_t = torch.linspace(-3, 3, 100)
    y = func(x_t)

    plt.plot(x_t.numpy(), y.numpy(), label='y = 4*x^2')
    plt.grid()
    plt.xlabel('x')
    plt.ylabel('y')
    plt.legend()
    plt.show()

# 梯度下降
if flag == 1:
    iters, losses, xes = [], [], []

    lr = 0.125  # 1 0.5 0.2 0.1 0.125
    max_iteration = 20  # 4 20

    for i in range(max_iteration):
        y = func(x)
        y.backward()

        print(f'Iter:{i}, x:{x.item():8}, x.grad:{x.grad.item():8}, loss:{y.item():10}')

        iters.append(i)
        xes.append(x.item())
        losses.append(y.item())

        x.data.sub_(lr * x.grad)
        x.grad.zero_()

    plt.subplot(121).plot(iters, losses, '-ro')
    plt.xlabel('Iteration')
    plt.ylabel('Loss')

    x_t = torch.linspace(-3, 3, 100)
    y = func(x_t)
    plt.subplot(122).plot(x_t.numpy(), y.numpy(), label='y = 4*x^2')
    plt.grid()
    ys = [func(torch.tensor(i)).item() for i in xes]
    plt.subplot(122).plot(xes, ys, '-ro')
    plt.legend()
    plt.show()

# 多个学习率的对比
if flag == 2:
    iteration = 100
    num_lr = 10
    lr_min, lr_max = 0.01, 0.2  # 0.5 0.3 0.2
    lr_list = np.linspace(lr_min, lr_max, num=num_lr)

    losses = [[] for _ in range(num_lr)]

    for idx, lr in enumerate(lr_list):
        x = torch.tensor([2.], requires_grad=True)

        for i in range(iteration):
            y = func(x)
            y.backward()

            x.data.sub_(lr * x.grad)
            x.grad.zero_()

            losses[idx].append(y.item())

    for idx, loss in enumerate(losses):
        plt.plot(range(len(loss)), loss, label=f'LR: {lr_list[idx]:.4f}')

    plt.legend()
    plt.xlabel('Iteration')
    plt.ylabel('Loss')
    plt.show()
