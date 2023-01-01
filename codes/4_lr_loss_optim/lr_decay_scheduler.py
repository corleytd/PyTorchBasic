# -*- coding: utf-8 -*-

'''
@Author   :   Corley Tang
@Contact  :   cutercorleytd@gmail.com
@Github   :   https://github.com/corleytd
@Time     :   2022-12-05 18:11
@Project  :   PyTorchBasic-lr_decay_scheduler
'''

import torch
from matplotlib import pyplot as plt
from torch import optim

from tools.common_tools import set_seed

set_seed()

LR = 0.5
iteration = 10
max_epoch = 200

# 生成数据
weights = torch.randn(1, requires_grad=True)
target = torch.zeros(1)

optimizer = optim.SGD([weights], lr=LR, momentum=0.9)

flag = 5

# 1.StepLR
if flag == 0:
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.1)  # 设置学习率下降策略
    lrs, epochs = [], []

    for epoch in range(max_epoch):
        # 获取当前学习率
        lrs.append(scheduler.get_last_lr())
        epochs.append(epoch)

        for i in range(iteration):
            loss = torch.pow(weights - target, 2)
            loss.backward()

            optimizer.step()
            optimizer.zero_grad()

        scheduler.step()

    plt.plot(epochs, lrs, label='Step LR Scheduler')
    plt.xlabel('Epoch')
    plt.ylabel('Learning Rate')

    plt.legend()
    plt.show()

# 2.MultiStepLR
if flag == 1:
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[30, 100, 150, 180], gamma=0.3)
    lrs, epochs = [], []

    for epoch in range(max_epoch):
        lrs.append(scheduler.get_last_lr())
        epochs.append(epoch)

        for i in range(iteration):
            loss = torch.pow(weights - target, 2)
            loss.backward()

            optimizer.step()
            optimizer.zero_grad()

        scheduler.step()

    plt.plot(epochs, lrs, label='Multi Step LR Scheduler')
    plt.xlabel('Epoch')
    plt.ylabel('Learning Rate')

    plt.legend()
    plt.show()

# 3.ExponentialLR
if flag == 2:
    gamma = 0.95
    scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=gamma)
    lrs, epochs = [], []

    for epoch in range(max_epoch):
        lrs.append(scheduler.get_last_lr())
        epochs.append(epoch)

        for i in range(iteration):
            loss = torch.pow(weights - target, 2)
            loss.backward()

            optimizer.step()
            optimizer.zero_grad()

        scheduler.step()

    plt.plot(epochs, lrs, label=f'Exponential LR Scheduler\ngamma:{gamma}')
    plt.xlabel('Epoch')
    plt.ylabel('Learning Rate')

    plt.legend()
    plt.show()

# 4.CosineAnnealingLR
if flag == 3:
    t_max = 40
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=t_max, eta_min=0)
    lrs, epochs = [], []

    for epoch in range(max_epoch):
        lrs.append(scheduler.get_last_lr())
        epochs.append(epoch)

        for i in range(iteration):
            loss = torch.pow(weights - target, 2)
            loss.backward()

            optimizer.step()
            optimizer.zero_grad()

        scheduler.step()

    plt.plot(epochs, lrs, label=f'Cosine Annealing LR Scheduler\nT_max:{t_max}')
    plt.xlabel('Epoch')
    plt.ylabel('Learning Rate')

    plt.legend()
    plt.show()

# 5.ReduceLROnPlateau
if flag == 4:
    loss = 0.5

    factor = 0.1
    mode = 'min'
    patience = 10
    cooldown = 3
    min_lr = 1e-5
    verbose = True

    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=factor, mode=mode, patience=patience,
                                                     cooldown=cooldown, min_lr=min_lr, verbose=verbose)

    for epoch in range(max_epoch):
        for i in range(iteration):
            optimizer.step()
            optimizer.zero_grad()

        if epoch == 10:
            loss = 0.1

        scheduler.step(loss)

# 6.LambdaLR
if flag == 5:
    lr_init = 0.1

    weights1 = torch.randn((16, 3, 128, 128))
    weights2 = torch.ones((5, 5))

    optimizer = optim.SGD([
        {'params': [weights1]},
        {'params': [weights2]}
    ], lr=lr_init)

    lambda1 = lambda epoch: 0.1 ** (epoch // 10)
    lambda2 = lambda epoch: 0.95 ** epoch

    scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=[lambda1, lambda2])

    lrs, epochs = [], []

    for epoch in range(max_epoch):
        for i in range(iteration):
            optimizer.step()
            optimizer.zero_grad()

        scheduler.step()

        lrs.append(scheduler.get_last_lr())
        epochs.append(epoch)
        print(f'epoch:{epoch:5d}, lr:{scheduler.get_last_lr()}')

    plt.plot(epochs, [lr[0] for lr in lrs], label='lambda 1')
    plt.plot(epochs, [lr[1] for lr in lrs], label='lambda 2')
    plt.xlabel('Epoch')
    plt.ylabel('Learning Rate')
    plt.title('Lambda LR')

    plt.legend()
    plt.show()
