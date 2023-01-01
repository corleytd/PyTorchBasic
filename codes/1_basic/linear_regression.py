# -*- coding: utf-8 -*-

'''
@Author   :   Corley Tang
@Contact  :   cutercorleytd@gmail.com
@Github   :   https://github.com/corleytd
@Time     :   2022-10-25 22:30
@Project  :   PyTorchBasic-linear_regression
'''

import matplotlib.pyplot as plt
import torch

torch.manual_seed(65537)

lr = 0.05  # 学习率

# 生成训练数据
x = torch.rand(50, 1) * 10  # 数据x
y = 1.5 * x + 5 + torch.randn(50, 1)  # 数据y，y=2x+5

# 构建线性回归参数
w = torch.randn((1,), requires_grad=True)
b = torch.zeros((1,), requires_grad=True)

for iteration in range(1000):
    # 前向传播
    wx = torch.mul(w, x)
    y_pred = torch.add(wx, b)

    # 计算MSE loss
    loss = (0.5 * (y - y_pred) ** 2).mean()

    # 反向传播
    loss.backward()

    # 更新参数
    w.data.sub_(lr * w.grad)
    b.data.sub_(lr * b.grad)

    # 清零梯度
    w.grad.zero_()
    b.grad.zero_()

    # 绘图
    if iteration % 20 == 0:
        plt.cla()
        plt.scatter(x.data.numpy(), y.data.numpy())
        plt.plot(x.data.numpy(), y_pred.data.numpy(), 'r-', lw=5)
        plt.text(2, 15, 'Loss=%.4f' % loss.item(), fontdict={'size': 20, 'color': 'red'})
        plt.xlim(1.5, 10)
        plt.ylim(7, 20)
        plt.title('Iteration: {}\nw: {:.6f} b: {:.6f}'.format(iteration, w.item(), b.item()))
        plt.pause(0.5)

        if loss.item() < 0.5:
            break

    plt.show()
