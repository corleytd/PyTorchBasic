# -*- coding: utf-8 -*-

'''
@Author   :   Corley Tang
@Contact  :   cutercorleytd@gmail.com
@Github   :   https://github.com/corleytd
@Time     :   2022-10-26 21:54
@Project  :   PyTorchBasic-logistic_regression
'''

import numpy as np
import torch
from matplotlib import pyplot as plt
from torch import nn

torch.manual_seed(65537)

# 1.生成数据
sample_num = 100
mean_value = 1.7
bias = -1
n_data = torch.ones(sample_num, 2)
x0 = torch.normal(mean_value * n_data, 1) + bias
y0 = torch.zeros(sample_num)
x1 = torch.normal(-mean_value * n_data, 1) + bias
y1 = torch.ones(sample_num)
x = torch.cat([x0, x1], 0)
y = torch.cat([y0, y1], 0)


# 2.选择模型
class LR(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Linear(2, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.features(x)
        x = self.sigmoid(x)
        return x


lr_net = LR()  # 实例化逻辑回归模型

# 3.选择损失函数
loss_fn = nn.BCELoss()

# 4.选择优化器
lr = 0.01
optimizer = torch.optim.SGD(lr_net.parameters(), lr=lr, momentum=0.9)

# 5.模型训练
for iteration in range(1000):
    # 前向传播
    y_pred = lr_net(x)

    # 计算损失
    loss = loss_fn(y_pred.squeeze(), y)

    # 反向传播
    loss.backward()

    # 更新参数
    optimizer.step()

    # 清空梯度
    optimizer.zero_grad()

    # 绘图
    if iteration % 20 == 0:
        mask = y_pred.ge(0.5).float().squeeze()  # 以0.5为阈值进行分类
        correct = (mask == y).sum()  # 统计预测正确的样本个数
        acc = correct.item() / y.size(0)  # 计算分类正确率

        plt.scatter(x0.data.numpy()[:, 0], x0.data.numpy()[:, 1], c='r', label='class 0')
        plt.scatter(x1.data.numpy()[:, 0], x1.data.numpy()[:, 1], c='b', label='class 1')

        w0, w1 = lr_net.features.weight[0]
        w0, w1 = w0.item(), w1.item()
        plot_b = lr_net.features.bias[0].item()
        plot_x = np.arange(-6, 6, 0.1)
        plot_y = (-w0 * plot_x - plot_b) / w1

        plt.xlim(-5, 7)
        plt.ylim(-7, 7)
        plt.plot(plot_x, plot_y)

        plt.text(-4.5, 5, 'Loss=%.4f' % loss.item(), fontdict={'size': 20, 'color': 'red'})
        plt.title(
            'Iteration: {}\nw0: {:.2f} w1: {:.2f} b: {:.2f} accuracy: {:.2f}'.format(iteration, w0, w1, plot_b, acc))
        plt.legend()

        plt.show()
        plt.pause(0.5)

        if acc > 0.99:
            break
