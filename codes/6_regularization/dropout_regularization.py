# -*- coding: utf-8 -*-

'''
@Author   :   Corley Tang
@Contact  :   cutercorleytd@gmail.com
@Github   :   https://github.com/corleytd
@Time     :   2022-12-10 18:46
@Project  :   PyTorchBasic-dropout_regularization
'''

import torch
from matplotlib import pyplot as plt
from torch import nn
from torch import optim
from torch.utils.tensorboard import SummaryWriter

from tools.common_tools import set_seed

set_seed()

# 超参数
n_hidden = 200
max_epoch = 2000
visual_interval = 400
lr = 0.01


# 1.生成数据
def gen_data(count=10, x_range=(-1, 1)):
    w = 1.5
    train_x = torch.linspace(*x_range, count).unsqueeze_(1)
    train_y = w * train_x + torch.normal(0, 0.5, train_x.size())
    test_x = torch.linspace(*x_range, count).unsqueeze_(1)
    test_y = w * train_x + torch.normal(0, 0.3, test_x.size())

    return train_x, train_y, test_x, test_y


train_x, train_y, test_x, test_y = gen_data()


# 2.定义模型
class MLP(nn.Module):
    def __init__(self, neural_num, dropout=0.5):
        super().__init__()
        self.linears = nn.Sequential(
            nn.Linear(1, neural_num),
            nn.ReLU(inplace=True),

            nn.Dropout(dropout),
            nn.Linear(neural_num, neural_num),
            nn.ReLU(inplace=True),

            nn.Dropout(dropout),
            nn.Linear(neural_num, neural_num),
            nn.ReLU(inplace=True),

            nn.Dropout(dropout),
            nn.Linear(neural_num, 1)
        )

    def forward(self, x):
        return self.linears(x)


net_normal = MLP(neural_num=n_hidden, dropout=0)
net_dr = MLP(neural_num=n_hidden, dropout=0.5)

# 3.定义优化器
optimizer_normal = optim.SGD(net_normal.parameters(), lr=lr, momentum=0.9)
optimizer_dr = optim.SGD(net_dr.parameters(), lr=lr, momentum=0.9, weight_decay=1e-2)

# 4.定义损失函数
loss_func = nn.MSELoss()

# 5.迭代训练
writer = SummaryWriter(comment='_dropout')

for epoch in range(1, max_epoch + 1):
    # 前向传播
    pred_normal, pred_dr = net_normal(train_x), net_dr(train_x)

    # 计算损失
    loss_normal, loss_dr = loss_func(pred_normal, train_y), loss_func(pred_dr, train_y)

    # 反向传播
    loss_normal.backward()
    loss_dr.backward()

    # 更新参数
    optimizer_normal.step()
    optimizer_dr.step()

    # 可视化
    if epoch % visual_interval == 0:
        net_normal.eval()
        net_dr.eval()

        for name, layer in net_normal.named_parameters():
            writer.add_histogram(name + '_grad_normal', layer.grad, epoch)
            writer.add_histogram(name + '_data_normal', layer, epoch)

        for name, layer in net_dr.named_parameters():
            writer.add_histogram(name + '_grad_dr', layer.grad, epoch)
            writer.add_histogram(name + '_data_dr', layer, epoch)

        # 测试
        pred_normal_test, pred_dr_test = net_normal(test_x), net_dr(test_x)

        # 绘图
        plt.clf()
        plt.scatter(train_x.data.numpy(), train_y.data.numpy(), c='blue', s=50, alpha=0.3, label='train')
        plt.scatter(test_x.data.numpy(), test_y.data.numpy(), c='red', s=50, alpha=0.3, label='test')
        plt.plot(test_x.data.numpy(), pred_normal_test.data.numpy(), 'r-', lw=3, label='no dropout')
        plt.plot(test_x.data.numpy(), pred_dr_test.data.numpy(), 'b--', lw=3, label='dropout')
        plt.text(-0.25, -1.5, f'no dropout loss={loss_normal.item():.8f}', fontdict={'size': 15, 'color': 'red'})
        plt.text(-0.25, -2, f'dropout loss={loss_dr.item():.8f}', fontdict={'size': 15, 'color': 'red'})

        plt.title(f'Epoch {epoch}')
        plt.ylim((-2.5, 2.5))
        plt.legend(loc='upper left')
        plt.show()
        plt.close()

        net_normal.train()
        net_dr.train()

    # 梯度清零
    optimizer_normal.zero_grad()
    optimizer_dr.zero_grad()
