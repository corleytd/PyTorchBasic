# -*- coding: utf-8 -*-

'''

@Time     :   2022-12-08 18:08
@Project  :   PyTorchBasic-l2_regularization
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
visual_interval = 200
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
    def __init__(self, neural_num):
        super().__init__()
        self.linears = nn.Sequential(
            nn.Linear(1, neural_num),
            nn.ReLU(inplace=True),
            nn.Linear(neural_num, neural_num),
            nn.ReLU(inplace=True),
            nn.Linear(neural_num, neural_num),
            nn.ReLU(inplace=True),
            nn.Linear(neural_num, 1)
        )

    def forward(self, x):
        return self.linears(x)


net_normal = MLP(neural_num=n_hidden)
net_decay = MLP(neural_num=n_hidden)

# 3.定义优化器
optimizer_normal = optim.SGD(net_normal.parameters(), lr=lr, momentum=0.9)
optimizer_decay = optim.SGD(net_decay.parameters(), lr=lr, momentum=0.9, weight_decay=1e-2)

# 4.定义损失函数
loss_func = nn.MSELoss()

# 5.迭代训练
writer = SummaryWriter(comment='_weight_decay')

for epoch in range(1, max_epoch + 1):
    # 前向传播
    pred_normal, pred_decay = net_normal(train_x), net_decay(train_x)

    # 计算损失
    loss_normal, loss_decay = loss_func(pred_normal, train_y), loss_func(pred_decay, train_y)

    # 反向传播
    loss_normal.backward()
    loss_decay.backward()

    # 更新参数
    optimizer_normal.step()
    optimizer_decay.step()

    # 可视化
    if epoch % visual_interval == 0:
        for name, layer in net_normal.named_parameters():
            writer.add_histogram(name + '_grad_normal', layer.grad, epoch)
            writer.add_histogram(name + '_data_normal', layer, epoch)

        for name, layer in net_decay.named_parameters():
            writer.add_histogram(name + '_grad_decay', layer.grad, epoch)
            writer.add_histogram(name + '_data_decay', layer, epoch)

        # 测试
        pred_normal_test, pred_decay_test = net_normal(test_x), net_decay(test_x)

        # 绘图
        plt.scatter(train_x.data.numpy(), train_y.data.numpy(), c='blue', s=50, alpha=0.3, label='train')
        plt.scatter(test_x.data.numpy(), test_y.data.numpy(), c='red', s=50, alpha=0.3, label='test')
        plt.plot(test_x.data.numpy(), pred_normal_test.data.numpy(), 'r-', lw=3, label='no weight decay')
        plt.plot(test_x.data.numpy(), pred_decay_test.data.numpy(), 'b--', lw=3, label='weight decay')
        plt.text(-0.25, -1.5, f'no weight decay loss={loss_normal.item():.6f}', fontdict={'size': 15, 'color': 'red'})
        plt.text(-0.25, -2, f'weight decay loss={loss_decay.item():.6f}', fontdict={'size': 15, 'color': 'red'})

        plt.title(f'Epoch {epoch}')
        plt.ylim((-2.5, 2.5))
        plt.legend(loc='upper left')
        plt.show()

    # 梯度清零
    optimizer_normal.zero_grad()
    optimizer_decay.zero_grad()
