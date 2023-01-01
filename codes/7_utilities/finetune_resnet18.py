# -*- coding: utf-8 -*-

'''
@Author   :   Corley Tang
@Contact  :   cutercorleytd@gmail.com
@Github   :   https://github.com/corleytd
@Time     :   2022-12-20 18:04
@Project  :   PyTorchBasic-finetune_resnet18
'''

import os

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn
from torch import optim
from torch.utils.data import DataLoader
from torchvision import transforms, models

from tools.common_tools import set_seed
from tools.datasets import AntBeeDataset

set_seed()  # 设置随机种子

# 设置超参数
MAX_EPOCH = 25
BATCH_SIZE = 32
lr = 1e-3
log_interval = 5
val_interval = 2
classes = 2
start_epoch = -1
lr_decay_step = 7
use_fine_tune = True
is_ban_conv = False
use_multi_lr = False
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'use device: {device}')

# 1.数据处理
split_path = '../../data/Ant_Bee_split'
train_path = os.path.join(split_path, 'train')
valid_path = os.path.join(split_path, 'val')

norm_mean = [0.485, 0.456, 0.406]
norm_std = [0.229, 0.224, 0.225]

train_transform = transforms.Compose([
    transforms.RandomResizedCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(norm_mean, norm_std)
])
valid_transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(norm_mean, norm_std)
])

# 创建Dataset
train_data = AntBeeDataset(train_path, train_transform)
valid_data = AntBeeDataset(valid_path, valid_transform)
# 构建DataLoader
train_loader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True)
valid_loader = DataLoader(valid_data, batch_size=BATCH_SIZE * 2)

# 2.搭建模型
# 2.1 定义模型
resnet = models.resnet18()

# 2.2 加载预训练参数
if use_fine_tune:
    pretrained_resnet_path = '../../data/pretrained_models/resnet18.pth'
    state_dict = torch.load(pretrained_resnet_path)
    resnet.load_state_dict(state_dict)

# M1.1: 冻结卷积层
if is_ban_conv:
    for param in resnet.conv1.parameters():
        param.requires_grad = False
    print(f'resnet conv layers banned, conv1.weight[0, 0, ...]: {resnet.conv1.weight[0, 0, 0]}')

# 2.3 替换全连接层
num_features = resnet.fc.in_features
resnet.fc = nn.Linear(num_features, classes)

resnet.to(device)

# 3.定义损失函数
criterion = nn.CrossEntropyLoss()

# 4.定义优化器
# M2: 卷积层设置较小的学习率
if use_multi_lr:
    fc_param_ids = list(map(id, resnet.fc.parameters()))  # 全连接层参数的内存地址
    non_fc_params = filter(lambda p: id(p) not in fc_param_ids, resnet.parameters())  # 非全连接层参数，即卷积层参数
    optimizer = optim.SGD([
        # {'params': non_fc_params, 'lr': lr * 0},  # M1.2: 冻结卷积层
        {'params': non_fc_params, 'lr': lr * 0.1},
        {'params': resnet.fc.parameters()}
    ], lr=lr, momentum=0.9)
else:
    optimizer = optim.SGD(resnet.parameters(), lr=lr, momentum=0.9)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)  # 设置学习率下降策略

# 5.训练
train_curve = []
valid_curve = []

for epoch in range(1, MAX_EPOCH + 1):
    avg_loss, correct, total = 0, 0, 0

    resnet.train()

    for iteration, data in enumerate(train_loader, 1):
        # 前向传播
        inputs, labels = data
        inputs, labels = inputs.to(device), labels.to(device)
        outputs = resnet(inputs)

        # 计算损失
        loss = criterion(outputs, labels)

        # 反向传播
        loss.backward()

        # 更新参数
        optimizer.step()

        # 梯度清零
        optimizer.zero_grad()

        # 统计结果
        _, preds = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (preds == labels).squeeze().sum().item()

        # 输出训练信息
        avg_loss += loss.item()
        train_curve.append(loss.item())
        if iteration % log_interval == 0:
            avg_loss /= log_interval
            print(
                f'Train: Epoch: {epoch:0>3}/{MAX_EPOCH:0>3} Iteration: {iteration:0>3}/{len(train_loader):0>3} Loss: {avg_loss:.4f} Acc: {correct / total:.2%}')
            avg_loss = 0

            if is_ban_conv:
                print(f'resnet conv layers banned, conv1.weight[0, 0, ...]: {resnet.conv1.weight[0, 0, 0]}')

    # 更新学习率
    scheduler.step()

    # 验证模型
    if epoch % val_interval == 0:
        val_correct, val_total, val_loss = 0, 0, 0
        resnet.eval()

        with torch.no_grad():
            for iteration, data in enumerate(valid_loader, 1):
                inputs, labels = data
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = resnet(inputs)
                loss = criterion(outputs, labels)

                _, preds = torch.max(outputs, 1)
                val_total += labels.size(0)
                val_correct += (preds == labels).squeeze().sum().item()

                val_loss += loss.item()

            avg_val_loss = val_loss / len(valid_loader)
            valid_curve.append(avg_val_loss)
            print(f'Valid: Epoch: {epoch:0>3}/{MAX_EPOCH:0>3} Loss: {avg_loss:.4f} Acc: {val_correct / val_total:.2%}')

# 画图
train_res_x = range(len(train_curve))
train_res_y = train_curve
train_iters = len(train_loader)

valid_res_x = np.arange(1,
                        len(valid_curve) + 1) * train_iters * val_interval - 1  # 由于valid中记录的是epoch loss，需要对记录点进行转换到iterations
valid_res_y = valid_curve

plt.plot(train_res_x, train_res_y, label='Train')
plt.plot(valid_res_x, valid_res_y, label='Valid')

plt.legend(loc='upper right')
plt.xlabel('Iteration')
plt.ylabel('Loss')
plt.show()

# 保存模型
torch.save(resnet.state_dict(), 'output/resnet18.ckpt')
