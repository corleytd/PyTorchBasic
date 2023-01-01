# -*- coding: utf-8 -*-

'''
@Author   :   Corley Tang
@Contact  :   cutercorleytd@gmail.com
@Github   :   https://github.com/corleytd
@Time     :   2022-11-02 12:16
@Project  :   PyTorchBasic-train_dog_cat
'''

import os

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from torchvision import transforms

from models.lenet import CatDogLeNet
from tools.common_tools import set_seed
from tools.datasets import CatDogDataset

set_seed()  # 设置随机种子
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 设置超参数
MAX_EPOCH = 20
BATCH_SIZE = 512
lr = 1e-3
log_interval = 10
val_interval = 1

# 1.数据处理
split_path = '../../data/Cat_Dog_split'
train_path = os.path.join(split_path, 'train')
valid_path = os.path.join(split_path, 'valid')

norm_mean = [0.485, 0.456, 0.406]
norm_std = [0.229, 0.224, 0.225]

train_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomCrop(224, padding=4),
    transforms.ToTensor(),
    transforms.Normalize(norm_mean, norm_std)
])
valid_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(norm_mean, norm_std)
])

# 创建dataset
train_data = CatDogDataset(train_path, train_transform)
valid_data = CatDogDataset(valid_path, valid_transform)
# 构建DataLoader
train_loader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True)
valid_loader = DataLoader(valid_data, batch_size=BATCH_SIZE * 2)

# 2.搭建模型
lenet = CatDogLeNet(classes=2)
lenet.initialize_weights()
lenet.to(device)

# 3.定义损失函数
criterion = nn.CrossEntropyLoss()

# 4.定义优化器
optimizer = optim.Adam(lenet.parameters(), lr=lr, betas=(0.9, 0.999), eps=1e-8, weight_decay=1e-1)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)

# 5.训练
train_curve = []
valid_curve = []

for epoch in range(1, MAX_EPOCH + 1):
    correct, total, avg_loss = 0, 0, 0

    lenet.train()
    for iteration, data in enumerate(train_loader, 1):
        # 前向传播
        inputs, labels = data
        inputs, labels = inputs.to(device), labels.to(device)
        outputs = lenet(inputs)

        # 计算损失
        loss = criterion(outputs, labels)

        # 反向传播
        loss.backward()

        # 更新参数
        optimizer.step()

        # 梯度清零
        optimizer.zero_grad()

        # 统计结果
        _, preds = torch.max(outputs, 1)
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

    # 更新学习率
    scheduler.step()

    # 验证模型
    if epoch % val_interval == 0:
        val_correct, val_total, val_loss = 0, 0, 0

        lenet.eval()
        with torch.no_grad():
            for iteration, data in enumerate(valid_loader, 1):
                inputs, labels = data
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = lenet(inputs)
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

# 由于valid中记录的是epoch loss，需要将记录点转换到iterations
valid_res_x = np.arange(1, len(valid_curve) + 1) * train_iters * val_interval - 1
valid_res_y = valid_curve

plt.plot(train_res_x, train_res_y, label='Train')
plt.plot(valid_res_x, valid_res_y, label='Valid')

plt.legend(loc='upper right')
plt.xlabel('Iteration')
plt.ylabel('Loss')
plt.show()

# 推理
test_path = os.path.join(split_path, 'test')
test_data = CatDogDataset(test_path, valid_transform)
test_loader = DataLoader(test_data, batch_size=BATCH_SIZE * 2)

test_correct, test_total, test_loss = 0, 0, 0

for iteration, data in enumerate(test_loader):
    # 前向
    inputs, labels = data
    inputs, labels = inputs.to(device), labels.to(device)
    outputs = lenet(inputs)
    loss = criterion(outputs, labels)

    _, preds = torch.max(outputs, 1)
    test_total += labels.size(0)
    test_correct += (preds == labels).squeeze().sum().item()

    test_loss += loss.item()

avg_test_loss = test_loss / len(test_loader)
print(f'Test: Loss: {avg_test_loss:.4f} Acc: {test_correct / test_total:.2%}')
