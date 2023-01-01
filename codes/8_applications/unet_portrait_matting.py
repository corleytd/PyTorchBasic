# -*- coding: utf-8 -*-

'''
@Author   :   Corley Tang
@Contact  :   cutercorleytd@gmail.com
@Github   :   https://github.com/corleytd
@Time     :   2022-12-24 15:45
@Project  :   PyTorchBasic-unet_portrait_matting
'''

import os

import numpy as np
import torch
from matplotlib import pyplot as plt
from torch import nn, optim
from torch.utils.data import DataLoader

from models.unet import UNet
from tools.common_tools import set_seed
from tools.datasets import PortraitDataset

set_seed()


# 计算dice系数
def compute_dice(y_pred, y_true):
    return torch.sum(y_pred[y_true == 1]) * 2 / (torch.sum(y_pred) + torch.sum(y_true))


# 超参数
LR = 0.01
BATCH_SIZE = 4
start_epoch = 1
max_epoch = 200
lr_step = 40
log_interval = 10
val_interval = 5
ckpt_interval = 40
vis_interval = 10
vis_num = 5
mask_threshold = 0.5
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 1.准备数据
data_dir = '../../data/Portrait_split'
train_path = os.path.join(data_dir, 'train')
valid_path = os.path.join(data_dir, 'valid')

# 数据预处理
train_set = PortraitDataset(train_path)
valid_set = PortraitDataset(valid_path)

train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True)
valid_loader = DataLoader(valid_set, batch_size=BATCH_SIZE * 2, shuffle=True)
train_batch_count = len(train_loader)

# 2.定义模型
model = UNet(in_channels=3, out_channels=1, num_features=32)
model.to(device)

# 3.定义损失函数
criterion = nn.MSELoss()

# 4.定义优化器
optimizer = optim.SGD(model.parameters(), lr=LR, momentum=0.9)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=lr_step, gamma=0.1)

# 5.迭代训练
train_curve, valid_curve, train_dice_curve, valid_dice_curve = [], [], [], []

for epoch in range(start_epoch, max_epoch + 1):
    train_loss_total, train_dice_total = 0, 0

    model.train()
    for iteration, (inputs, labels) in enumerate(train_loader, 1):
        inputs, labels = inputs.to(device), labels.to(device)

        # 前向传播
        outputs = model(inputs)

        # 计算损失
        loss = criterion(outputs, labels)

        # 反向传播
        loss.backward()

        # 更新参数
        optimizer.step()

        # 梯度清零
        optimizer.zero_grad()

        # 统计结果
        train_curve.append(loss.item())
        train_dice = compute_dice(outputs.ge(mask_threshold).cpu(), labels.cpu())
        train_dice_curve.append(train_dice)
        train_loss_total += loss.item()

        if iteration % log_interval == 0:
            print(f'Train: Epoch: [{epoch:0>3}/{max_epoch:0>3}] Iteration: [{iteration:0>3}/{train_batch_count:0>3}] '
                  f'running_loss: {loss.item():.4f} mean_loss: {train_loss_total / iteration:.4f} '
                  f'running_dice: {train_dice:.4f} lr: {scheduler.get_last_lr()[0]:.6f}')

    scheduler.step()

    if epoch % ckpt_interval == 0:
        checkpoint = {
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'epoch': epoch
        }
        ckpt_path = f'output/unet_epoch_{epoch}.ckpt'
        torch.save(checkpoint, ckpt_path)

    # 验证模型
    if epoch % val_interval == 0:
        model.eval()

        valid_loss_total, valid_dice_total = 0, 0
        with torch.no_grad():
            for idx, (inputs, labels) in enumerate(valid_loader):
                inputs, labels = inputs.to(device), labels.to(device)

                outputs = model(inputs)
                loss = criterion(outputs, labels)

                valid_dice = compute_dice(outputs.ge(mask_threshold).cpu(), labels.cpu())
                valid_dice_total += valid_dice
                valid_loss_total += loss.item()

            valid_loss_mean = valid_loss_total / len(valid_loader)
            valid_dice_mean = valid_dice_total / len(valid_loader)
            valid_curve.append(valid_loss_mean)
            valid_dice_curve.append(valid_dice_mean)

            print(
                f'Valid: Epoch: [{epoch:0>3}/{max_epoch:0>3}] mean_loss: {valid_loss_mean:.4f} mean_dice: {valid_dice_mean:.4f}')

    # 可视化
    if epoch % vis_interval == 0:
        with torch.no_grad():
            for idx, (inputs, labels) in enumerate(valid_loader):
                if idx > vis_num:
                    break
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                img_hwc = inputs.cpu().numpy()[0, :, :, :].transpose((1, 2, 0)).astype('uint8')
                plt.subplot(121).imshow(img_hwc)
                preds = outputs.ge(mask_threshold).cpu().numpy()[0, :, :, :].astype('uint8')
                mask_preds = preds.squeeze() * 255
                plt.subplot(122).imshow(mask_preds, cmap='gray')
                plt.show()

    # 可视化损失
    train_x, train_y = range(len(train_curve)), train_curve
    # 由于valid中记录的是epoch loss，需要对记录点进行转换到iterations
    valid_x, valid_y = np.arange(1, len(valid_curve) + 1) * train_batch_count * val_interval, valid_curve

    plt.plot(train_x, train_y, label='Train')
    plt.plot(valid_x, valid_y, label='Valid')

    plt.xlabel('Iteration')
    plt.ylabel('Loss')
    plt.title(f'Plot in {max_epoch} epochs')
    plt.legend(loc='upper right')
    plt.show()

    # 可视化dice指标
    train_x, train_y = range(len(train_dice_curve)), train_dice_curve
    valid_x, valid_y = np.arange(1, len(valid_dice_curve) + 1) * train_batch_count * val_interval, valid_dice_curve

    plt.plot(train_x, train_y, label='Train')
    plt.plot(valid_x, valid_y, label='Valid')

    plt.xlabel('Iteration')
    plt.ylabel('DICE')
    plt.title(f'Plot in {max_epoch} epochs')
    plt.legend(loc='upper right')
    plt.show()

    torch.cuda.empty_cache()
