# -*- coding: utf-8 -*-

'''
@Author   :   Corley Tang
@Contact  :   cutercorleytd@gmail.com
@Github   :   https://github.com/corleytd
@Time     :   2022-11-13 22:14
@Project  :   PyTorchBasic-loss_function
'''

import numpy as np
import torch
from matplotlib import pyplot as plt
from torch import nn
from torch.nn import functional as F

flag = 14

# 制造数据
inputs = torch.tensor([[1., 2], [1, 3], [1, 3]])
target = torch.tensor([0, 1, 1], dtype=torch.long)

# 1.1交叉熵损失函数：reduction
if flag == 0:
    # 定义损失函数
    loss_f_none = nn.CrossEntropyLoss(reduction='none')
    loss_f_sum = nn.CrossEntropyLoss(reduction='sum')
    loss_f_mean = nn.CrossEntropyLoss(reduction='mean')

    # 前向传播
    loss_none = loss_f_none(inputs, target)
    loss_sum = loss_f_sum(inputs, target)
    loss_mean = loss_f_mean(inputs, target)

    print('Cross Entropy Loss:\n', loss_none, loss_sum, loss_mean)

# 手动计算
if flag == 0:
    idx = 0
    input_1 = inputs.detach().numpy()[idx]
    target_1 = target.numpy()[idx]

    # 第1项
    x_class = input_1[target_1]

    # 第2项
    sigma_exp_x = np.sum(list(map(np.exp, input_1)))
    sigma_exp_x_log = np.log(sigma_exp_x)

    # Loss结果
    loss = -x_class + sigma_exp_x_log

    print('第一个样本Loss为：', loss)

# 1.2交叉熵损失函数：weight
if flag == 1:
    weights = torch.tensor([1., 2])

    loss_f_none = nn.CrossEntropyLoss(weights, reduction='none')
    loss_f_sum = nn.CrossEntropyLoss(weights, reduction='sum')
    loss_f_mean = nn.CrossEntropyLoss(weights, reduction='mean')

    loss_none = loss_f_none(inputs, target)
    loss_sum = loss_f_sum(inputs, target)
    loss_mean = loss_f_mean(inputs, target)

    print('Weights:', weights)
    print('Cross Entropy Loss:\n', loss_none, loss_sum, loss_mean)

# 手动计算
if flag == 1:
    weights = torch.tensor([1., 2])

    loss_f_none = nn.CrossEntropyLoss(reduction='none')
    loss_none = loss_f_none(inputs, target)

    loss_weighted, weights_sum = 0, 0
    for idx, class_ in enumerate(target.numpy()):
        weights_sum += weights[class_]
        loss_weighted += loss_none[idx] * weights[class_]

    loss_weighted /= weights_sum

    print(loss_weighted)

# 2.负对数似然损失函数
if flag == 2:
    weights = torch.tensor([1., 1])

    loss_f_none = nn.NLLLoss(weights, reduction='none')
    loss_f_sum = nn.NLLLoss(weights, reduction='sum')
    loss_f_mean = nn.NLLLoss(weights, reduction='mean')

    loss_none = loss_f_none(inputs, target)
    loss_sum = loss_f_sum(inputs, target)
    loss_mean = loss_f_mean(inputs, target)

    print('NLL Loss:\n', loss_none, loss_sum, loss_mean)

inputs = torch.tensor([[1., 2], [2, 2], [3, 4], [4, 5]])
target = torch.tensor([[1., 0], [1, 0], [0, 1], [0, 1]])

# 3.1二分类交叉熵损失函数
if flag == 3:
    inputs_s = torch.sigmoid(inputs)

    loss_f_none = nn.BCELoss(reduction='none')
    loss_f_sum = nn.BCELoss(reduction='sum')
    loss_f_mean = nn.BCELoss(reduction='mean')

    loss_none = loss_f_none(inputs_s, target)
    loss_sum = loss_f_sum(inputs_s, target)
    loss_mean = loss_f_mean(inputs_s, target)

    print('BCE Loss:\n', loss_none, loss_sum, loss_mean)

# 3.2手动计算
if flag == 3:
    idx, class_ = 0, 0

    inputs_s = torch.sigmoid(inputs)
    x_i = inputs_s[idx, class_].item()
    y_i = target[idx, class_].item()

    loss_i = -y_i * np.log(x_i) if y_i else -(1 - y_i) * np.log(1 - x_i)

    print('Input: ', inputs_s)
    print('Loss 1,1', loss_i)

# 4.1带Logits的二分类交叉熵损失函数
if flag == 4:
    loss_f_none = nn.BCEWithLogitsLoss(reduction='none')
    loss_f_sum = nn.BCEWithLogitsLoss(reduction='sum')
    loss_f_mean = nn.BCEWithLogitsLoss(reduction='mean')

    loss_none = loss_f_none(inputs, target)
    loss_sum = loss_f_sum(inputs, target)
    loss_mean = loss_f_mean(inputs, target)

    print('BCE Loss:\n', loss_none, loss_sum, loss_mean)

# 4.2带Logits的二分类交叉熵损失函数：pos_weight
if flag == 4:
    pos_weight = torch.tensor([2.])

    loss_f_none = nn.BCEWithLogitsLoss(reduction='none', pos_weight=pos_weight)
    loss_f_sum = nn.BCEWithLogitsLoss(reduction='sum', pos_weight=pos_weight)
    loss_f_mean = nn.BCEWithLogitsLoss(reduction='mean', pos_weight=pos_weight)

    loss_none = loss_f_none(inputs, target)
    loss_sum = loss_f_sum(inputs, target)
    loss_mean = loss_f_mean(inputs, target)

    print('Pos Weight:', pos_weight)
    print('BCE Loss:\n', loss_none, loss_sum, loss_mean)

inputs = torch.ones((2, 2))
target = torch.ones((2, 2)) * 3

# 5.L1损失函数
if flag == 5:
    loss_func = nn.L1Loss(reduction='none')
    loss = loss_func(inputs, target)
    print('L1 Loss:', loss)

# 6.均方误差损失函数
if flag == 6:
    loss_func = nn.MSELoss(reduction='none')
    loss = loss_func(inputs, target)
    print('MES Loss:', loss)

inputs = torch.linspace(-3, 3, 500)
target = torch.zeros_like(inputs)

# 7.带平滑的L1损失函数
if flag == 7:
    loss_func = nn.SmoothL1Loss(reduction='none')
    loss_l1_smooth = loss_func(inputs, target)
    loss_l1 = np.abs(inputs - target)

    plt.plot(inputs.numpy(), loss_l1_smooth.numpy(), label='Smooth L1 Loss')
    plt.plot(inputs.numpy(), loss_l1, label='L1 Loss')

    plt.xlabel('x_i - y_i')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid()
    plt.show()

inputs = torch.randn((2, 2))
target = torch.randn((2, 2))

# 8.泊松负对数似然损失函数
if flag == 8:
    loss_func = nn.PoissonNLLLoss(reduction='none')
    loss = loss_func(inputs, target)

    print(f'input: {inputs}\ntarget: {target}\nPoisson NLL Loss: {loss}')

# 手动计算
if flag == 8:
    loss = torch.exp(inputs) - target * inputs
    print(f'Loss by hand: {loss}')

inputs = torch.tensor([[0.13, 0.70, 0.81], [0.73, 0.52, 0.72]])
target = torch.tensor([[0.05, 0.05, 0.9], [0.6, 0.1, 0.3]])

# 9.KL散度损失函数
if flag == 9:
    inputs_ls = F.log_softmax(inputs, -1)

    loss_f_none = nn.KLDivLoss(reduction='none')
    loss_f_sum = nn.KLDivLoss(reduction='sum')
    loss_f_bs_mean = nn.KLDivLoss(reduction='batchmean')

    loss_none = loss_f_none(inputs_ls, target)
    loss_sum = loss_f_sum(inputs_ls, target)
    loss_bs_mean = loss_f_bs_mean(inputs_ls, target)

    print(f'loss_none:\n{loss_none}\nloss_sum:\n{loss_sum}\nloss_bs_mean:\n{loss_bs_mean}')

# 手动计算
if flag == 9:
    inputs_ls = F.log_softmax(inputs, -1)

    loss = target * (torch.log(target) - inputs_ls)

    print(f'Loss by hand:\n{loss}')

x1 = torch.tensor([[1.], [2], [3]])
x2 = torch.tensor([[2.], [2], [2]])
target = torch.tensor([1., 1, -1])

# 10.MarginRankingLoss
if flag == 10:
    loss_func = nn.MarginRankingLoss(reduction='none')

    loss = loss_func(x1, x2, target)

    print(loss)

inputs = torch.tensor([[0.1, 0.2, 0.4, 0.8]])
target = torch.tensor([[0, 3, -1, -1]])

# 11.MultiLabelMarginLoss
if flag == 11:
    loss_func = nn.MultiLabelMarginLoss(reduction='none')

    loss = loss_func(inputs, target)

    print(loss)

# 手动计算
if flag == 11:
    x = inputs[0]
    l0 = (1 - (x[0] - x[1])) + (1 - (x[0] - x[2]))  # 0
    l3 = (1 - (x[3] - x[1])) + (1 - (x[3] - x[2]))  # 3
    loss = (l0 + l3) / x.shape[0]
    print(loss)

inputs = torch.tensor([[0.3, 0.7], [.5, 0.5]])
target = torch.tensor([[-1., 1], [1, -1]])

# 12.SoftMarginLoss
if flag == 12:
    loss_func = nn.SoftMarginLoss(reduction='none')

    loss = loss_func(inputs, target)

    print(loss)

# 手动计算
if flag == 12:
    loss = torch.log(1 + torch.exp(-target * inputs))

    print(loss)

inputs = torch.tensor([[0.3, 0.7, 0.8]])
target = torch.tensor([[0., 1, 1]])

# 13.MultiLabelSoftMarginLoss
if flag == 13:
    loss_func = nn.MultiLabelSoftMarginLoss(reduction='none')

    loss = loss_func(inputs, target)
    print(loss)

# 手动计算
if flag == 13:
    target_1 = target == 1
    target_0 = target == 0
    loss = torch.log(1 / (1 + torch.exp(-inputs[target_1]))).sum() + torch.log(
        torch.exp(-inputs[target_0]) / (1 + torch.exp(-inputs[target_0])))
    loss /= -target.shape[-1]
    print(loss)

inputs = torch.tensor([[0.1, 0.7, 0.2], [0.2, 0.2, 0.6]])
target = torch.tensor([1, 2])

# 14.MultiMarginLoss
if flag == 14:
    loss_func = nn.MultiMarginLoss(reduction='none')
    loss = loss_func(inputs, target)
    print(loss)

# 手动计算
if flag == 14:
    x = inputs[0]
    margin = 1

    l_0 = max(0, margin - x[1] + x[0])
    l_2 = max(0, margin - x[1] + x[2])
    loss = (l_0 + l_2) / x.shape[0]
    print(loss)

anchor = torch.tensor([[1., 2, 3]])
pos = torch.tensor([[1., 1.5, 3]])
neg = torch.tensor([[2, 2, 2.5]])

# 15.TripletMarginLoss
if flag == 15:
    loss_func = nn.TripletMarginLoss()

    loss = loss_func(anchor, pos, neg)
    print(loss)

if flag == 15:
    d_ap = torch.norm(anchor - pos, p=2)
    d_an = torch.norm(anchor - neg, p=2)
    diff = d_ap - d_an + 1
    loss = max(diff.data, 0)
    print(loss)

inputs = torch.tensor([[1, 0.8, 0.5]])
target = torch.tensor([[1, 1, -1]])

# 16.HingeEmbeddingLoss
if flag == 16:
    loss_func = nn.HingeEmbeddingLoss(reduction='none')

    loss = loss_func(inputs, target)
    print(loss)

if flag == 16:
    margin = 1
    loss = inputs.clone()
    target_0 = target == -1
    loss[target_0] = torch.max(torch.tensor(0.), margin - inputs[target_0])
    print(loss)

inputs1 = torch.tensor([[0.3, 0.5, 0.7], [0.6, 0.4, 0.2]])
inputs2 = torch.tensor([[0.2, 0.3, 0.5], [0.4, 0.3, 0.3]])
target = torch.tensor([1., -1])

# 17.CosineEmbeddingLoss
if flag == 17:
    loss_func = nn.CosineEmbeddingLoss(reduction='none')

    loss = loss_func(inputs1, inputs2, target)
    print(loss)

if flag == 17:
    cos_sim = torch.cosine_similarity(inputs1, inputs2)
    target_1 = target == 1
    target_0 = target == -1
    loss = cos_sim.clone()
    loss[target_1] = 1 - cos_sim[target_1]
    loss[target_0] = torch.max(torch.tensor(0.), cos_sim[target_0] - 0)
    print(loss)

# 18.CTCLoss
if flag == 18:
    T = 50  # 输入序列长度
    C = 20  # 类别数（包括空白）
    N = 16  # 批量大小
    S = 30  # 批次中最长的目标序列长度
    S_min = 10  # 最小目标长度

    # 对于*size=(T, N, C)，初始化输入向量的随机批次
    inputs = torch.randn(T, N, C).log_softmax(2).detach().requires_grad_()
    # 初始化随机批目标（0=空白，1:C=类）
    target = torch.randint(1, C, (N, S))

    input_lengths = torch.full((N,), T)
    target_lengths = torch.randint(S_min, S, (N,))

    ctc_loss = nn.CTCLoss()
    loss = ctc_loss(inputs, target, input_lengths, target_lengths)
    print(loss)
