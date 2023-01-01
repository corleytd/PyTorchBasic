# -*- coding: utf-8 -*-

'''
@Author   :   Corley Tang
@Contact  :   cutercorleytd@gmail.com
@Github   :   https://github.com/corleytd
@Time     :   2022-12-30 13:58
@Project  :   PyTorchBasic-rnn_name_classification
'''

import glob
import os
import random
import string
import time
import unicodedata

import matplotlib.pyplot as plt
import torch
from torch import nn

from models.rnn import RNN
from tools.common_tools import set_seed

set_seed()

all_letters = string.ascii_letters + ' .,;'
letter_count = len(all_letters)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
data_path = '../../data/names'
model_path = 'output/rnn_state_dict.pkl'

# 超参数
learning_rate = 5e-3
log_interval = 5000
vis_interval = 5000
iterations = 200000
hidden_size = 128


def unicode2ascii(s):
    return ''.join([c for c in unicodedata.normalize('NFD', s) if unicodedata.category(c) != 'Mn' and c in all_letters])


def read_lines(filename):
    with open(filename, encoding='utf-8') as f:
        lines = f.read().strip().split('\n')
    return [unicode2ascii(line) for line in lines]


def letter2tensor(letter):
    tensor = torch.zeros(1, letter_count)
    tensor[0, all_letters.find(letter)] = 1
    return tensor


def sent2tensor(sent):
    tensor = torch.zeros(len(sent), 1, letter_count)
    for idx, letter in enumerate(sent):
        tensor[idx, 0, all_letters.find(letter)] = 1
    return tensor


def category_from_output(output, categories):
    _, top_1 = output.topk(1)
    category_1 = top_1[0].item()
    return categories[category_1], category_1


def random_select(ls):
    return ls[random.randint(0, len(ls) - 1)]


def random_train_samples(categories, category_lines):
    category = random.choice(categories)
    line = random.choice(category_lines[category])
    category_tensor = torch.tensor([categories.index(category)])
    line_tensor = sent2tensor(line)
    return category, line, category_tensor, line_tensor


def time_spent(start):
    end = time.time()
    time_spent = int(end - start)
    minute, second = divmod(time_spent, 60)
    return f'{minute}m {second}s'


def train(model: RNN, criterion, category_tensor, line_tensor):
    hidden = model.init_hidden()
    hidden = hidden.to(device)
    sent_tensor = line_tensor.to(device)
    category_tensor = category_tensor.to(device)

    output = None
    for i in range(sent_tensor.shape[0]):
        output, hidden = model(sent_tensor[0], hidden)

    loss = criterion(output, category_tensor)
    loss.backward()

    # 手动实现参数的更新
    for p in model.parameters():
        p.data.add_(p.grad.data, alpha=-learning_rate)

    model.zero_grad()

    return output, loss.item()


def predict(name, model: RNN, n_preds=3):
    print(f'\n> {name}')
    with torch.no_grad():
        hidden = model.init_hidden().to(device)
        name_tensor = sent2tensor(name).to(device)
        output = None
        for idx in range(name_tensor.shape[0]):
            output, hidden = model(name_tensor[idx], hidden)

        top, top_idx = output.topk(n_preds, 1, True)
        for idx in range(n_preds):
            value = top[0][idx].item()
            category_idx = top_idx[0][idx].item()
            print(f'({value:.2f} {categories[category_idx]})')


# 1.获取数据
category_lines = {}
categories = []

for file in glob.glob(os.path.join(data_path, '*.txt')):
    category = os.path.splitext(os.path.basename(file))[0]
    categories.append(category)
    lines = read_lines(file)
    category_lines[category] = lines

num_category = len(categories)

# 2.定义模型
model = RNN(letter_count, hidden_size, num_category)
model.to(device)

# 3.定义损失函数
criterion = nn.NLLLoss()

# 4.迭代训练
cur_loss = 0
losses = []
start = time.time()
for iteration in range(1, iterations + 1):
    # 获取训练数据
    category, line, category_tensor, line_tensor = random_train_samples(categories, category_lines)

    # 前向传播
    output, loss = train(model, criterion, category_tensor, line_tensor)
    cur_loss += loss

    # 输出日志
    if iteration % log_interval == 0:
        guess, guess_idx = category_from_output(output, categories)
        res = '√' if guess == category else f'× ({category})'
        print(
            f'Train: Iter: {iteration:>7}\ttime: {time_spent(start):>8s}\tloss: {loss:.4f}\tname: {line:>15s}\tpred: {guess:>15s}\tlabel: {res:>15s}')

    # 记录损失
    if iteration % vis_interval == 0:
        losses.append(cur_loss / vis_interval)
        cur_loss = 0

torch.save(model.state_dict(), model_path)
plt.plot(losses)
plt.show()

# 5.预测
predict('Yue Tingsong', model)
predict('Yu tingsong', model)
predict('yutingsong', model)
predict('Corley Tang', model)
