# -*- coding: utf-8 -*-

'''
@Author   :   Corley Tang
@Contact  :   cutercorleytd@gmail.com
@Github   :   https://github.com/corleytd
@Time     :   2022-12-15 17:11
@Project  :   PyTorchBasic-model_save_load
'''

import torch

from models.lenet import LeNet

model_path = 'output/model.pkl'
state_dict_path = 'output/state_dict.pkl'

flag = 3

# 1.保存
if flag == 0:
    model = LeNet(classes=2)

    print('before init:', model.conv1.weight[0, 0, 0])
    model.initialize_weights()
    print('after init:', model.conv1.weight[0, 0, 0])

    # 保存整个模型
    torch.save(model, model_path)

    # 保存模型参数
    state_dict = model.state_dict()
    torch.save(state_dict, state_dict_path)

# 2.加载完整模型
if flag == 1:
    model = torch.load(model_path)
    print(model)

# 3.加载state_dict
if flag == 2:
    state_dict = torch.load(state_dict_path)
    print(state_dict.keys())

# 4.模型加载state_dict
if flag == 3:
    model = LeNet(classes=2)

    print('before load:', model.conv1.weight[0, 0, 0])
    state_dict = torch.load(state_dict_path)
    model.load_state_dict(state_dict)
    print('after load:', model.conv1.weight[0, 0, 0])
