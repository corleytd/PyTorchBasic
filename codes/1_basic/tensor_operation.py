# -*- coding: utf-8 -*-

'''
@Author   :   Corley Tang
@Contact  :   cutercorleytd@gmail.com
@Github   :   https://github.com/corleytd
@Time     :   2022-10-25 11:21
@Project  :   PyTorchBasic-tensor_operation
'''

import torch

torch.manual_seed(65537)

flag = False
if flag:
    t = torch.ones((2, 3))
    t0 = torch.cat([t, t], dim=0)  # 不扩展维度
    t1 = torch.cat([t, t, t], dim=1)
    print(t0, t0.shape)
    print(t1, t1.shape)

flag = False
if flag:
    t = torch.ones((2, 3))
    t_s = torch.stack([t, t], 2)  # 扩展维度
    print(t_s, t_s.shape)
    t_s = torch.stack([t, t], 0)
    print(t_s, t_s.shape)
    t_s = torch.stack([t, t, t], 0)
    print(t_s, t_s.shape)

flag = False
if flag:
    t = torch.ones((2, 7))
    tens = torch.chunk(t, 3, 1)  # 平均切分
    for idx, ten in enumerate(tens):
        print(idx, ten, ten.shape)

flag = False
if flag:
    t = torch.ones((2, 7))
    tens = torch.split(t, 3, dim=1)  # 按长度切分
    for idx, ten in enumerate(tens):
        print(idx, ten, ten.shape)
    print()
    tens = torch.split(t, [2, 3, 2], 1)
    for idx, ten in enumerate(tens):
        print(idx, ten, ten.shape)

flag = False
if flag:
    t = torch.randint(0, 10, size=(3, 4))
    idx = torch.tensor([0, 1, 3], dtype=torch.long)
    t_sel = torch.index_select(t, 1, idx)
    print(t)
    print(t_sel)

flag = False
if flag:
    t = torch.randint(0, 10, size=(3, 4))
    mask = t.ge(5)
    t_sel = torch.masked_select(t, mask)
    print(t)
    print(t_sel)

flag = False
if flag:
    t = torch.randperm(15)
    t_res = torch.reshape(t, (3, 5))
    print(t)
    print(t_res, end='\n\n')
    t_res = torch.reshape(t, (-1, 3))
    print(t_res, end='\n\n')

    t[0] = 20
    print(t)
    print(t_res, end='\n\n')

    print(id(t.data), id(t_res.data))

flag = False
if flag:
    t = torch.rand(2, 3, 4)
    t_t = torch.transpose(t, 1, 2)
    print(t.shape, t_t.shape)

flag = False
if flag:
    t = torch.rand((1, 2, 3, 1))
    t_sq = torch.squeeze(t)
    t0 = torch.squeeze(t, 0)
    t1 = torch.squeeze(t, 1)
    print(t.shape, t_sq.shape, t0.shape, t1.shape)

flag = True
if flag:
    t1 = torch.randn((2, 3))
    t2 = torch.ones_like(t1)
    t_add = torch.add(t1, t2)
    print(t1)
    print(t2)
    print(t_add)
    t_add = torch.add(t1, t2, alpha=10)
    print(t_add)
