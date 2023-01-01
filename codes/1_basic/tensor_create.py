# -*- coding: utf-8 -*-

'''
@Author   :   Corley Tang
@Contact  :   cutercorleytd@gmail.com
@Github   :   https://github.com/corleytd
@Time     :   2022-10-24 21:45
@Project  :   PyTorchBasic-tensor_create
'''

import numpy as np
import torch

torch.manual_seed(65537)

flag = False

if flag:
    arr = np.ones((5, 5))
    print(arr.dtype)
    t = torch.tensor(arr)
    print(t)
    t = torch.tensor(arr, device='cuda')
    print(t)

flag = False
if flag:
    arr = np.array([[1, 3, 5], [2, 4, 6]])
    t = torch.from_numpy(arr)
    print(arr, t)
    print(id(arr), id(t.data))
    # 共享内存
    arr[0, 0] = 7
    print(arr, t)
    print(id(arr), id(t.data))
    t[0, 0] = 8
    print(arr, t)
    print(id(arr.data), id(t.data))

flag = False
if flag:
    out_t = torch.tensor([1])
    t = torch.zeros((3, 4), out=out_t)  # 将得到的张量赋值给传入的变量
    print(t, out_t)
    print(id(t), id(out_t), id(t) == id(out_t))

flag = False
if flag:
    t = torch.full((3, 5), 10.2)
    print(t)

flag = False
if flag:
    t = torch.arange(0, 106, 17)
    print(t)

flag = False
if flag:
    t = torch.linspace(0, 100, 15)
    print(t)

flag = True
if flag:
    t = torch.normal(0., 3., size=(2, 5))
    print(t)
    t = torch.normal(0., torch.arange(1, 5, dtype=torch.float))
    print(t)
    t = torch.normal(torch.arange(1, 5, dtype=torch.float), 1)
    print(t)
    t = torch.normal(torch.arange(1, 5, dtype=torch.float), torch.arange(1, 5, dtype=torch.float))
    print(t)
