# -*- coding: utf-8 -*-

'''
@Author   :   Corley Tang
@Contact  :   cutercorleytd@gmail.com
@Github   :   https://github.com/corleytd
@Time     :   2022-10-26 21:10
@Project  :   PyTorchBasic-auto_grad
'''

import torch

torch.manual_seed(65537)

flag = False
if flag:
    w = torch.tensor([1.], requires_grad=True)
    x = torch.tensor([2.], requires_grad=True)

    a = torch.add(w, x)
    b = torch.add(w, 1)
    y = torch.mul(a, b)

    print(w.grad, x.grad)
    y.backward(retain_graph=True)
    print(w.grad, x.grad)
    y.backward()

flag = False
if flag:
    w = torch.tensor([1.], requires_grad=True)
    x = torch.tensor([2.], requires_grad=True)

    a = torch.add(w, x)
    b = torch.add(w, 1)

    y1 = torch.mul(a, b)
    y2 = torch.add(a, b)

    loss = torch.cat([y1, y2], dim=0)
    grad_ten = torch.tensor([1., 2.])

    print(w.grad, x.grad)
    loss.backward(gradient=grad_ten, retain_graph=True)
    print(w.grad, x.grad)

flag = False
if flag:
    x = torch.tensor([3.], requires_grad=True)
    y = x ** 2

    grad_1 = torch.autograd.grad(y, x, create_graph=True)
    print(grad_1)

    grad_2 = torch.autograd.grad(grad_1[0], x)
    print(grad_2)

flag = False
if flag:
    w = torch.tensor([1.], requires_grad=True)
    x = torch.tensor([2.], requires_grad=True)

    a = torch.add(w, x)
    b = torch.add(w, 1)
    y = torch.mul(a, b)

    for i in range(3):
        y.backward(retain_graph=True)
        print(w.grad, x.grad)
        w.grad.zero_()
        x.grad.zero_()

flag = False
if flag:
    w = torch.tensor([1.], requires_grad=True)
    x = torch.tensor([2.], requires_grad=True)

    a = torch.add(w, x)
    b = torch.add(w, 1)
    y = torch.mul(a, b)

    print(w.requires_grad, x.requires_grad, a.requires_grad, b.requires_grad, y.requires_grad)

flag = False
if flag:
    w = torch.tensor([1.], requires_grad=True)
    x = torch.tensor([2.], requires_grad=True)

    a = torch.add(w, x)
    b = torch.add(w, 1)
    y = torch.mul(a, b)

    # w.add_(1)
    y.backward()

flag = True
if flag:
    a = torch.ones((1,))
    print(id(a), a)

    a = a + 1
    print(id(a), a)

    a += 1
    print(id(a), a)

    a.add_(1)
    print(id(a), a)
