# -*- coding: utf-8 -*-

'''
@Author   :   Corley Tang
@Contact  :   cutercorleytd@gmail.com
@Github   :   https://github.com/corleytd
@Time     :   2022-10-26 18:05
@Project  :   PyTorchBasic-computational_graph
'''

import torch

w = torch.tensor([1.], requires_grad=True)
x = torch.tensor([2.], requires_grad=True)

a = torch.add(w, x)
b = torch.add(w, 1)
b.retain_grad()
y = torch.mul(a, b)

y.backward()
print(w.grad)

print(w.is_leaf, x.is_leaf, a.is_leaf, b.is_leaf, y.is_leaf)
print(w.grad, x.grad, a.grad, b.grad, y.grad)
print(w.grad_fn, x.grad_fn, a.grad_fn, b.grad_fn, y.grad_fn)
