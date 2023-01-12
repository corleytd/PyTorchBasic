# -*- coding: utf-8 -*-

'''
@Author   :   Corley Tang
@Contact  :   cutercorleytd@gmail.com
@Github   :   https://github.com/corleytd
@Time     :   2022-12-08 11:14
@Project  :   PyTorchBasic-hook_methods
'''

import torch
from torch import nn

from tools.common_tools import set_seed

set_seed()

flag = 2


class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(1, 2, 3)
        self.pool = nn.MaxPool2d(2, 2)

    def forward(self, x):
        x = self.conv(x)
        x = self.pool(x)
        return x


# 1.tensor hook get
if flag == 0:
    w = torch.tensor([1.], requires_grad=True)
    x = torch.tensor([2.], requires_grad=True)
    a = w + x
    b = w + 1
    y = a * b

    a_grad = []


    def grad_hook(grad):
        a_grad.append(grad)


    handle = a.register_hook(grad_hook)

    y.backward()

    # 查看梯度
    print(f'gradient:', w.grad, x.grad, a.grad, b.grad, y.grad)
    print(f'a.grad from hook', a_grad[0])

    handle.remove()

# 2.tensor hook set
if flag == 0:
    w = torch.tensor([1.], requires_grad=True)
    x = torch.tensor([2.], requires_grad=True)
    a = w + x
    b = w + 1
    y = a * b


    def grad_hook(grad):
        grad *= 2
        return grad ** 2


    handle = w.register_hook(grad_hook)

    y.backward()

    # 查看梯度
    print(f'w.grad from hook', w.grad)

    handle.remove()

# 3.Module.register_forward_hook
if flag == 1:
    input_block, fmap_block = [], []


    def forward_hook(module, inputs, output):
        input_block.append(inputs)
        fmap_block.append(output)


    # 初始化网络
    net = Net()
    net.conv.weight[0].detach().fill_(1)
    net.conv.weight[1].detach().fill_(2)
    net.conv.bias.detach().zero_()

    # 注册hook
    net.conv.register_forward_hook(forward_hook)

    # 前向传播
    fake_img = torch.ones((1, 1, 4, 4))  # BCHW
    output = net(fake_img)

    # 查看输出
    print(f'input shape: {input_block[0][0].shape}\ninput value: {input_block[0]}')
    print(f'feature map shape: {fmap_block[0].shape}\nfeature map value: {fmap_block[0]}')
    print(f'output shape: {output.shape}\noutput value: {output}')

# 3.Module.register_forward_pre_hook and Module.register_backward_hook
if flag == 2:
    def forward_pre_hook(module, inputs):
        print(f'forward_pre_hook input: {inputs}')


    def backward_hook(module, grad_inputs, grad_outputs):
        print(f'backward_hook grad_inputs: {grad_inputs}')
        print(f'backward_hook grad_outputs: {grad_outputs}')


    # 初始化网络
    net = Net()
    net.conv.weight[0].detach().fill_(1)
    net.conv.weight[1].detach().fill_(2)
    net.conv.bias.detach().zero_()

    # 注册hook
    net.conv.register_forward_pre_hook(forward_pre_hook)
    # net.conv.register_backward_hook(backward_hook)  # 已弃用
    net.conv.register_full_backward_hook(backward_hook)

    # 前向传播
    fake_img = torch.ones((1, 1, 4, 4))  # BCHW
    output = net(fake_img)
    target = torch.randn_like(output)

    loss_func = nn.L1Loss()
    loss = loss_func(target, output)
    loss.backward()
