# -*- coding: utf-8 -*-

'''
@Author   :   Corley Tang
@Contact  :   cutercorleytd@gmail.com
@Github   :   https://github.com/corleytd
@Time     :   2022-12-30 13:42
@Project  :   PyTorchBasic-rnn
'''

import torch
from torch import nn


class RNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        self.hidden_size = hidden_size
        self.u = nn.Linear(input_size, hidden_size)
        self.w = nn.Linear(hidden_size, hidden_size)
        self.v = nn.Linear(hidden_size, output_size)

        self.tanh = nn.Tanh()
        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, inputs, hidden):
        u_x = self.u(inputs)
        hidden = self.w(hidden)
        hidden = self.tanh(hidden + u_x)
        output = self.softmax(self.v(hidden))
        return output, hidden

    def init_hidden(self):
        return torch.zeros(1, self.hidden_size)
