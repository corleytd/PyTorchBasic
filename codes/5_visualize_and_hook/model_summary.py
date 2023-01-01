# -*- coding: utf-8 -*-

'''
@Author   :   Corley Tang
@Contact  :   cutercorleytd@gmail.com
@Github   :   https://github.com/corleytd
@Time     :   2022-12-07 14:46
@Project  :   PyTorchBasic-model_summary
'''

from torchsummary import summary

from models.lenet import LeNet, CatDogLeNet

lenet = LeNet(2)
lenet.initialize_weights()

summary(lenet, input_size=(3, 32, 32), device='cpu')
print(lenet)

lenet = CatDogLeNet(2)
lenet.initialize_weights()

summary(lenet, input_size=(3, 112, 112), device='cpu')
print(lenet)
