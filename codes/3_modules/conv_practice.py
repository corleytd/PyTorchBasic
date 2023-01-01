# -*- coding: utf-8 -*-

'''
@Author   :   Corley Tang
@Contact  :   cutercorleytd@gmail.com
@Github   :   https://github.com/corleytd
@Time     :   2022-11-08 14:27
@Project  :   PyTorchBasic-conv_practice
'''

import torch
from PIL import Image
from matplotlib import pyplot as plt
from torch import nn
from torchvision import transforms

from tools.common_tools import transform_invert

# 输入
inputs = torch.stack([torch.full((5, 5), 1.), torch.full((5, 5), 2.), torch.full((5, 5), 3.)])
inputs.unsqueeze_(0)

# 1.1
print(f'{"1.1":~^80}')
conv1 = nn.Conv2d(3, 1, kernel_size=3, stride=1, bias=False)
conv2 = nn.Conv2d(3, 1, kernel_size=3, stride=1, bias=False)
conv1.weight.data[:] = 1
conv2.weight.data[:] = 2
outputs = conv1(inputs)
print(f'卷积前尺寸：{inputs.shape} 卷积后尺寸：{outputs.shape}')
print(outputs)
outputs = conv2(inputs)
print(outputs)

# 1.2
print(f'\n{"1.2":~^80}')
conv1 = nn.Conv2d(3, 1, kernel_size=3, stride=1, bias=False, padding=1)
conv2 = nn.Conv2d(3, 1, kernel_size=3, stride=1, bias=False, padding=1)
conv1.weight.data[:] = 1
conv2.weight.data[:] = 2
outputs = conv1(inputs)
print(f'卷积前尺寸：{inputs.shape} 卷积后尺寸：{outputs.shape}')
print(outputs)
outputs = conv2(inputs)
print(outputs)

# 2
print(f'\n{"2":~^80}')
flag = 0
# 加载图片
img_path = '../../data/imgs/greatwall.jpeg'
img = Image.open(img_path).convert('RGB')

# 转化成向量
img_transform = transforms.Compose([transforms.ToTensor()])
img_tensor = img_transform(img)
img_tensor.unsqueeze_(0)  # (C, H, W) -> (B, C, H, W)
img_tensor.unsqueeze_(2)  # (B, C, H, W) -> (B, C, D, H, W)

# 卷积层
if flag:
    conv_layer = nn.Conv3d(3, 1, (1, 3, 3), padding=(1, 0, 0))
else:
    conv_layer = nn.Conv3d(3, 1, (3, 3, 3), padding=(1, 0, 0))
nn.init.xavier_normal_(conv_layer.weight.data)

# 训练
img_conv = conv_layer(img_tensor)

print(f'卷积前尺寸：{img_tensor.shape}\n卷积后尺寸：{img_conv.shape}')
img_conv = transform_invert(img_conv.squeeze(0).squeeze(1), img_transform)
img_raw = transform_invert(img_tensor.squeeze(), img_transform)
plt.subplot(122).imshow(img_conv, cmap='gray')
plt.subplot(121).imshow(img_raw)
plt.show()
