# -*- coding: utf-8 -*-

'''
@Author   :   Corley Tang
@Contact  :   cutercorleytd@gmail.com
@Github   :   https://github.com/corleytd
@Time     :   2022-11-08 12:01
@Project  :   PyTorchBasic-pool_linear_layers
'''

import torch
from PIL import Image
from torch import nn
from torchvision import transforms

from tools.common_tools import set_seed

set_seed()

# 1.加载图片
img_path = '../../data/imgs/greatwall.jpeg'
img = Image.open(img_path).convert('RGB')

# 转化成向量
img_transform = transforms.Compose([transforms.ToTensor()])
img_tensor = img_transform(img)
img_tensor.unsqueeze_(0)  # (C, H, W) -> (B, C, H, W)

flag = 4

# 2.池化层
# 最大池化
if flag == 0:
    maxpool_layer = nn.MaxPool2d((2, 2), stride=(2, 2))

    img_pool = maxpool_layer(img_tensor)
# 平均池化
elif flag == 1:
    avgpool_layer = nn.AvgPool2d((2, 2), stride=(2, 2))

    img_pool = avgpool_layer(img_tensor)
elif flag == 2:
    img_tensor = torch.ones((1, 1, 7, 7))
    avgpool_layer = nn.AvgPool2d((3, 3), stride=(3, 3), divisor_override=2)

    img_pool = avgpool_layer(img_tensor)
    print(f'raw_img: {img_tensor}\npooling_img: {img_pool}')

# 可视化
# print(f'池化前尺寸：{img_tensor.shape}\n池化后尺寸：{img_pool.shape}')
# img_pool = transform_invert(img_pool[0, 0:1, ...], img_transform)
# img_raw = transform_invert(img_tensor.squeeze(), img_transform)
# plt.subplot(122).imshow(img_pool)
# plt.subplot(121).imshow(img_raw)
# plt.show()

# 3.反池化层
if flag == 3:
    img_tensor = torch.randint(1, 20, (1, 1, 4, 4), dtype=torch.float32)

    # pooling
    maxpool_layer = nn.MaxPool2d((2, 2), stride=(2, 2), return_indices=True)
    img_pool, indices = maxpool_layer(img_tensor)

    # unpooling
    img_reconstruct = torch.randn_like(img_pool)
    maxunpool_layer = nn.MaxUnpool2d((2, 2), stride=(2, 2))
    img_unpool = maxunpool_layer(img_reconstruct, indices)

    print(f'raw_img:\n{img_tensor}\nimg_pool:\n{img_pool}')
    print(f'img_reconstruction:\n{img_reconstruct}\nimg_unpool:\n{img_unpool}')

# 4.线性层
elif flag == 4:
    inputs = torch.tensor([[1., 2, 3]])
    linear_layer = nn.Linear(3, 4)
    linear_layer.weight.data = torch.tensor([
        [1., 1, 1],
        [2, 2, 2],
        [3, 3, 3],
        [4, 4, 4]
    ])
    linear_layer.bias.data.fill_(0.5)

    outputs = linear_layer(inputs)

    print(inputs, inputs.shape)
    print(linear_layer.weight.data, linear_layer.weight.data.shape)
    print(outputs, outputs.shape)
