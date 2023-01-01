# -*- coding: utf-8 -*-

'''
@Author   :   Corley Tang
@Contact  :   cutercorleytd@gmail.com
@Github   :   https://github.com/corleytd
@Time     :   2022-11-08 11:12
@Project  :   PyTorchBasic-convolution_layers
'''

from PIL import Image
from matplotlib import pyplot as plt
from torch import nn
from torchvision import transforms

from tools.common_tools import transform_invert, set_seed

set_seed()

# 1.加载图片
img_path = '../../data/imgs/greatwall.jpeg'
img = Image.open(img_path).convert('RGB')

# 转化成向量
img_transform = transforms.Compose([transforms.ToTensor()])
img_tensor = img_transform(img)
img_tensor.unsqueeze_(0)  # (C, H, W) -> (B, C, H, W)

# 2.卷积层
flag = 0
# 2维卷积
if flag:
    conv_layer = nn.Conv2d(3, 1, 3)
    nn.init.xavier_normal_(conv_layer.weight.data)

    # 训练
    img_conv = conv_layer(img_tensor)
# 转置卷积
else:
    conv_layer = nn.ConvTranspose2d(3, 1, 3, stride=2)
    nn.init.xavier_normal_(conv_layer.weight.data)

    # 训练
    img_conv = conv_layer(img_tensor)

# 3.可视化
print(f'卷积前尺寸：{img_tensor.shape}\n卷积后尺寸：{img_conv.shape}')
img_conv = transform_invert(img_conv[0, 0:1, ...], img_transform)
img_raw = transform_invert(img_tensor.squeeze(), img_transform)
plt.subplot(122).imshow(img_conv, cmap='gray')
plt.subplot(121).imshow(img_raw)
plt.show()
