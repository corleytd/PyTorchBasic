# -*- coding: utf-8 -*-

'''
@Author   :   Corley Tang
@Contact  :   cutercorleytd@gmail.com
@Github   :   https://github.com/corleytd
@Time     :   2022-12-26 10:29
@Project  :   PyTorchBasic-unet_portrait_inference
'''

import os
import random
import time
from collections import OrderedDict

import numpy as np
import torch
from PIL import Image
from matplotlib import pyplot as plt

from models.unet import UNet
from tools.common_tools import set_seed

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
set_seed()

valid_path = '../../data/Portrait_split/valid'
test_path = '../../data/Portrait_split/test'
model_path = 'output/unet_epoch_200.ckpt'
time_total = 0
mask_threshold = 0.5

# 1.获取所有图片
images = [os.path.join(valid_path, image) for image in os.listdir(valid_path) if 'matte' not in image]
images += [os.path.join(test_path, image) for image in os.listdir(test_path)]
random.shuffle(images)
img_count = len(images)

# 2.定义模型
model = UNet(in_channels=3, out_channels=1, num_features=32)
model_checkpoint = torch.load(model_path, map_location='cpu')['model_state_dict']
new_state_dict = OrderedDict()
for k, v in model_checkpoint.items():
    new_state_dict[k.replace('module.', '')] = v
model.load_state_dict(new_state_dict)
model.to(device)
model.eval()

# 3.迭代推理
for idx, image in enumerate(images, 1):
    # 3.1 将图片转化为张量
    img_rgb = Image.open(image).convert('RGB')
    img_rgb = img_rgb.resize((224, 224))  # HWC
    img_arr = np.array(img_rgb)
    img_arr = img_arr.transpose((2, 0, 1))  # CHW

    img_tensor = torch.tensor(img_arr).to(torch.float)
    img_tensor.unsqueeze_(0)
    img_tensor = img_tensor.to(device)

    # 3.2 前向传播
    start = time.time()
    outputs = model(img_tensor)
    end = time.time()

    # 3.3 可视化
    img_hwc = img_tensor.cpu().numpy()[0, :, :, :].transpose((1, 2, 0)).astype('uint8')
    plt.subplot(121).imshow(img_hwc)
    preds = outputs.ge(mask_threshold).cpu().numpy()[0, :, :, :].astype('uint8')
    mask_preds = preds.squeeze() * 255
    plt.subplot(122).imshow(mask_preds, cmap='gray')
    plt.show()

    cur_time = end - start
    time_total += cur_time

    print(f'{idx}/{img_count} {image} {cur_time:.3f}s')

print(f'total time: {time_total:.4f}s')
