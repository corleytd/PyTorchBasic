# -*- coding: utf-8 -*-

'''
@Author   :   Corley Tang
@Contact  :   cutercorleytd@gmail.com
@Github   :   https://github.com/corleytd
@Time     :   2022-12-22 18:26
@Project  :   PyTorchBasic-inference_resnet18
'''

import os
import random
import time

import torch
from PIL import Image
from matplotlib import pyplot as plt
from torch import nn
from torchsummary import summary
from torchvision import models, transforms

from tools.common_tools import set_seed

set_seed()

# 超参数
vis = True  # 是否进行可视化
vis_row_size = 4  # 每行展示的图片数
use_cuda = False
if use_cuda:
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
else:
    device = torch.device('cpu')

norm_mean = [0.485, 0.456, 0.406]
norm_std = [0.229, 0.224, 0.225]

inference_transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(norm_mean, norm_std)
])
classes = ['ants', 'bees']


def get_all_images(path):
    '''
    获取指定路径下的所有验证图片
    :param path: 图片路径
    :return: 打乱后的图片
    '''
    all_images = []
    for root, dirs, files in os.walk(path):
        if 'val' in root:
            for file in files:
                if file.endswith('.jpg'):
                    all_images.append(os.path.join(root, file))
    random.shuffle(all_images)
    return all_images[:len(all_images) // 2]


def get_model(ckpt_path, vis_model=False):
    '''
    获取模型
    :param ckpt_path: 检查点路径
    :param vis_model: 是否可视化模型
    :return: 加载了state_dict的模型
    '''
    model = models.resnet18()
    num_features = model.fc.in_features
    model.fc = nn.Linear(num_features, 2)

    checkpoint = torch.load(ckpt_path)
    model.load_state_dict(checkpoint)

    if vis_model:
        summary(model, input_size=(3, 224, 224), device='cpu')

    return model


def img_to_tensor(img_path, transform):
    img_rgb = Image.open(img_path).convert('RGB')
    img_tensor = transform(img_rgb)
    img_tensor.unsqueeze_(0)
    return img_rgb, img_tensor


# 1.获取数据
data_dir = '../../data/Ant_Bee_split'
images = get_all_images(data_dir)
img_count = len(images)

# 2.定义模型
model_path = 'output/resnet18.ckpt'
model = get_model(model_path, True)
model.to(device)

# 3.推理
total_time = 0
img_list, preds = [], []

model.eval()
with torch.no_grad():
    for idx, img in enumerate(images, 1):
        # 3.1 将图片转化为张量
        img_rgb, img_tensor = img_to_tensor(img, inference_transform)
        img_tensor = img_tensor.to(device)

        start = time.time()
        # 3.2 前向传播将张量转化为模型输出的向量
        output = model(img_tensor)
        end = time.time()

        # 3.3 可视化
        _, pred = torch.max(output, 1)
        pred_label = classes[int(pred)]

        if vis:
            img_list.append(img_rgb)
            preds.append(pred_label)

            if idx % (vis_row_size * vis_row_size) == 0 or idx == img_count:
                for i, (img_, pred) in enumerate(zip(img_list, preds), 1):
                    plt.subplot(vis_row_size, vis_row_size, i).imshow(img_)
                    plt.title(f'predict: {pred_label}')
                plt.show()
                plt.close()
                img_list, preds = [], []

        cur_time = end - start
        total_time += cur_time

        print(f'{idx}/{img_count}: {img} {cur_time:.3f}s')

if torch.cuda.is_available():
    print(f'GPU name: {torch.cuda.get_device_name()}')
print(f'\ndevice: {device} total time: {total_time:.2f}s mean: {total_time / img_count:.3f}s')

'''
CPU:
device: cuda total time: 4.02s mean: 0.053s
CPU:
device: cpu total time: 9.18s mean: 0.121s
'''
