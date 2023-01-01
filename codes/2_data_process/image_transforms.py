# -*- coding: utf-8 -*-

'''
@Author   :   Corley Tang
@Contact  :   cutercorleytd@gmail.com
@Github   :   https://github.com/corleytd
@Time     :   2022-11-02 17:47
@Project  :   PyTorchBasic-image_transforms
'''

import os

import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from torchvision import transforms

from custom_transforms import SaltPepperNoise
from tools.common_tools import set_seed, transform_invert
from tools.datasets import RMBDataset

set_seed()  # 设置随机种子

# 设置超参数
MAX_EPOCH = 6
BATCH_SIZE = 16

# 1.数据处理
split_path = '../../data/RMB_split'
train_path = os.path.join(split_path, 'train')
valid_path = os.path.join(split_path, 'valid')

norm_mean = [0.485, 0.456, 0.406]
norm_std = [0.229, 0.224, 0.225]

train_transform = transforms.Compose([
    transforms.Resize((224, 224)),

    # 1.CenterCrop
    # transforms.CenterCrop(512),

    # 2.RandomCrop
    # transforms.RandomCrop(224, padding=16),
    # transforms.RandomCrop(224, padding=(16, 64)),
    # transforms.RandomCrop(224, padding=16, fill=(12, 76, 125)),
    # transforms.RandomCrop(512, pad_if_needed=True),
    # transforms.RandomCrop(224, padding=64, padding_mode='edge'),
    # transforms.RandomCrop(224, padding=64, padding_mode='reflect'),
    # transforms.RandomCrop(1024, padding=1024, padding_mode='symmetric'),

    # 3.RandomResizedCrop
    # transforms.RandomResizedCrop(224),
    # transforms.RandomResizedCrop(224, scale=(0.5, 0.51)),

    # 4.FiveCrop
    # transforms.FiveCrop(112),
    # transforms.Lambda(lambda crops: torch.stack([transforms.ToTensor()(crop) for crop in crops])),

    # 5.TenCrop
    # transforms.TenCrop(112, vertical_flip=True),
    # transforms.TenCrop(112, vertical_flip=False),
    # transforms.Lambda(lambda crops: torch.stack([transforms.ToTensor()(crop) for crop in crops])),

    # 6.HorizontalFlip
    # transforms.RandomHorizontalFlip(1),

    # 7.VerticalFlip
    # transforms.RandomVerticalFlip(0.6),

    # 8.RandomRotation
    # transforms.RandomRotation(90),
    # transforms.RandomRotation(90, expand=True),
    # transforms.RandomRotation(90, center=(0, 0)),
    # transforms.RandomRotation(90, center=(0, 0), expand=True),

    # 9.Pad
    # transforms.Pad(padding=32, fill=(152, 77, 41), padding_mode='constant'),
    # transforms.Pad(padding=(8, 64), fill=(152, 77, 41), padding_mode='constant'),
    # transforms.Pad(padding=(8, 16, 32, 64), fill=(152, 77, 41), padding_mode='constant'),
    # transforms.Pad(padding=(8, 16, 32, 64), fill=(152, 77, 41), padding_mode='symmetric'),

    # 10.ColorJitter
    # transforms.ColorJitter(brightness=0.5),
    # transforms.ColorJitter(contrast=0.5),
    # transforms.ColorJitter(saturation=0.5),
    # transforms.ColorJitter(hue=0.3),

    # 11.GrayScale
    # transforms.Grayscale(num_output_channels=3),

    # 12.Affine
    # transforms.RandomAffine(degrees=30),
    # transforms.RandomAffine(degrees=0, translate=(0.2, 0.3), fillcolor=(16, 92, 251)),
    # transforms.RandomAffine(degrees=0, scale=(0.3, 0.7)),
    # transforms.RandomAffine(degrees=0, shear=(0, 30, 0, 45)),
    # transforms.RandomAffine(degrees=0, shear=30, fill=(225, 6, 2)),

    # 13.RandomErasing
    # transforms.ToTensor(),
    # transforms.RandomErasing(0.7, scale=(0.02, 0.33), ratio=(0.3, 3.3), value=(251/255, 114/255, 153/255)),
    # transforms.RandomErasing(0.7, scale=(0.02, 0.33), ratio=(0.3, 3.3), value='random'),

    # 14.RandomChoice
    # transforms.RandomChoice([transforms.RandomVerticalFlip(0.8), transforms.RandomHorizontalFlip(0.8)]),

    # 15.RandomApply
    # transforms.RandomApply([
    #     transforms.RandomAffine(degrees=0, shear=45, fill=(78, 110, 242)),
    #     transforms.Grayscale(num_output_channels=3)
    # ], 0.7),

    # 16.RandomOrder
    # transforms.RandomOrder([
    #     transforms.RandomRotation(30),
    #     transforms.Pad(padding=32),
    #     transforms.RandomAffine(degrees=0, translate=(0.1, 0.3), scale=(0.8, 1.2))
    # ]),

    # 17.SaltPepperNoise
    SaltPepperNoise(0.9, p=0.5),

    transforms.ToTensor(),
    transforms.Normalize(norm_mean, norm_std)
])

# 创建dataset
train_data = RMBDataset(train_path, train_transform)

# 构建DataLoader
train_loader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True)

for epoch in range(1, MAX_EPOCH + 1):
    for iteration, data in enumerate(train_loader, 1):
        inputs, labels = data  # inputs: (B, C, H, W)

        img_tensor = inputs[0, ...]  # img_tensor: (C, H, W)
        img = transform_invert(img_tensor, train_transform)
        plt.imshow(img)
        plt.show()
        plt.pause(0.5)
        plt.close()

        # ncrops = inputs.shape[1]  # inputs: (B, bcrops, C, H, W)
        # for i in range(ncrops):
        #     img_tensor = inputs[0, i, ...]  # img_tensor: (C, H, W)
        #     img = transform_invert(img_tensor, train_transform)
        #     plt.imshow(img)
        #     plt.show()
        #     plt.pause(0.5)
        #     plt.close()
