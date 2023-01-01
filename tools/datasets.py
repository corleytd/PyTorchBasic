# -*- coding: utf-8 -*-

'''
@Author   :   Corley Tang
@Contact  :   cutercorleytd@gmail.com
@Github   :   https://github.com/corleytd
@Time     :   2022-10-31 22:18
@Project  :   PyTorchBasic-datasets
'''

import os
import random
import re

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

from tools.common_tools import set_seed

set_seed()

rmb_labels = {'1': 0, '100': 1}
cat_dog_labels = {'cat': 0, 'dog': 1}
ant_bee_labels = {'ants': 0, 'bees': 1}


class RMBDataset(Dataset):
    '''人民币分类任务Dataset'''

    def __init__(self, data_dir, transform=None):
        '''
        :param data_dir: 数据集所在路径
        :param transform: 数据预处理
        '''
        self.data_info = self.get_img_info(data_dir)  # 存储所有图片路径和标签，以便于在DataLoader中通过index读取样本
        self.transform = transform

    def __getitem__(self, index):
        img_path, label = self.data_info[index]
        img = Image.open(img_path).convert('RGB')

        if self.transform:
            img = self.transform(img)  # 进行transform，转为张量

        return img, label

    def __len__(self):
        return len(self.data_info)

    @staticmethod
    def get_img_info(data_dir):
        data_info = []
        for sub_dir in os.listdir(data_dir):  # 遍历类别
            imgs = os.listdir(os.path.join(data_dir, sub_dir))
            imgs = list(filter(lambda item: item.endswith('.jpg'), imgs))

            for img in imgs:
                img_path = os.path.join(data_dir, sub_dir, img)
                label = rmb_labels[sub_dir]
                data_info.append((img_path, label))

        return data_info


class CatDogDataset(Dataset):
    '''猫狗分类任务Dataset'''

    def __init__(self, data_dir, transform=None):
        '''
        :param data_dir: 数据集所在路径
        :param transform: 数据预处理
        '''
        self.data_info = self.get_img_info(data_dir)  # 存储所有图片路径和标签，以便于在DataLoader中通过index读取样本
        self.transform = transform

    def __getitem__(self, index):
        img_path, label = self.data_info[index]
        img = Image.open(img_path).convert('RGB')

        if self.transform:
            img = self.transform(img)  # 进行transform，转为张量

        return img, label

    def __len__(self):
        return len(self.data_info)

    @staticmethod
    def get_img_info(data_dir):
        data_info = []
        imgs = os.listdir(data_dir)

        for img in imgs:
            img_path = os.path.join(data_dir, img)
            label = cat_dog_labels[img[:3]]
            data_info.append((img_path, label))

        return data_info


class AntBeeDataset(Dataset):
    def __init__(self, data_dir, transform=None):
        self.data_info = self.get_img_info(data_dir)
        self.transform = transform

    def __getitem__(self, index):
        img_path, label = self.data_info[index]
        img = Image.open(img_path).convert('RGB')

        if self.transform:
            img = self.transform(img)

        return img, label

    def __len__(self):
        return len(self.data_info)

    def get_img_info(self, data_dir):
        data_info = []
        for sub_dir in os.listdir(data_dir):  # 遍历类别
            imgs = os.listdir(os.path.join(data_dir, sub_dir))
            imgs = list(filter(lambda item: item.endswith('.jpg'), imgs))

            for img in imgs:
                img_path = os.path.join(data_dir, sub_dir, img)
                label = ant_bee_labels[sub_dir]
                data_info.append((img_path, label))

        return data_info


class PortraitDataset(Dataset):
    def __init__(self, data_dir, transform=None, in_size=224):
        super().__init__()
        self.data_dir = data_dir
        self.transform = transform
        self.in_size = in_size
        self.label_paths = self._get_image_list()

    def __getitem__(self, index):
        label_path = self.label_paths[index]
        img_path = label_path.replace('_matte', '')

        img_rgb = Image.open(img_path).convert('RGB')
        img_rgb = img_rgb.resize((self.in_size, self.in_size), Image.BILINEAR)
        img_rgb = np.array(img_rgb)  # HWC
        img_rgb = img_rgb.transpose((2, 0, 1))  # CHW

        label_l = Image.open(label_path).convert('L')
        label_l = label_l.resize((self.in_size, self.in_size), Image.NEAREST)
        label_l = np.array(label_l)  # HW
        label_l = label_l[np.newaxis, :, :]  # CHW
        label_l[label_l != 0] = 1

        img_tensor = torch.from_numpy(img_rgb).float()
        label_tensor = torch.from_numpy(label_l).float()

        return img_tensor, label_tensor

    def __len__(self):
        return len(self.label_paths)

    def _get_image_list(self):
        files = os.listdir(self.data_dir)
        label_paths = [os.path.join(self.data_dir, label_name) for label_name in files if
                       label_name.endswith('matte.png')]
        random.shuffle(label_paths)
        return label_paths


class PennFundanDataset(Dataset):
    def __init__(self, data_dir, transform):
        self.data_dir = data_dir
        self.transform = transform
        self.img_dir = os.path.join(data_dir, 'PNGImages')
        self.label_dir = os.path.join(data_dir, 'Annotation')
        self.names = [name[:-4] for name in os.listdir(self.img_dir) if name.endswith('.png')]

    def __getitem__(self, index):
        name = self.names[index]
        img_path = os.path.join(self.img_dir, name + '.png')
        label_path = os.path.join(self.label_dir, name + '.txt')

        # 加载图片
        image = Image.open(img_path).convert('RGB')

        # 加载边框和标签
        with open(label_path) as f:
            points = [re.findall(r'\d+', line) for line in f.readlines() if 'Xmin' in line]
            boxes = [[int(p) for p in point][-4:] for point in points]
        boxes = torch.tensor(boxes, dtype=torch.float)
        labels = torch.ones(boxes.shape[0], dtype=torch.long)

        target = {'boxes': boxes, 'labels': labels}
        if self.transform:
            image, target = self.transform(image, target)

        return image, target

    def __len__(self):
        return len(self.names)


class CelebADataset(Dataset):
    def __init__(self, data_dir, transform):
        self.data_dir = data_dir
        self.transform = transform
        self.img_names = os.listdir(self.data_dir)

    def __getitem__(self, index):
        img_path = os.path.join(self.data_dir, self.img_names[index])
        img = Image.open(img_path).convert('RGB')

        if self.transform:
            img = self.transform(img)

        return img

    def __len__(self):
        return len(self.img_names)
