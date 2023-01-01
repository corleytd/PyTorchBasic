# -*- coding: utf-8 -*-

'''
@Author   :   Corley Tang
@Contact  :   cutercorleytd@gmail.com
@Github   :   https://github.com/corleytd
@Time     :   2022-11-01 11:49
@Project  :   PyTorchBasic-common_tools
'''

import os
import platform
import random

import numpy as np
import torch
from PIL import Image
from torch.backends import cudnn
from torchvision import transforms


def set_seed(seed=42):
    '''设置随机种子'''
    random.seed(seed)  # 保证后续使用random函数时，产生固定的随机数
    np.random.seed(seed)  # 保证后续numpy使用random函数时，产生固定的随机数
    torch.manual_seed(seed)  # 固定CPU随机种子
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)  # 固定当前GPU随机种子
        torch.cuda.manual_seed_all(seed)  # 固定所有GPU随机种子
    torch.backends.cudnn.benchmark = False  # 是否固定网络结构的模型优化，自动寻找最适合当前配置的高效算法，用于提高模型的效率
    torch.backends.cudnn.deterministic = True  # 确保每次返回的卷积算法是确定的，固定网络结构，以保证算法的结果可复现


def transform_invert(img_, transform_methods):
    '''
    将transform后的图片数据恢复成原来的格式，即反transform
    :param img_: Tensor
    :param transform_methord: torchvision.transforms
    :return: PIL Image
    '''
    if 'Normalize' in str(transform_methods):
        norm_transform = [trans for trans in transform_methods.transforms if isinstance(trans, transforms.Normalize)]
        mean = torch.tensor(norm_transform[0].mean, dtype=img_.dtype, device=img_.device)
        std = torch.tensor(norm_transform[0].std, dtype=img_.dtype, device=img_.device)
        img_.mul_(std[:, None, None]).add_(mean[:, None, None])

    img_ = img_.transpose(0, 2).transpose(0, 1)  # C*H*W -> H*W*C
    if 'ToTensor' in str(transform_methods) or img_.max() < 1:
        img_ = img_.detach().numpy() * 255

    if img_.shape[-1] == 3:
        img_ = Image.fromarray(img_.astype('uint8')).convert('RGB')
    elif img_.shape[-1] == 1:
        img_ = Image.fromarray(img_.astype('uint8').squeeze())
    else:
        raise Exception('Invalid img shape, expected 1 or 3 in axis 2, but got {}!'.format(img_.shape[-1]))

    return img_


def get_gpu_memory():
    if platform.system() == 'Windows':
        return []
    else:
        output = os.popen('nvidia-smi -q -d Memory | grep -A5 GPU | grep Free')
        gpu_memory = [int(x.split()[2]) for x in output.readlines()]
        return gpu_memory
