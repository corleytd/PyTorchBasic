# -*- coding: utf-8 -*-

'''
@Author   :   Corley Tang
@Contact  :   cutercorleytd@gmail.com
@Github   :   https://github.com/corleytd
@Time     :   2022-12-06 12:15
@Project  :   PyTorchBasic-tensorboard_methods
'''
import os

import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image
from torch import nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchsummary import summary
from torchvision.models import alexnet
from torchvision.transforms import transforms
from torchvision.utils import make_grid

from models.lenet import LeNet
from tools.common_tools import set_seed
from tools.datasets import RMBDataset

set_seed()

flag = 9

# 1.path
if flag == 0:
    log_dir = 'train_log'
    # writer = SummaryWriter(log_dir=log_dir, comment='tensorboard_path',filename_suffix='.bin')
    writer = SummaryWriter(comment='tensorboard_path', filename_suffix='.bin')

    for x in range(100):
        writer.add_scalar('y=pow_2_x', 2 ** x, x)

    writer.close()

# 2.scalar
if flag == 1:
    writer = SummaryWriter(comment='tensorboard_scalar')

    for x in range(100):
        writer.add_scalar('y=3x', 3 * x, x)
        writer.add_scalar('y=pow_3_x', 3 ** x, x)
        writer.add_scalars('sincos', {
            'xsinx': x * np.sin(x),
            'xcosx': x * np.cos(x)
        }, x)

    writer.close()

# 3.histogram
if flag == 2:
    writer = SummaryWriter(comment='tensorboard_histogram')

    for x in range(3):
        np.random.seed(x ** 2 + 1)

        data_union = np.arange(100)
        data_norm = np.random.normal(size=1000)
        writer.add_histogram('distribution_union', data_union, x)
        writer.add_histogram('distribution_norm', data_norm, x)

        plt.subplot(121).hist(data_union, label='union')
        plt.subplot(122).hist(data_norm, label='normal')

        plt.legend()
        plt.show()

    writer.close()

# 4.add_image
if flag == 3:
    writer = SummaryWriter(comment='tensorboard_image')

    # img1: random
    fake_img = torch.randn(3, 512, 512)
    writer.add_image('img', fake_img, 1)

    # img2: white
    fake_img = torch.ones(3, 512, 512)
    writer.add_image('img', fake_img, 2)

    # img3: black
    fake_img = torch.ones(3, 512, 512) * 1.1
    writer.add_image('img', fake_img, 3)

    # img4: HW
    fake_img = torch.rand(512, 512)
    writer.add_image('img', fake_img, 4, dataformats='HW')

    # img5: HWC
    fake_img = torch.rand(512, 512, 3)
    writer.add_image('img', fake_img, 5, dataformats='HWC')

    writer.close()

# 5.make_grid
if flag == 4:
    writer = SummaryWriter(comment='tensorboard_make_grid')

    split_path = '../../data/RMB_split'
    train_path = os.path.join(split_path, 'train')

    transform = transforms.Compose([
        transforms.Resize((32, 64)),
        transforms.ToTensor()
    ])
    train_data = RMBDataset(train_path, transform)
    train_loader = DataLoader(train_data, batch_size=16, shuffle=True)
    data_batch, _ = next(iter(train_loader))

    img_grid = make_grid(data_batch, nrow=4, normalize=True, scale_each=True)
    writer.add_image('grid', img_grid, 1)
    img_grid = make_grid(data_batch, nrow=4, normalize=True, scale_each=False)
    writer.add_image('grid', img_grid, 2)
    img_grid = make_grid(data_batch, nrow=4, normalize=False, scale_each=True)
    writer.add_image('grid', img_grid, 3)
    img_grid = make_grid(data_batch, nrow=4, normalize=False, scale_each=False)
    writer.add_image('grid', img_grid, 4)

    writer.close()

# 6.kernel visualization
if flag == 5:
    writer = SummaryWriter(comment='kernel_visual')

    alexnet_model = alexnet(pretrained=True)

    kernel_num = -1
    visual_limit = 1

    for module in alexnet_model.modules():
        if isinstance(module, nn.Conv2d):
            kernel_num += 1
            if kernel_num > visual_limit:
                break
            kernels = module.weight
            c_out, c_in, k_w, k_h = tuple(kernels.shape)

            for idx in range(c_out):
                kernel = kernels[idx, :, :, :].unsqueeze(1)
                kernel_grid = make_grid(kernel, normalize=True, scale_each=True, nrow=c_in)
                writer.add_image(f'{kernel_num}_Conv_layer_split_in_channel', kernel_grid, idx)

            kernel_all = kernels.view(-1, 3, k_h, k_w)
            kernel_grid = make_grid(kernel_all, normalize=True, scale_each=True, nrow=8)
            writer.add_image(f'{kernel_num}_all', kernel_grid, c_out)

            print(f'{kernel_num}_conv_layer shape: {kernels.shape}')

    writer.close()

# 7.feature map visualization
if flag == 6:
    writer = SummaryWriter(comment='feature_map_visual')

    img_path = '../../data/imgs/greatwall.jpeg'
    norm_mean = [0.49139968, 0.48215827, 0.44653124]
    norm_std = [0.24703233, 0.24348505, 0.26158768]

    img_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(norm_mean, norm_std)
    ])

    pil_img = Image.open(img_path).convert('RGB')
    img_tensor = img_transform(pil_img)
    img_tensor.unsqueeze_(0)  # CHW -> BCHW

    # 定义模型
    alexnet_model = alexnet(pretrained=True)

    # 前向传播
    conv_layer1 = alexnet_model.features[0]
    fmap = conv_layer1(img_tensor)

    # 格式转换
    fmap.transpose_(0, 1)  # BCHW
    fmap_grid = make_grid(fmap, normalize=True, scale_each=True, nrow=8)

    writer.add_image('fmap', fmap_grid, global_step=0)

    writer.close()

# 8.feature map visualization with hook
if flag == 7:
    writer = SummaryWriter(comment='feature_map_visual_with_hook')

    img_path = '../../data/imgs/greatwall.jpeg'
    norm_mean = [0.49139968, 0.48215827, 0.44653124]
    norm_std = [0.24703233, 0.24348505, 0.26158768]

    img_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(norm_mean, norm_std)
    ])

    pil_img = Image.open(img_path).convert('RGB')
    img_tensor = img_transform(pil_img)
    img_tensor.unsqueeze_(0)  # CHW -> BCHW

    # 定义模型
    alexnet_model = alexnet(pretrained=True)

    # 定义勾子函数
    fmaps = {}


    def forward_hook(m, i, o):
        fmaps[str(m.weight.shape)].append(o)


    # 注册hook
    for name, module in alexnet_model.named_modules():
        if isinstance(module, nn.Conv2d):
            key = str(module.weight.shape)
            fmaps.setdefault(key, [])

            n1, n2 = name.split('.')
            alexnet_model._modules[n1]._modules[n2].register_forward_hook(forward_hook)

    # 前向传播
    output = alexnet_model(img_tensor)

    # add_image
    for name, fmap in fmaps.items():
        fmap = fmap[0]
        fmap.transpose_(0, 1)

        fmap_grid = make_grid(fmap, normalize=True, scale_each=True, nrow=int(fmap.shape[0] ** 0.5))
        writer.add_image(f'fmap_with_hook_in_{name}', fmap_grid, global_step=1)

    writer.close()

# 9.add_graph
if flag == 8:
    writer = SummaryWriter(comment='add_graph')

    # 数据
    fake_img = torch.randn(1, 3, 32, 32)

    # 模型
    lenet = LeNet(classes=2)

    writer.add_graph(lenet, fake_img)
    writer.close()

# 10.torchsummary
if flag == 9:
    lenet = LeNet(classes=2)
    summary(lenet, (3, 32, 32), device='cpu')
