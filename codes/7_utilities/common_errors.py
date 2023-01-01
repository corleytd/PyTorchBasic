# -*- coding: utf-8 -*-

'''
@Author   :   Corley Tang
@Contact  :   cutercorleytd@gmail.com
@Github   :   https://github.com/corleytd
@Time     :   2022-12-21 19:02
@Project  :   PyTorchBasic-common_errors
'''

import random

import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import transforms

from models.lenet import LeNet
from tools.datasets import RMBDataset

flag = 7

# 1.ValueError: num_samples=0
if flag == 0:
    # train_dir = '../../data/RMB_split'  # ValueError: num_samples should be a positive integer value, but got num_samples=0
    train_dir = '../../data/RMB_split/train'

    train_data = RMBDataset(train_dir)
    train_loader = DataLoader(dataset=train_data, batch_size=16, shuffle=True)

# 2.TypeError
if flag == 1:
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.FiveCrop(200),
        transforms.Lambda(lambda crops: torch.stack([transforms.ToTensor()(crop) for crop in crops])),
        # transforms.ToTensor()  # TypeError: pic should be PIL Image or ndarray. Got <class 'torch.Tensor'>
    ])
    train_dir = '../../data/RMB_split/train'
    train_data = RMBDataset(train_dir, transform=transform)
    loader = DataLoader(dataset=train_data, batch_size=16, shuffle=True)
    data, label = next(iter(loader))

# 3.RuntimeError
if flag == 2:
    class FooDataset(Dataset):
        def __init__(self, num_data, data_dir=None, transform=None):
            self.num_data = num_data
            self.data_dir = data_dir
            self.transform = transform

        def __getitem__(self, item):
            # size = random.randint(60, 64)  # RuntimeError: stack expects each tensor to be equal size, but got [3, 60, 60] at entry 0 and [3, 62, 62] at entry 2
            size = random.randint(64, 64)
            fake_data = torch.randn(3, size, size)
            fake_label = torch.randint(0, 10, size=(1,))

            return fake_data, fake_label

        def __len__(self):
            return self.num_data


    dataset = FooDataset(10)
    data_loader = DataLoader(dataset=dataset, batch_size=4)

    data, labels = next(iter(data_loader))

# 4.RuntimeError
if flag == 3:
    class FooDataset(Dataset):
        def __init__(self, num_data, shape, data_dir=None, transform=None):
            self.num_data = num_data
            self.shape = shape
            self.data_dir = data_dir
            self.transform = transform

        def __getitem__(self, item):
            fake_data = torch.randn(self.shape)
            fake_label = torch.randint(0, 10, size=(1,))

            if self.transform:
                fake_data = self.transform(fake_data)

            return fake_data, fake_label

        def __len__(self):
            return self.num_data


    # 1.构造数据
    # 1, 32: RuntimeError: Given groups=1, weight of size [6, 3, 5, 5], expected input[16, 1, 32, 32] to have 3 channels, but got 1 channels instead
    # 3, 36: RuntimeError: mat1 and mat2 shapes cannot be multiplied (16x576 and 400x120)
    channel, img_size = 3, 32
    train_data = FooDataset(32, (channel, img_size, img_size))
    data_loader = DataLoader(train_data, batch_size=16, shuffle=True)

    # 2.定义模型
    model = LeNet(classes=2)

    # 3.前向传播
    batch_data, batch_labels = next(iter(data_loader))
    output = model(batch_data)

# 5.AttributeError
if flag == 4:
    class FooNet(nn.Module):
        def __init__(self):
            super().__init__()
            self.linear = nn.Linear(3, 3, bias=False)
            self.conv = nn.Conv2d(3, 6, 5)
            self.pool = nn.MaxPool2d(5)

        def forward(self, x):
            return self.linear(x)


    model = FooNet()
    torch.save(model, 'output/foonet.pkl')
    for name, layer in model.named_modules():
        print(name)

    model = nn.DataParallel(model)
    for name, layer in model.named_modules():
        print(name)

    # print(model.linear)  # AttributeError: 'DataParallel' object has no attribute 'linear'
    print(model.module.linear)

# 6.AttributeError
if flag == 5:
    # model = torch.load('output/foonet.pkl')  # AttributeError: Can't get attribute 'FooNet' on <module '__main__'

    class FooNet(nn.Module):
        def __init__(self):
            super().__init__()
            self.linear = nn.Linear(3, 3, bias=False)
            self.conv = nn.Conv2d(3, 6, 5)
            self.pool = nn.MaxPool2d(5)

        def forward(self, x):
            return self.linear(x)


    model = torch.load('output/foonet.pkl')
    print(model)

# 7.IndexError
if flag == 6:
    inputs = torch.tensor([[1., 2], [1, 3], [1, 4]])
    # target = torch.tensor([0, 1, 2])  # IndexError: Target 2 is out of bounds.
    target = torch.tensor([0, 1, 1])

    criterion = nn.CrossEntropyLoss()

    loss = criterion(inputs, target)

# 8.
if flag == 7:
    a = torch.tensor([1])
    b = torch.tensor([2], device='cuda')
    # y = a + b  # RuntimeError: Expected all tensors to be on the same device, but found at least two devices, cuda:0 and cpu!

    a.to('cuda')
    # y = a + b  # RuntimeError: Expected all tensors to be on the same device, but found at least two devices, cuda:0 and cpu!

    a = a.to('cuda')
    y = a + b
