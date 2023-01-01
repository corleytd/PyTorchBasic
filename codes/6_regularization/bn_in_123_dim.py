# -*- coding: utf-8 -*-

'''
@Author   :   Corley Tang
@Contact  :   cutercorleytd@gmail.com
@Github   :   https://github.com/corleytd
@Time     :   2022-12-14 21:37
@Project  :   PyTorchBasic-bn_in_123_dim
'''

import torch
from torch import nn

flag = 1

# 1.BatchNorm1d
if flag == 0:
    batch_size = 3
    num_features = 5
    momentum = 0.3
    features_shape = 1

    feature_map = torch.ones(features_shape)  # 1D
    feature_maps = torch.stack([feature_map * i for i in range(1, num_features + 1)], dim=0)  # 2D
    feature_maps_bs = torch.stack([feature_maps for _ in range(batch_size)], dim=0)  # 3D
    print(f'input data:\n{feature_maps_bs}\nshape: {feature_maps_bs.shape}')

    bn = nn.BatchNorm1d(num_features=num_features, momentum=momentum)

    running_mean, running_var = 0, 1

    for i in range(2):
        outputs = bn(feature_maps_bs)
        print(f'\niteration: {i}, running_mean: {bn.running_mean}, running_var: {bn.running_var}')

        mean_t, var_t = 2, 0

        running_mean = (1 - momentum) * running_mean + momentum * mean_t
        running_var = (1 - momentum) * running_var + momentum * var_t

        print(f'iteration: {i}, second running_mean by hand: {running_mean}, second running_var by hand: {running_var}')

# 2.BatchNorm2d
if flag == 1:
    batch_size = 3
    num_features = 6
    momentum = 0.3
    features_shape = (2, 2)

    feature_map = torch.ones(features_shape)  # 2D
    feature_maps = torch.stack([feature_map * i for i in range(1, num_features + 1)], dim=0)  # 3D
    feature_maps_bs = torch.stack([feature_maps for _ in range(batch_size)], dim=0)  # 4D
    print(f'input data:\n{feature_maps_bs}\nshape: {feature_maps_bs.shape}')

    bn = nn.BatchNorm2d(num_features=num_features, momentum=momentum)

    running_mean, running_var = 0, 1

    for i in range(2):
        outputs = bn(feature_maps_bs)

        print(
            f'\niteration: {i}, running_mean.shape: {bn.running_mean.shape}, running_var.shape: {bn.running_var.shape}')

        print(f'iteration: {i}, weight.shape: {bn.weight.shape}, bias.shape: {bn.bias.shape}')

# 3.BatchNorm3d
if flag == 1:
    batch_size = 3
    num_features = 4
    momentum = 0.3
    features_shape = (2, 2, 3)

    feature_map = torch.ones(features_shape)  # 3D
    feature_maps = torch.stack([feature_map * i for i in range(1, num_features + 1)], dim=0)  # 4D
    feature_maps_bs = torch.stack([feature_maps for _ in range(batch_size)], dim=0)  # 5D
    print(f'input data:\n{feature_maps_bs}\nshape: {feature_maps_bs.shape}')

    bn = nn.BatchNorm3d(num_features=num_features, momentum=momentum)

    running_mean, running_var = 0, 1

    for i in range(2):
        outputs = bn(feature_maps_bs)

        print(
            f'\niteration: {i}, running_mean.shape: {bn.running_mean.shape}, running_var.shape: {bn.running_var.shape}')

        print(f'iteration: {i}, weight.shape: {bn.weight.shape}, bias.shape: {bn.bias.shape}')
