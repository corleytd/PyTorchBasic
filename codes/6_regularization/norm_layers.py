# -*- coding: utf-8 -*-

'''
@Author   :   Corley Tang
@Contact  :   cutercorleytd@gmail.com
@Github   :   https://github.com/corleytd
@Time     :   2022-12-15 12:03
@Project  :   PyTorchBasic-norm_layers
https://blog.csdn.net/qq_43426908/article/details/123119919
'''

import torch
from torch import nn

flag = 4

# 1.LayerNorm
if flag == 0:
    batch_size = 2
    num_features = 3
    features_shape = (2, 2)

    feature_map = torch.ones(features_shape)  # 2D
    feature_maps = torch.stack([feature_map * i for i in range(1, num_features + 1)], dim=0)  # 3D
    feature_maps_bs = torch.stack([feature_maps for _ in range(batch_size)], dim=0)  # 4D B*C*H*W

    ln = nn.LayerNorm(feature_maps_bs.size()[1:], elementwise_affine=True)

    output = ln(feature_maps_bs)
    print(ln.weight.shape)
    print(feature_maps_bs[0, ...])
    print(output[0, ...])

# 2.LayerNorm: elementwise_affine False
if flag == 1:
    batch_size = 2
    num_features = 3
    features_shape = (2, 2)

    feature_map = torch.ones(features_shape)  # 2D
    feature_maps = torch.stack([feature_map * i for i in range(1, num_features + 1)], dim=0)  # 3D
    feature_maps_bs = torch.stack([feature_maps for _ in range(batch_size)], dim=0)  # 4D B*C*H*W

    ln = nn.LayerNorm(feature_maps_bs.size()[1:], elementwise_affine=False)

    output = ln(feature_maps_bs)
    print(feature_maps_bs[0, ...])
    print(output[0, ...])

# 3.LayerNorm: input shape
if flag == 2:
    batch_size = 8
    num_features = 6
    features_shape = (3, 4)

    feature_map = torch.ones(features_shape)  # 2D
    feature_maps = torch.stack([feature_map * i for i in range(1, num_features + 1)], dim=0)  # 3D
    feature_maps_bs = torch.stack([feature_maps for _ in range(batch_size)], dim=0)  # 4D B*C*H*W

    # ln = nn.LayerNorm([4])
    # ln = nn.LayerNorm([3, 4])
    # ln = nn.LayerNorm([6, 4]) # RuntimeError
    ln = nn.LayerNorm([6, 3, 4])

    output = ln(feature_maps_bs)
    print(ln.weight.shape)
    print(feature_maps_bs[0, ...])
    print(output[0, ...])

# 4.InstanceNorm2d
if flag == 3:
    batch_size = 3
    num_features = 4
    momentum = 0.3
    features_shape = (2, 2)

    feature_map = torch.ones(features_shape)  # 2D
    feature_maps = torch.stack([feature_map * i for i in range(1, num_features + 1)], dim=0)  # 3D
    feature_maps_bs = torch.stack([feature_maps for _ in range(batch_size)], dim=0)  # 4D
    print(f'input data:\n{feature_maps_bs}\nshape: {feature_maps_bs.shape}')

    instance_norm = nn.InstanceNorm2d(num_features=num_features, momentum=momentum)

    outputs = instance_norm(feature_maps_bs)
    print(outputs)

# 5.GroupNorm
if flag == 4:
    batch_size = 3
    num_features = 4
    num_groups = 2
    features_shape = (2, 2)

    feature_map = torch.ones(features_shape)  # 2D
    feature_maps = torch.stack([feature_map * i for i in range(1, num_features + 1)], dim=0)  # 3D
    feature_maps_bs = torch.stack([feature_maps for _ in range(batch_size)], dim=0)  # 4D
    print(f'input data:\n{feature_maps_bs}\nshape: {feature_maps_bs.shape}')

    gn = nn.GroupNorm(num_groups, num_features)
    outputs = gn(feature_maps_bs)

    print(gn.weight.shape)
    print(outputs[0])
