# -*- coding: utf-8 -*-

'''
@Author   :   Corley Tang
@Contact  :   cutercorleytd@gmail.com
@Github   :   https://github.com/corleytd
@Time     :   2022-12-30 10:37
@Project  :   PyTorchBasic-gan_inference
'''

import numpy as np
import torch
from matplotlib import pyplot as plt
from torchvision import transforms
from torchvision.utils import make_grid

from models.dcgan import Generator
from tools.common_tools import set_seed

set_seed(65535)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
ckpt_path = 'output/gan_log/checkpoint_epoch_20.ckpt'

# 超参数
image_size = 64
image_num = 64
nc = 3
nz = 100
ngf = 64
ndf = 64
transform = transforms.Compose([
    transforms.Resize(image_size),
    transforms.CenterCrop(image_size),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))  # (-1, 1)
])
add_noise = False

# 1.生成数据
fixed_noise = torch.randn(image_num, nz, 1, 1, device=device)
if add_noise:
    z_idx = 0
    single_noise = torch.randn(1, nz, 1, 1, device=device)
    for i in range(image_num):
        cur_add_noise = single_noise.clone()
        cur_add_noise = cur_add_noise[0, z_idx, 0, 0] + i * 0.01
        fixed_noise[i, ...] = cur_add_noise

# 2.获取模型、加载参数
generator = Generator(nz, ngf, nc)
checkpoint = torch.load(ckpt_path)
generator.load_state_dict(checkpoint['generator_state_dict'])
generator.to(device)

# 3.推理
with torch.no_grad():
    out_fake = generator(fixed_noise).detach().cpu()
img_grid = make_grid(out_fake, padding=2, normalize=True).numpy()
img_grid = np.transpose(img_grid, (1, 2, 0))
plt.imshow(img_grid)
plt.show()
