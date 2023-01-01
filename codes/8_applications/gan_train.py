# -*- coding: utf-8 -*-

'''
@Author   :   Corley Tang
@Contact  :   cutercorleytd@gmail.com
@Github   :   https://github.com/corleytd
@Time     :   2022-12-28 19:56
@Project  :   PyTorchBasic-gan_train
'''

import os

import imageio.v2 as imageio
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.utils import make_grid

from models.dcgan import Generator, Discriminator
from tools.common_tools import set_seed
from tools.datasets import CelebADataset

set_seed()

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# data_path = '../../data/img_align_celeba_2k'  # 2k
data_path = '../../data/img_align_celeba'  # 200k
out_path = 'output/gan_log'
if not os.path.exists(out_path):
    os.makedirs(out_path)

# 超参数
ckpt_interval = 10
log_interval = 10
image_size = 64
nc = 3
nz = 100
ngf = 128
ndf = 128
num_epochs = 20
fixed_noise = torch.randn(64, nz, 1, 1, device=device)
real_idx, fake_idx = 0.9, 0.1
lr = 2e-4
batch_size = 512
beta1 = 0.5
transform = transforms.Compose([
    transforms.Resize(image_size),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))  # (-1, 1)
])

# 1.获取数据
train_set = CelebADataset(data_path, transform=transform)
train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
train_count = len(train_loader)

# 展示图片
img_batch = next(iter(train_loader))
plt.title('Train Images')
plt.imshow(np.transpose(make_grid(img_batch.to(device)[:64], padding=2, normalize=True).cpu(), (1, 2, 0)))
plt.show()

# 2.定义模型
generator = Generator(nz=nz, ngf=ngf, nc=nc)
generator.initialize_weights()
generator.to(device)
discriminator = Discriminator(nc=nc, ndf=ndf)
discriminator.initialize_weights()
discriminator.to(device)

# 3.定义损失函数
criterion = nn.BCELoss()

# 4.定义优化器
optimizer_gen = optim.Adam(generator.parameters(), lr=lr, betas=(beta1, 0.999))
optimizer_dis = optim.Adam(discriminator.parameters(), lr=lr, betas=(beta1, 0.999))
scheduler_gen = optim.lr_scheduler.StepLR(optimizer_gen, step_size=8, gamma=0.1)
scheduler_dis = optim.lr_scheduler.StepLR(optimizer_dis, step_size=8, gamma=0.1)

# 5.迭代训练
losses_gen, losses_dis = [], []
for epoch in range(1, num_epochs + 1):
    for iteration, images in enumerate(train_loader, 1):
        # 5.1 训练判别器
        cur_batch_size = images.size(0)
        real_img = images.to(device)
        real_label = torch.full((cur_batch_size,), real_idx, dtype=torch.float, device=device)

        noise = torch.randn(cur_batch_size, nz, 1, 1, device=device)
        fake_img = generator(noise)
        fake_label = torch.full((cur_batch_size,), fake_idx, dtype=torch.float, device=device)

        # 使用真实图片训练判别器
        out_dis_real = discriminator(real_img)
        loss_dis_real = criterion(out_dis_real.view(-1), real_label)

        # 使用虚假图片训练判别器
        out_dis_fake = discriminator(fake_img.detach())
        loss_dis_fake = criterion(out_dis_fake.view(-1), fake_label)

        # 反向传播
        loss_dis_real.backward()
        loss_dis_fake.backward()
        loss_dis = loss_dis_real + loss_dis_fake

        # 更新参数
        optimizer_dis.step()

        # 记录概率
        d_x = out_dis_real.mean().item()  # D(x)
        d_g_z1 = out_dis_fake.mean().item()  # D(G(z1))

        # 清空梯度
        discriminator.zero_grad()

        # 5.2 训练生成器
        label_for_train_gen = real_label
        out_dis_fake_2 = discriminator(fake_img)

        loss_gen = criterion(out_dis_fake_2.view(-1), label_for_train_gen)
        # 反向传播
        loss_gen.backward()

        # 更新参数
        optimizer_gen.step()

        # 记录概率
        d_g_z2 = out_dis_fake_2.mean().item()  # D(G(z2))

        # 清空梯度
        generator.zero_grad()

        if iteration % log_interval == 0:
            print(
                f'Epoch: [{epoch:0>3}/{num_epochs:0>3}] Iteration: [{iteration:0>3}/{train_count:0>3}] Loss_Dis: {loss_dis.item():.4f} Loss_Gen: {loss_gen.item():.4f} D(x): {d_x:.4f} D(G(z)): {d_g_z1:.4f} / {d_g_z2:.4f}')

        # 保存损失
        losses_gen.append(loss_gen.item())
        losses_dis.append(loss_dis.item())

    scheduler_dis.step()
    scheduler_gen.step()

    # 通过将G的输出保存在固定噪声上，检查生成器的运行情况
    with torch.no_grad():
        out_fake = generator(fixed_noise).detach().cpu()
    img_grid = make_grid(out_fake, padding=2, normalize=True).numpy()
    img_grid = np.transpose(img_grid, (1, 2, 0))
    plt.imshow(img_grid)
    plt.title(f'Epoch: {epoch}')
    plt.savefig(os.path.join(out_path, f'epoch_{epoch}.png'))
    plt.show()

    # 保存检查点
    if epoch % ckpt_interval == 0:
        checkpoint = {
            'generator_state_dict': generator.state_dict(),
            'discriminator_state_dict': discriminator.state_dict(),
            'epoch': epoch
        }
        ckpt_path = os.path.join(out_path, f'checkpoint_epoch_{epoch}.ckpt')
        torch.save(checkpoint, ckpt_path)

# 损失画图
plt.figure(figsize=(10, 5))
plt.title('Generator and Discriminator Loss During Training')
plt.plot(losses_gen, label='Generator')
plt.plot(losses_dis, label='Discriminator')
plt.xlabel('Iteration')
plt.ylabel('Loss')
plt.legend()
plt.savefig(os.path.join(out_path, 'loss.png'))
plt.show()

# 保存动图
epochs = [int(name[6:-4]) for name in os.listdir(out_path) if 'epoch' in name and 'png' in name]
epochs.sort()

imgs_gif = [imageio.imread(os.path.join(out_path, f'epoch_{epoch}.png')) for epoch in epochs]
imageio.mimsave(os.path.join(out_path, 'generation_animation.gif'), imgs_gif, fps=2)
print('Done!')
