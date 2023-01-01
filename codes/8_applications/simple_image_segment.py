# -*- coding: utf-8 -*-

'''
@Author   :   Corley Tang
@Contact  :   cutercorleytd@gmail.com
@Github   :   https://github.com/corleytd
@Time     :   2022-12-23 18:01
@Project  :   PyTorchBasic-simple_image_segment
'''

import os
import time

import torch
from PIL import Image
from matplotlib import pyplot as plt
from torchvision import transforms

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 图片数据预处理
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# 1.获取所有图片
image_path = '../../data/segs'
images = os.listdir(image_path)

for image in images:
    # 2.加载图片和模型
    img_path = os.path.join(image_path, image)
    img_rgb = Image.open(img_path).convert('RGB')

    model = torch.hub.load('pytorch/vision:main', 'deeplabv3_resnet101', pretrained=True)
    model.to(device)
    model.eval()

    # 3.预处理图片
    img_tensor = transform(img_rgb)
    img_tensor.unsqueeze_(0)
    img_tensor = img_tensor.to(device)

    # 4.前向传播
    with torch.no_grad():
        start = time.time()
        output = model(img_tensor)['out']
        output = output[0]
        print(f'device: {device}, input shape: {img_tensor.shape}, output shape: {output.shape}'
              f' time used: {time.time() - start:.3f}s')

    preds = output.argmax(0)

    # 5.可视化
    palette = torch.tensor([2 ** 25 - 1, 2 ** 15 - 1, 2 ** 21 - 1])
    colors = torch.as_tensor([i for i in range(21)])[:, None] * palette
    colors = (colors % 255).numpy().astype('uint8')

    # 绘制每种颜色中21个类的语义分割预测
    r = Image.fromarray(preds.byte().cpu().numpy()).resize(img_rgb.size)
    r.putpalette(colors)
    plt.subplot(121).imshow(img_rgb)
    plt.subplot(122).imshow(r)
    plt.show()
