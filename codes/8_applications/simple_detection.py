# -*- coding: utf-8 -*-

'''
@Author   :   Corley Tang
@Contact  :   cutercorleytd@gmail.com
@Github   :   https://github.com/corleytd
@Time     :   2022-12-27 12:29
@Project  :   PyTorchBasic-simple_detection
'''

import os
import time

import torch
from PIL import Image
from matplotlib import pyplot as plt
from torchvision import transforms, models

device = torch.device('cpu')
max_vis = 40
threshold = 0.5
# classes_coco
COCO_INSTANCE_CATEGORY_NAMES = [
    '__background__', 'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat',
    'traffic light', 'fire hydrant', 'N/A', 'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse',
    'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'N/A', 'backpack', 'umbrella', 'N/A', 'N/A', 'handbag',
    'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove',
    'skateboard', 'surfboard', 'tennis racket', 'bottle', 'N/A', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl',
    'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair',
    'couch', 'potted plant', 'bed', 'N/A', 'dining table', 'N/A', 'N/A', 'toilet', 'N/A', 'tv', 'laptop', 'mouse',
    'remote', 'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'N/A', 'book', 'clock',
    'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush'
]

# 1.获取数据
image_path = '../../data/dets'
images = os.listdir(image_path)
transform = transforms.Compose([
    transforms.ToTensor()
])

# 2.定义模型
model = models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
model.to(device)
model.eval()

# 3.预处理数据
img_list = [os.path.join(image_path, image) for image in images]
img_rgbs = [Image.open(image).convert('RGB') for image in img_list]
img_tensors = [transform(image).to(device) for image in img_rgbs]

with torch.no_grad():
    start = time.time()
    # 4.前向传播
    print(f'input image tensor shape: {img_tensors[0].shape}')
    outputs = model(img_tensors)
    print(f'time spent: {time.time() - start:.3f}s')

    for img_rgb, output in zip(img_rgbs, outputs):
        for k, v in output.items():
            print(f'{k}: {v.shape}')

        # 5.可视化
        out_boxes = output['boxes'].cpu()
        out_scores = output['scores'].cpu()
        out_labels = output['labels'].cpu()

        fig, ax = plt.subplots(figsize=(12, 12))
        ax.imshow(img_rgb, aspect='equal')

        num_boxes = out_boxes.shape[0]
        for idx in range(0, min(num_boxes, max_vis)):
            score = out_scores[idx].numpy()
            box = out_boxes[idx].numpy()
            class_name = COCO_INSTANCE_CATEGORY_NAMES[out_labels[idx]]

            if score < threshold:
                continue

            ax.add_patch(plt.Rectangle((box[0], box[1]), box[2] - box[0], box[3] - box[1], fill=False, edgecolor='red',
                                       linewidth=3.5))
            ax.text(box[0], box[1] - 2, f'{class_name} {score:.3f}', bbox={'facecolor': 'blue', 'alpha': 0.5},
                    fontsize=14, color='white')
        plt.show()
