# -*- coding: utf-8 -*-

'''
@Author   :   Corley Tang
@Contact  :   cutercorleytd@gmail.com
@Github   :   https://github.com/corleytd
@Time     :   2022-12-27 21:27
@Project  :   PyTorchBasic-fasterrcnn_detection
'''

import os
import random
import time

import torch
from PIL import Image
from matplotlib import pyplot as plt
from torch import optim
from torch.utils.data import DataLoader
from torchvision import models, transforms
from torchvision.transforms import functional as F

from tools.common_tools import set_seed
from tools.datasets import PennFundanDataset

set_seed()


def vis_bbox(image, output, classes, max_vis=40, prob_threshold=0.5):
    fig, ax = plt.subplots(figsize=(12, 12))
    ax.imshow(image, aspect='equal')

    out_boxes = output['boxes'].cpu()
    out_scores = output['scores'].cpu()
    out_labels = output['labels'].cpu()

    num_boxes = out_boxes.shape[0]
    for idx in range(0, min(num_boxes, max_vis)):
        score = out_scores[idx].numpy()
        box = out_boxes[idx].numpy()
        class_name = classes[out_labels[idx]]

        if score < prob_threshold:
            continue

        ax.add_patch(plt.Rectangle((box[0], box[1]), box[2] - box[0], box[3] - box[1], fill=False, edgecolor='red',
                                   linewidth=3.5))
        ax.text(box[0], box[1] - 2, f'{class_name} {score:.3f}', bbox={'facecolor': 'blue', 'alpha': 0.5}, fontsize=14,
                color='white')
    plt.show()


class Compose:
    def __init__(self, transform):
        self.transform = transform

    def __call__(self, image, target):
        for t in self.transform:
            image, target = t(image, target)
        return image, target


class RandomHorizontalFlip:
    def __init__(self, prob):
        self.prob = prob

    def __call__(self, image, target):
        if random.random() < self.prob:
            height, width = image.shape[-2:]
            image = image.flip(-1)
            bbox = target['boxes']
            bbox[:, [0, 2]] = width - bbox[:, [2, 0]]
            target['boxes'] = bbox
        return image, target


class ToTensor:
    def __call__(self, image, target):
        image = F.to_tensor(image)
        return image, target


# 超参数
LR = 1e-3
num_classes = 2
batch_size = 16
start_epoch, max_epoch = 1, 30
vis_num = 20
transform = Compose([ToTensor(), RandomHorizontalFlip(0.5)])
# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
device = torch.device('cpu')
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

# 1.整理数据
train_path = '../../data/PennFudanPed'
train_dataset = PennFundanDataset(data_dir=train_path, transform=transform)
train_loader = DataLoader(train_dataset, batch_size=batch_size, collate_fn=lambda batch: tuple(zip(*batch)))
train_count = len(train_loader)

# 2.定义模型
model = models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
in_features = model.roi_heads.box_predictor.cls_score.in_features
# 用一个新的替换预训练的头
model.roi_heads.box_predictor = models.detection.faster_rcnn.FastRCNNPredictor(in_features, num_classes)
model.to(device)

# 3.定义优化器
params = [param for param in model.parameters() if param.requires_grad]
optimizer = optim.SGD(params, lr=LR, momentum=0.9, weight_decay=5e-4)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)

# 4.迭代训练
for epoch in range(start_epoch, max_epoch + 1):
    for iteration, (images, targets) in enumerate(train_loader, 1):
        images = [image.to(device) for image in images]
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        outputs = model(images, targets)
        loss = sum(outputs.values())

        print(
            f'Train: Epoch: [{epoch:0>3}/{max_epoch:0>3}] Iteration: [{iteration:0>3}/{train_count:0>3}] Loss:{loss:.4f}')

        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

    scheduler.step()

# 5.测试
img_dir = os.path.join(train_path, 'PNGImages')
images = os.listdir(img_dir)
random.shuffle(images)
transform = transforms.Compose([transforms.ToTensor()])

model.eval()
for image in images[:vis_num]:
    img_path = os.path.join(img_dir, image)

    # 图片处理
    image_rgb = Image.open(img_path).convert('RGB')
    img_tensor = transform(image_rgb)
    img_tensor = img_tensor.to(device)

    # 前向传播
    with torch.no_grad():
        start = time.time()
        output = model([img_tensor])[0]
        print(f'input image tensor shape: {img_tensor.shape} time spent: {time.time() - start:.3f}s')

    # 可视化
    vis_bbox(image_rgb, output, COCO_INSTANCE_CATEGORY_NAMES, max_vis=20, prob_threshold=0.5)
