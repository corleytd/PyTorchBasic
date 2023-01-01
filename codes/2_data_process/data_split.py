# -*- coding: utf-8 -*-

'''
@Author   :   Corley Tang
@Contact  :   cutercorleytd@gmail.com
@Github   :   https://github.com/corleytd
@Time     :   2022-10-31 18:27
@Project  :   PyTorchBasic-data_split
'''

import os
import random
import shutil

from tqdm import tqdm

random.seed(65537)


def make_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)


def split_rmb_data():
    data_path = '../../data/RMB_data'
    split_path = '../../data/RMB_split'
    train_path = os.path.join(split_path, 'train')
    valid_path = os.path.join(split_path, 'valid')
    test_path = os.path.join(split_path, 'test')

    if not os.path.exists(data_path):
        raise FileNotFoundError('数据不存在')

    train_pct, valid_pct, test_pct = 0.8, 0.1, 0.1

    for sub_path in os.listdir(data_path):
        imgs = os.listdir(os.path.join(data_path, sub_path))
        imgs = list(filter(lambda item: item.endswith('.jpg'), imgs))
        random.shuffle(imgs)
        img_count = len(imgs)
        if img_count == 0:
            continue

        train_count = int(img_count * train_pct)
        valid_count = int(img_count * valid_pct)
        test_count = int(img_count * test_pct)

        for i in tqdm(range(img_count)):
            if i < train_count:
                out_dir = os.path.join(train_path, sub_path)
            elif i < train_count + valid_count:
                out_dir = os.path.join(valid_path, sub_path)
            else:
                out_dir = os.path.join(test_path, sub_path)

            make_dir(out_dir)
            tgt_path = os.path.join(out_dir, imgs[i])
            src_path = os.path.join(data_path, sub_path, imgs[i])

            shutil.copy(src_path, tgt_path)

        print(f'Class: {sub_path}, train: {train_count}, valid: {valid_count}, test: {test_count}')

    print(f'已在{split_path}中创建划分好的数据！')


def split_cat_dog_data():
    # 划分数据
    data_path = '../../data/Cat_Dog_data'
    split_path = '../../data/Cat_Dog_split'
    train_path = os.path.join(split_path, 'train')
    valid_path = os.path.join(split_path, 'valid')
    test_path = os.path.join(split_path, 'test')

    if not os.path.exists(data_path):
        raise FileNotFoundError('数据不存在')

    train_pct, valid_pct, test_pct = 0.8, 0.1, 0.1

    imgs = os.listdir(data_path)
    imgs = list(filter(lambda item: item.endswith('.jpg') and ('cat' in item or 'dog' in item), imgs))
    random.shuffle(imgs)
    img_count = len(imgs)
    if img_count == 0:
        return

    train_count = int(img_count * train_pct)
    valid_count = int(img_count * valid_pct)
    test_count = int(img_count * test_pct)
    cat_count, dog_count = 0, 0

    for idx, img in tqdm(enumerate(imgs)):
        if idx < train_count:
            out_dir = train_path
        elif idx < train_count + valid_count:
            out_dir = valid_path
        else:
            out_dir = test_path

        make_dir(out_dir)
        if 'cat' in img:
            cat_count += 1
        elif 'dog' in img:
            dog_count += 1

        tgt_path = os.path.join(out_dir, img)
        src_path = os.path.join(data_path, img)

        shutil.copy(src_path, tgt_path)

    print(
        f'Class: cat {cat_count}, Class: dog {dog_count}, train: {train_count}, valid: {valid_count}, test: {test_count}')

    print(f'已在{split_path}中创建划分好的数据！')


if __name__ == '__main__':
    # split_rmb_data()
    split_cat_dog_data()
