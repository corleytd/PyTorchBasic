# -*- coding: utf-8 -*-

'''
@Author   :   Corley Tang
@Contact  :   cutercorleytd@gmail.com
@Github   :   https://github.com/corleytd
@Time     :   2022-11-03 11:32
@Project  :   PyTorchBasic-custom_transforms
'''

import random

import numpy as np
from PIL import Image


class SaltPepperNoise:
    '''自定义椒盐噪声'''

    def __init__(self, snr, p=0.8):
        '''
        初始化
        :param snr: Signal Noise Rate
        :param p: 增加椒盐噪声的概率
        '''
        self.snr = snr
        self.p = p

    def __call__(self, img):
        '''
        :param img: PIL Image
        :return: PIL Image
        '''
        if random.random() < self.p:
            img_ = np.array(img).copy()
            h, w, c = img_.shape
            signal_rate = self.snr
            noise_rate = 1 - self.snr
            mask = np.random.choice((0, 1, 2), size=(h, w, 1), p=[signal_rate, noise_rate / 2, noise_rate / 2])
            mask = np.repeat(mask, c, axis=2)
            img_[mask == 1] = 255  # 盐噪声
            img_[mask == 2] = 0  # 椒噪声
            return Image.fromarray(img_.astype('uint8')).convert('RGB')
        else:
            return img
