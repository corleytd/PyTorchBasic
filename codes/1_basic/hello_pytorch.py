# -*- coding: utf-8 -*-

'''
@Author   :   Corley Tang
@Contact  :   cutercorleytd@gmail.com
@Github   :   https://github.com/corleytd
@Time     :   2022-10-24 10:46
@Project  :   PyTorchBasic-hello_pytorch
'''

import torch

print(f'Hello world, Hello PyTorch {torch.__version__}')
print(f'CUDA is available: {torch.cuda.is_available()}, version: {torch.version.cuda}')
