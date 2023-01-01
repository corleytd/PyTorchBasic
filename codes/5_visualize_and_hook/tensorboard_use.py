# -*- coding: utf-8 -*-

'''
@Author   :   Corley Tang
@Contact  :   cutercorleytd@gmail.com
@Github   :   https://github.com/corleytd
@Time     :   2022-12-06 10:18
@Project  :   PyTorchBasic-tensorboard_use
'''

import numpy as np
from torch.utils.tensorboard import SummaryWriter

writer = SummaryWriter(comment='tensorboard_first_use')

for x in range(100):
    writer.add_scalar('y=2x', 2 * x, x)
    writer.add_scalar('y=pow(2, x)', 2 ** x, x)

    writer.add_scalars('data/scalar_group', {
        'xsinx': x * np.sin(x),
        'xcosx': x * np.cos(x),
        'arctanx': np.arctan(x)
    }, x)

writer.close()
