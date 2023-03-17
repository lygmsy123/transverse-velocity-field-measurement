
"""
    该函数是计算两幅图像之间的峰值信噪比度量（PSNR）
"""


import numpy as np
import math

def psnr(target, ref, border=0):
    # target: 目标图像   ref: 参考图像   scale: 尺寸大小  border:边缘大小（即不参与计算的部分）
    # assume RGB image
    if not target.shape == ref.shape:
        raise ValueError('Input images must have the same dimensions.')
    b, c, h, w = target.shape
    target = target[:, :, border:h-border, border:w-border]
    ref = ref[:, :, border:h-border, border:w-border]

    target = target.astype(np.float64)
    ref = ref.astype(np.float64)
    mse = np.mean((target - ref) ** 2)

    if mse == 0:
        return float('inf')
    return 
