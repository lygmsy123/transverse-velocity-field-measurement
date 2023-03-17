
"""
    该函数是计算非刚性配准的客观评价指标
    PSNR:两幅图像之间的峰值信噪比度量 （值越大，图像的配准质量越好）
    SSIM:两幅图像之间的结构相似度（越接近1越好）
    CC:两幅图像之间的相关系数（越接近1越好）
    RV：两幅图像之间的残差图方差（越小，说明算法越稳定）
    RM：两幅图像之间的残差图均值（越小，说明算法越稳定）
"""


import numpy as np
import math
import cv2


def psnr(target, ref, border=0):
    # target: 目标图像   ref: 参考图像   scale: 尺寸大小  border:边缘大小（即不参与计算的部分）
    # assume RGB image
    if not target.shape == ref.shape:
        raise ValueError('Input images must have the same dimensions.')
    b, c, h, w = target.shape
    target = target.squeeze(0).cpu().numpy()
    ref = ref.squeeze(0).cpu().numpy()
    target = target[:, border:h-border, border:w-border]
    ref = ref[:, border:h-border, border:w-border]

    target = target.astype(np.float64)
    ref = ref.astype(np.float64)
    mse = np.mean((target - ref) ** 2)

    if mse == 0:
        return float('inf')
    return 20 * math.log10(255.0 / math.sqrt(mse))


def ssim(target, ref, border=0):

    def ssim_two(img1, img2):
        C1 = math.pow((0.01 * 255), 2)
        C2 = math.pow((0.03 * 255), 2)

        img1 = img1.astype(np.float64)
        img2 = img2.astype(np.float64)
        kernel = cv2.getGaussianKernel(11, 1.5)
        window = np.outer(kernel, kernel.transpose())

        mu1 = cv2.filter2D(img1, -1, window)[5:-5, 5:-5]  # valid
        mu2 = cv2.filter2D(img2, -1, window)[5:-5, 5:-5]  # valid
        mu1_sq = mu1 ** 2
        mu2_sq = mu2 ** 2
        mu1_mu2 = mu1 * mu2
        sigma1_sq = cv2.filter2D(img1 ** 2, -1, window)[5:-5, 5:-5] - mu1_sq
        sigma2_sq = cv2.filter2D(img2 ** 2, -1, window)[5:-5, 5:-5] - mu2_sq
        sigma12 = cv2.filter2D(img1 * img2, -1, window)[5:-5, 5:-5] - mu1_mu2

        ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) *
                                                                (sigma1_sq + sigma2_sq + C2))
        return ssim_map.mean()

    if not target.shape == ref.shape:
        raise ValueError('Input images must have the same dimensions.')
    b, c, h, w = target.shape
    target = target.squeeze(0).cpu().numpy()
    ref = ref.squeeze(0).cpu().numpy()
    target = target[:, border:h - border, border:w - border]
    ref = ref[:, border:h - border, border:w - border]
    if c == 1:
        return ssim_two(np.squeeze(target), np.squeeze(ref))
    elif c == 3:
        ssims = []
        for i in range(3):
            ssims.append(ssim_two(target[i], ref[i]))
        return np.array(ssims).mean()


def cc(target, ref, border=0):
    if not target.shape == ref.shape:
        raise ValueError('Input images must have the same dimensions.')
    b, c, h, w = target.shape
    target = target.squeeze(0).cpu().numpy()
    ref = ref.squeeze(0).cpu().numpy()
    target = target[:, border:h - border, border:w - border]
    ref = ref[:, border:h - border, border:w - border]
    target = target.astype(np.float64)
    ref = ref.astype(np.float64)
    target_E = target - np.mean(target)
    ref_E = ref - np.mean(ref)
    r = np.sum(target_E * ref_E) / np.sqrt(np.sum(target_E * target_E) * np.sum(ref_E * ref_E))
    return r


def rv_rm(target, ref, border=0):
    if not target.shape == ref.shape:
        raise ValueError('Input images must have the same dimensions.')
    b, c, h, w = target.shape
    target = target.squeeze(0).cpu().numpy()
    ref = ref.squeeze(0).cpu().numpy()
    target = target[:, border:h - border, border:w - border]
    ref = ref[:, border:h - border, border:w - border]
    target = target.astype(np.float64)
    ref = ref.astype(np.float64)
    # res计算target和ref之间的残差图
    res = np.abs(target - ref)
    rm = np.mean(res)
    rv = np.var(res)
    return rv, rm

