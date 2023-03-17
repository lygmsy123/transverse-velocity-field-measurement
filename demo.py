import sys

from core.unet.ablation_experments.extra_loss.pwc import PWCNETUL
from core.utils import flow_viz
from core.utils.utils import InputPadder

sys.path.append('core')

import argparse
import os
import cv2
import glob
import numpy as np
import torch
import matplotlib.pyplot as plt
from PIL import Image
# ------
from comp_dif import imgs_dif
from saveWrapedImage import saveImage
from evaluationIndicators.warp import flow_warp
from evaluationIndicators.utils import psnr, ssim, cc, rv_rm
from core.utils.vectorArrowViz import arrow_viz
# import scipy.io as scio
# import torch.nn.functional as F
# ------


DEVICE = 'cuda'


def flow_mean(flo):
    flo_mag = torch.sum(flo ** 2, dim=1).sqrt()
    flo_mag = flo_mag[:, 50:250, 10:210]
    flow_m = torch.mean(flo_mag)
    return flow_m


def plt_save(img, name):
    if (isinstance(img, torch.Tensor)):
        img = np.mean(img.squeeze(0).cpu().numpy(), axis=0).astype(np.float32)
    fig = plt.figure(1, facecolor='white', dpi=200)
    ax = plt.axes()
    plt.axis('off')
    plt.imshow(img, cmap="Greys_r")
    f = plt.gcf()
    rgb_path = "datasets/" + name + ".jpg"
    f.savefig(rgb_path)
    f.clear()
    fig.clear()


def load_image(imfile):
    img = np.array(Image.open(imfile)).astype(np.uint8)
    # img = np.array(Image.open(imfile)).astype(np.int16)
    # img = (img + 32768).astype(np.int32)
    # img = ((img + 32768).astype(np.float32) / 256.0).astype(np.uint8)
    # img = ((img * 255.0) // 32767).astype(np.uint8)
    if len(img.shape) == 2:
        img = np.tile(img[..., None], (1, 1, 3))
    img = torch.from_numpy(img).permute(2, 0, 1).float()
    return img[None].to(DEVICE)


def viz(flo, filepath):
    # 自己定义的存储图像光流的文件夹路径和名字
    # filepath 是读取的图像的路径，借用该路径获取图像名，
    # 暂时限死flo_viz的存储地址为saveWrapped/flo_viz
    img_name = filepath.split("\\")[-1]
    img_path = "saveWrapped/flo_viz/ours/" + img_name
    # img = img[0].permute(1,2,0).cpu().numpy()
    flo = flo[0].permute(1,2,0).cpu().numpy()
    
    # map flow to rgb image
    flo = flow_viz.flow_to_image(flo)  # [532, 580, 3]
    # flo = flo[50:250, 10:210, :]
    # flo = flo[70:120, 20:70, :]
    fig = plt.figure(1, facecolor='white', dpi=200)
    plt.axis('off')
    plt.imshow(flo)
    f = plt.gcf()
    f.savefig(img_path)
    f.clear()
    fig.clear()
    # cv2.imwrite(img_path, flo)
    # img_flo = np.concatenate([img, flo], axis=0)

    # import matplotlib.pyplot as plt
    # plt.imshow(img_flo / 255.0)
    # plt.show()

    # cv2.imshow('image', img_flo[:, :, [2,1,0]]/255.0)
    # cv2.waitKey()


def demo(args):

    model = torch.nn.DataParallel(PWCNETUL(args))
    model.load_state_dict(torch.load(args.model))

    model = model.module
    model.to(DEVICE)
    model.eval()
    # --------------------------

    # -------------------------
    total_res_mean = 0
    total_res_variance = 0
    # 用于统计循环次数，便于统计度量均值大小
    metrics_count = 0
    total_sou_mean = 0
    total_sou_variance = 0
    count = 0

    total_res_cc, res_cc = 0, 0
    total_sou_cc, sou_cc = 0, 0

    total_res_ssim, res_ssim = 0, 0
    total_sou_ssim, sou_ssim = 0, 0
    flow_u, flow_v = 0.0, 0.0
    # flow_ustd, flow_vstd = 0.0, 0.0
    # ------------------------

    with torch.no_grad():
        image1s = glob.glob(os.path.join(args.path, '1', '*.png')) + \
                 glob.glob(os.path.join(args.path, '1', '*.jpg'))
        image2s = glob.glob(os.path.join(args.path, '2', '*.png')) + \
                 glob.glob(os.path.join(args.path, '2', '*.jpg'))
        for imfile1, imfile2 in zip(image1s, image2s):
            count += 1
            image1 = load_image(imfile1)
            image2 = load_image(imfile2)

            padder = InputPadder(image1.shape)  # 对图像进行填充，使得尺寸能够被8整除
            image1, image2 = padder.pad(image1, image2) # image1.size=[1,3,552,600] torch.float32类型 value=[0,255]

            # flow_low, flow_up, occu_mask = model(image1, image2, iters=12, test_mode=True, add_bw=args.add_bw_flow, small_bw=args.small_bw)  # flow_up[1,2,552,600]
            flow_low, flow_up, occu_mask = model(image1, image2, iters=12, test_mode=True)  # flow_up[1,2,552,600]
            # viz(flow_up[:, :, 10:-10, 10:-10], imfile1)
            warp_img = flow_warp(image1, flow_up)
            # plt_save(warp_img[:, :, 60:260, 20:220], name="warp")
            # plt_save(image1[:, :, 60:260, 20:220], name="image1")
            # plt_save(image2[:, :, 60:260, 20:220], name="image2")

            # saveImage(imfile1, warp_img, "saveWrapped/ours")  # imfile1
            # arrow_viz(flow_up, image1, save_path="saveWrapped/arrow/ours", imgname=imfile1, interval=2)
            # img_dif = imgs_dif(warp_img, image2)
            # plt_save(img_dif[60:260, 20:220], name="img_dif")
            # saveImage(imfile1, img_dif, "residualImage/ours", 'residual-')

            # 新添加的度量 cc（相关系数）

            # 对optical数据集2
            # image1 = image1[:, :, 2:-2, 2:-2]
            # image2 = image2[:, :, 2:-2, 2:-2]
            # warp_img = warp_img[:, :, 2:-2, 2:-2]

            border = 10
            sou_variance, sou_mean = rv_rm(image1, image2, border)
            res_variance, res_mean = rv_rm(warp_img, image2, border)
            sou_ssim = ssim(image1, image2, border)
            res_ssim = ssim(warp_img, image2, border)
            sou_cc = cc(image1, image2, border)
            res_cc = cc(warp_img, image2, border)
            print(count, "  ", "source_ssim = ", sou_ssim, "\tresidual_ssim = ", res_ssim)
            print(count, "  ", "source_cc = ", sou_cc, "\tresidual_cc = ", res_cc)
            print(count, "  ", "source_mean = ", sou_mean, "\tsource_variance = ", sou_variance)
            print(count, "  ", "residual_mean = ", res_mean, "\tresidual_variance = ", res_variance)
            total_sou_cc += sou_cc
            total_sou_ssim += sou_ssim
            total_sou_mean += sou_mean
            total_sou_variance += sou_variance
            total_res_mean += res_mean
            total_res_variance += res_variance
            total_res_cc += res_cc
            total_res_ssim += res_ssim

            flow_u += torch.mean(flow_up[0, 0, 10:-10, 10:-10])
            flow_v += torch.mean(flow_up[0, 1, 10:-10, 10:-10])
            # flow_ustd += torch.std(flow_up[0, 0, 10:-10, 10:-10])
            # flow_vstd += torch.std(flow_up[0, 1, 10:-10, 10:-10])

    total_sou_mean = total_sou_mean / count
    total_sou_variance = total_sou_variance / count
    total_res_mean = total_res_mean / count
    total_res_variance = total_res_variance / count
    total_sou_cc = total_sou_cc / count
    total_res_cc = total_res_cc / count
    total_sou_ssim = total_sou_ssim / count
    total_res_ssim = total_res_ssim / count
    # total_res_mean_noc = total_res_mean_noc / count
    # total_res_variance_noc = total_res_variance_noc / count

    print("total_sou_ssim = ", total_sou_ssim, "\ttotal_res_ssim = ", total_res_ssim)
    print("total_sou_cc = ", total_sou_cc, "\ttotal_res_cc = ", total_res_cc)
    print("total_sou_mean = ", total_sou_mean, "\ttotal_sou_variance = ", total_sou_variance)
    print("total_res_mean = ", total_res_mean, "\ttotal_res_variance = ", total_res_variance)
    print("total_flow_u_mean = ", (flow_u / len(image1s)), "\ttotal_flow_v_mean = ", (flow_v / len(image1s)))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', help="restore checkpoint")
    parser.add_argument('--path', help="dataset for evaluation")
    parser.add_argument('--small', action='store_true', help='use small model')
    parser.add_argument('--mixed_precision', action='store_true', help='use mixed precision')
    parser.add_argument('--alternate_corr', action='store_true', help='use efficent correlation implementation')
    parser.add_argument('--datatest', action='store_true', help='test dataset need several scene files')
    parser.add_argument('--add_bw_flow', action='store_true', help='add backward flow')
    parser.add_argument('--small_bw', action='store_true', help='add backward flow of small flow module')
    args = parser.parse_args()
    # --model=models/raft-solar22.pth --path=filament_image --add_bw_flow --datatest
    # --model=models/raft-solar22.pth --path=datasets/Solar_demo/train --add_bw_flow
    demo(args)
