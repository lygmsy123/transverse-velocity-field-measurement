import sys

# from core.traft import TRAFT
# from core.module30.raft_msy import MRAFT
# from core.module27.raft_27_01 import TRAFT
# from core.raft_27 import FRAFT  # ********************重要模型********************
# from core.module27.raft_27_03 import FRAFT
# from core.module27.raft_27_04 import FRAFT
# from core.module27.raft_27_05 import FRAFT
# from core.module27.raft_27_07 import FRAFT
# from core.module27_improve.raft06_04_08 import FRAFT
# from core.convneXT.add_depth_conv.raft import FRAFT
# from core.unet.original_net.raft import FRAFT
# from core.unet.increase_skip_layer_net.raft import FRAFT
# from core.unet.improve_res_net.module01.raft import FRAFT
from core.unet.improve_res_net.module04.raft import FRAFT
# from core.unet.improve_res_net.module07.raft import FRAFT
from core.utils import flow_viz
from core.utils.utils import InputPadder

sys.path.append('core')

import argparse
import os
import cv2
import glob
import numpy as np
import torch
from PIL import Image
# ------
from comp_dif import imgs_dif
from saveWrapedImage import saveImage  # 后添加的
from calResidualMetrics import calmetrics  # 计算残差度量（均值和方差）
from calSourceImage import calsource
from wrap_image_rewrite import Wrap_Rewrite

# import scipy.io as scio
# import torch.nn.functional as F
# ------


DEVICE = 'cuda'


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


def viz(img, flo):
    img = img[0].permute(1, 2, 0).cpu().numpy()
    flo = flo[0].permute(1, 2, 0).cpu().numpy()

    # map flow to rgb image
    flo = flow_viz.flow_to_image(flo)
    img_flo = np.concatenate([img, flo], axis=0)

    # import matplotlib.pyplot as plt
    # plt.imshow(img_flo / 255.0)
    # plt.show()

    cv2.imshow('image', img_flo[:, :, [2, 1, 0]] / 255.0)
    cv2.waitKey()


def demo(args):
    # model = torch.nn.DataParallel(RAFT(args))
    # model = torch.nn.DataParallel(TRAFT(args))
    # model = torch.nn.DataParallel(MRAFT(args))
    model = torch.nn.DataParallel(FRAFT(args))
    model.load_state_dict(torch.load(args.model))

    model = model.module
    model.to(DEVICE)
    model.eval()  # 不启用BatchNormalization和Dropout,pytorch框架会自动把BN和D固定住
    # --------------------------
    # dataFile = 'flow.mat'
    # dataf = scio.loadmat(dataFile)
    # flow_msy = dataf['flow']
    # flow_msy = torch.from_numpy(flow_msy)
    # flow_msy = flow_msy.unsqueeze(0)
    # # print(flow_msy.shape)
    # h, w = flow_msy.shape[-2:]
    # # print(h, w)
    # pad_h = (((h // 8) + 1) * 8 - h) % 8  # 这两段的目的就是获取能被n整除，尺寸需要填充的像素数
    # pad_w = (((w // 8) + 1) * 8 - w) % 8
    # # print(pad_h, pad_w)
    # pad = [pad_w // 2, pad_w - pad_w // 2, 0, pad_h]
    # # print(pad)
    # flow_msy = F.pad(flow_msy, pad, mode='replicate')
    # # print(flow_msy.shape)
    # flow_visual = flow_msy[0].permute(1, 2, 0).cpu().numpy()
    # flow_visual = flow_viz.flow_to_image(flow_visual)
    # cv2.imshow('flow', flow_visual / 255.0)
    # -------------------------
    total_res_mean = 0
    total_res_variance = 0
    # 用于统计循环次数，便于统计度量均值大小
    metrics_count = 0
    total_sou_mean = 0
    total_sou_variance = 0
    count = 0

    total_res_mean_noc = 0
    total_res_variance_noc = 0
    # ------------------------

    with torch.no_grad():
        images = glob.glob(os.path.join(args.path, '*.png')) + \
                 glob.glob(os.path.join(args.path, '*.jpg'))

        images = sorted(images)
        for imfile1, imfile2 in zip(images[:-1], images[1:]):
            metrics_count += 1  # 本行后添加
            if args.datatest:
                count += 1
                if count % 2 == 1:
                    image1 = load_image(imfile1)
                    image2 = load_image(imfile2)
                else:
                    continue
            else:
                # for i in range(len(images) // 2):
                image2 = load_image(imfile1)
                image1 = load_image(imfile2)
            # if i >= 167:
            #     break
            # image1 = load_image(images[2 * i])
            # image2 = load_image(images[2 * i + 1])

            padder = InputPadder(image1.shape)  # 对图像进行填充，使得尺寸能够被8整除
            image1, image2 = padder.pad(image1, image2)  # image1.size=[1,3,552,600] torch.float32类型 value=[0,255]

            flow_low, flow_up, occu_mask = model(image1, image2, iters=12, test_mode=True, add_bw=args.add_bw_flow,
                                                 small_bw=args.small_bw)  # flow_up[1,2,552,600]
            # -----------------------------------
            # cv2.namedWindow('image', cv2.WINDOW_NORMAL) 在这种情况下，用cv2.namedWindow()函数可以指定窗口是否可以调整大小。
            # 在默认情况下，标志为cv2.WINDOW_AUTOSIZE。但是，如果指定标志为cv2.WINDOW_Normal，则可以调整窗口的大小。

            # wrap_img = Wrap(image1, flow_up)
            occ_sum = occu_mask.sum()
            wrap_img = Wrap_Rewrite(image1, flow_up, 0)
            saveImage(imfile1, wrap_img, "saveWrapped/Solar01")  # imfile1
            print(metrics_count, "  occu_mask_sum = ", occ_sum)
            occu_mask1 = ((occu_mask > 0.5).float().squeeze(0).squeeze(0).cpu().numpy() * 255).astype(np.uint8)
            saveImage(imfile1, occu_mask1, "Occulation_map")
            # cv2.imshow('wrap_image', occu_mask1 / 255.0)  # 因为此时的wrap_img类型为float32
            # cv2.waitKey()
            img_dif = imgs_dif(wrap_img, image2)
            saveImage(imfile1, img_dif, "residualImage/Solar01", 'residual-')
            res_mean, res_variance = calmetrics(img_dif)
            sou_mean, sou_variance = calsource(image1, image2)
            print(metrics_count, "  ", "source_mean = ", sou_mean, "\tsource_variance = ", sou_variance)
            print(metrics_count, "  ", "residual_mean = ", res_mean, "\tresidual_variance = ", res_variance)
            total_sou_mean += sou_mean
            total_sou_variance += sou_variance
            total_res_mean += res_mean
            total_res_variance += res_variance

            if args.small_bw or args.add_bw_flow:
                occu_mask1 = occu_mask.squeeze(0).squeeze(0).cpu().numpy()
                img_dif_noc = imgs_dif(wrap_img * occu_mask1, image2 * occu_mask)
                res_mean_noc, res_variance_noc = calmetrics(img_dif_noc)
                total_res_mean_noc += res_mean_noc
                total_res_variance_noc += res_variance_noc
                print(metrics_count, "  ", "residual_mean_noc = ", res_mean_noc, "\tresidual_variance_noc = ",
                      res_variance_noc)
            # cv2.imshow('Difference between images', img_dif * 255)  # 因为img_dif类型为int32，在imshow中会除以256，再映射到[0,255]
            # -----------------------------------
            # viz(image1, flow_up)
            # flo = flow_up[0].permute(1, 2, 0).cpu().numpy()
            # flo = flow_viz.flow_to_image(flo)
            # cv2.imshow('image', flo / 255.0)
            # cv2.waitKey()
    if args.datatest:
        count = (count + 1) / 2
        print(count)  # 49
        total_sou_mean = total_sou_mean / count
        total_sou_variance = total_sou_variance / count
        total_res_mean = total_res_mean / count
        total_res_variance = total_res_variance / count
        total_res_mean_noc = total_res_mean_noc / count
        total_res_variance_noc = total_res_variance_noc / count
    else:
        total_sou_mean = total_sou_mean / metrics_count
        total_sou_variance = total_sou_variance / metrics_count
        total_res_mean = total_res_mean / metrics_count
        total_res_variance = total_res_variance / metrics_count
        total_res_mean_noc = total_res_mean_noc / metrics_count
        total_res_variance_noc = total_res_variance_noc / metrics_count
    print("total_sou_mean = ", total_sou_mean, "\ttotal_sou_variance = ", total_sou_variance)
    print("total_res_mean = ", total_res_mean, "\ttotal_res_variance = ", total_res_variance)
    if args.small_bw or args.add_bw_flow:
        print("total_res_mean = ", total_res_mean_noc, "\ttotal_res_variance_noc = ", total_res_variance_noc)


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
