import cv2
import numpy as np
import torch
import os
import glob


def saveImage(path, img, savefilename, addname=''):
    # 传入的img类型是float32类型，值在0-255范围内
    if (isinstance(img, torch.Tensor)):
        img = np.mean(img.squeeze(0).cpu().numpy(), axis=0).astype(np.float32)
    img_name = addname + path.split("\\")[-1]
    img_path = savefilename + "/" + img_name
    # img = img.astype(np.float64) 前面代码修改了传入的img类型，暂时不需要此行代码
    img = img.astype(np.int32)
    img = img[10:-10, 10:-10]
    # 增加，显示局部50 * 50 区域
    # img = img[50:250, 10:210]
    cv2.imwrite(img_path, img)


# wholePaths = glob.glob(os.path.join("solar", "*jpg"))
# wholePaths = sorted(wholePaths)
# for wholePath in wholePaths:
#     img_name = wholePath.split("\\")[-1]
#     print("img_name = ", img_name)
#
# print("wholePath = ", wholePaths[0])