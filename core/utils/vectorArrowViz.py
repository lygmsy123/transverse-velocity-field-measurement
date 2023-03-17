import argparse

import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
import numpy as np

import cmaps
import os
import glob
from core.utils import frame_utils
from core.utils import flow_viz


def arrow_viz(flo, img, save_path, imgname, positionX=50, positionY=10, interval=30):
    # count = 0

    # flow_root = glob.glob(os.path.join(flo_path, '*.flo'))
    # flow_root = sorted(flow_root)
    # img_root = glob.glob(os.path.join(img_path, '*.jpg'))
    # img_root = sorted(img_root)

    filename = imgname.split("/")[-1]
    flow = flo.squeeze(0).cpu().numpy()
    img = img.squeeze(0).cpu().numpy()
    flow = np.array(flow).astype(np.float32) * -1
    img = np.array(img).astype(np.uint8)
    flow = flow[:, 10:-10, 10:-10]
    img = img[:, 10:-10, 10:-10]
    # flow = flow[:, positionX:positionX+200, positionY:positionY+200]
    # img = img[:, positionX:positionX + 200, positionY:positionY + 200]
    flow = flow[:, positionX+20:positionX+70, positionY+10:positionY+60]
    img = img[:, positionX+20:positionX + 70, positionY+10:positionY + 60]
    c, h, w = flow.shape
    u = flow[0]
    v = flow[1] * -1
    x = np.linspace(0, w, w)
    y = np.linspace(0, h, h)
    # 左上角为坐标系，垂直方向为x，水平方向为y
    coord_y, coord_x = np.meshgrid(x, y)
    # 由于坐标系为左下角，而coord和uv均为左上角，将coord_x和v取反即可（即垂直方向反序排列）
    # 而对于u，v位移，与正常理解的是反的，所以 u = u * -1, v = v * -1
    coord_x = coord_x[::-1, :]
    v = v[::-1, :]
    # 颜色矩阵构建
    # flow_color = viz(flow)
    ws_map = [(0, 0.5), (0.5, 1), (1, 1.5), (1.5, 2), (2, 2.5), (2.5, 3), (3, 3.5), (3.5, 4), (4, 100)]
    flow_color = np.zeros_like(u, dtype=float)
    flowspeed = np.sqrt(u ** 2 + v ** 2)
    # flowspeedmax = np.max(flowspeed)
    u_norm = u  # / 2  # flowspeedmax * interval
    v_norm = v  # / 2  # flowspeedmax * interval
    for i in range(len(ws_map)):
        flow_color[np.where((flowspeed > ws_map[i][0]) & (flowspeed <= ws_map[i][1]))] = i
    norm = Normalize()
    norm.autoscale(flow_color)
    fig = plt.figure(1, facecolor='white', dpi=200)
    ax = plt.axes()
    plt.axis('off')
    # 这里指定数据点的坐标系原点在xy轴的左下角，而注释的坐标系原点在这个图像(figure)
    # 的左下角所以才会出现注释内容下移覆盖了x轴
    # fig.clf()
    # plt.gca().invert_yaxis()
    img_gray = np.mean(img, axis=0)
    plt.imshow(img_gray, cmap="Greys_r")
    coord_x = coord_x[interval::interval, interval::interval]
    coord_y = coord_y[interval::interval, interval::interval]
    u = u_norm[interval::interval, interval::interval]
    v = v_norm[interval::interval, interval::interval]
    flow_color = flow_color[interval::interval, interval::interval]
    # ax.quiver(coord_x, coord_y, u, v, angles='xy', scale_units='xy', scale=1)
    ax.quiver(coord_y, coord_x, u, v, norm(flow_color), cmap=cmaps.amwg_blueyellowred, width=0.01, scale=15, headwidth=3,
              alpha=1, units='inches')
    f = plt.gcf()
    rgb_path = save_path + "/" + filename
    f.savefig(rgb_path)
    f.clear()
    fig.clear()
    # count = count + 1
    # print("第 ", count, " 处理完毕")