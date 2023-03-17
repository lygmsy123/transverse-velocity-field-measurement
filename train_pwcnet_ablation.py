from __future__ import print_function, division

import random
import sys


from core.unet.ablation_experments.extra_loss.pwc import PWCNETUL
import argparse
import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import evaluate
import datasets
from torch.utils.tensorboard import SummaryWriter
from core.utils.warp_occ_utils import flow_warp
from core.losses.loss_blocks import loss_smooth, loss_photometric, loss_gradient


sys.path.append('core')
par_dir = os.path.dirname(os.path.abspath(__file__))
os.chdir(par_dir)

try:
    from torch.cuda.amp import GradScaler
except:
    # dummy GradScaler for PyTorch < 1.6
    class GradScaler:
        def __init__(self):
            pass

        def scale(self, loss):
            return loss

        def unscale_(self, optimizer):
            pass

        def step(self, optimizer):
            optimizer.step()

        def update(self):
            pass

# exclude extremly large displacements
MAX_FLOW = 400
SUM_FREQ = 100
VAL_FREQ = 5000


def data_loss(img1, img2, flow):
    # 数据项包含：1 光度损失 2 梯度损失
    # 目前未考虑遮挡
    warped_img = flow_warp(img1, flow)
    loss_ph = loss_photometric(warped_img, img2)
    loss_gr = loss_gradient(warped_img, img2)
    return loss_ph, loss_gr


def data_loss_msy(img1, img2, flow, img1_org, img2_org, occ=1):
    # 数据项包含：1 光度损失 2 梯度损失
    # 目前未考虑遮挡
    warped_img = flow_warp(img1, flow)
    warped_img_org = flow_warp(img1_org, flow)
    loss_ph = loss_photometric(warped_img_org, img2_org, occ)
    loss_gr = loss_gradient(warped_img, img2, occ)
    return loss_ph, loss_gr


def smooth_loss(flow, image, alpha=10):
    loss_1st, loss_2nd = loss_smooth(flow, image, alpha)
    return loss_1st, loss_2nd


def sequence_loss_msy(flow_preds_fw, flow_gt, image1, image2, valid, image1_org, image2_org, gamma=0.8, max_flow=MAX_FLOW, occu_mask=None):
    """ Loss function defined over sequence of flow predictions """
    n_predictions = len(flow_preds_fw)
    flow_loss = 0.0
    # exlude invalid pixels and extremely large diplacements
    mag = torch.sum(flow_gt ** 2, dim=1).sqrt()
    valid = (valid >= 0.5) & (mag < max_flow)  # torch.Size([1,368,768])
    for i in range(n_predictions):
        exponent = n_predictions - i - 1
        i_weight = gamma ** exponent
        b, _, h, w = flow_preds_fw[i].size()
        flow_gt_scaled = F.interpolate((flow_gt / (2 ** exponent)), (h, w), mode='bilinear')
        i_loss = (flow_preds_fw[i] - flow_gt_scaled).abs()  # torch.Size([1,2,368,768])
        valid_scaled = (F.interpolate(valid[:, None].float(), (h, w), mode='bilinear') > 0.8)
        flow_loss += i_weight * (valid_scaled * i_loss).mean()
        # 添加的额外的约束，即非EPE损失
        image1_scaled = F.interpolate(image1, (h, w), mode='bilinear')
        image2_scaled = F.interpolate(image2, (h, w), mode='bilinear')
        image1_org_scaled = F.interpolate(image1_org, (h, w), mode='bilinear')
        image2_org_scaled = F.interpolate(image2_org, (h, w), mode='bilinear')
        l_ph, l_gr = data_loss_msy(image1_scaled, image2_scaled, flow_preds_fw[i], image1_org_scaled, image2_org_scaled)
        # l_gr = data_loss(image1_scaled, image2_scaled, flow_preds_fw[i])
        l_1st, l_2nd = smooth_loss(flow_preds_fw[i], image2_scaled)
        # l_bw = 0.5 * l_gr + 0.05 * (l_1st + l_2nd)
        l_bw = 0.5 * l_ph + 0.5 * l_gr + 0.05 * (l_1st + l_2nd)
        flow_loss += 0.1 * l_bw * i_weight

    epe = torch.sum((flow_preds_fw[-1] - flow_gt) ** 2, dim=1).sqrt()
    epe = epe.view(-1)[valid.view(-1)]  # torch.Size([282624])

    metrics = {
        'epe': epe.mean().item(),
        '1px': (epe < 1).float().mean().item(),
        '3px': (epe < 3).float().mean().item(),
        '5px': (epe < 5).float().mean().item(),
        'loss': flow_loss.item(),
    }
    return flow_loss, metrics


def sequence_loss(flow_preds_fw, flow_gt, image1, image2, valid, gamma=0.8, max_flow=MAX_FLOW, occu_mask=None):
    """ Loss function defined over sequence of flow predictions """
    n_predictions = len(flow_preds_fw)
    flow_loss = 0.0
    # exlude invalid pixels and extremely large diplacements
    mag = torch.sum(flow_gt ** 2, dim=1).sqrt()
    valid = (valid >= 0.5) & (mag < max_flow)  # torch.Size([1,368,768])
    for i in range(n_predictions):
        exponent = n_predictions - i - 1
        i_weight = gamma ** exponent
        b, _, h, w = flow_preds_fw[i].size()
        flow_gt_scaled = F.interpolate((flow_gt / 2 ** exponent), (h, w), mode='bilinear')
        i_loss = (flow_preds_fw[i] - flow_gt_scaled).abs()  # torch.Size([1,2,368,768])
        valid_scaled = (F.interpolate(valid[:, None].float(), (h, w), mode='bilinear') > 0.8)
        flow_loss += i_weight * (valid_scaled * i_loss).mean()

    epe = torch.sum((flow_preds_fw[-1] - flow_gt) ** 2, dim=1).sqrt()
    epe = epe.view(-1)[valid.view(-1)]  # torch.Size([282624])

    metrics = {
        'epe': epe.mean().item(),
        '1px': (epe < 1).float().mean().item(),
        '3px': (epe < 3).float().mean().item(),
        '5px': (epe < 5).float().mean().item(),
        'loss': flow_loss.item(),
    }
    return flow_loss, metrics


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)  # numel()返回数组中元素的个数


def fetch_optimizer(args, model):
    """ Create the optimizer and learning rate scheduler """
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.wdecay, eps=args.epsilon)

    scheduler = optim.lr_scheduler.OneCycleLR(optimizer, args.lr, args.num_steps + 100,
                                              pct_start=0.05, cycle_momentum=False, anneal_strategy='linear')

    return optimizer, scheduler


class Logger:
    def __init__(self, model, scheduler):
        self.model = model
        self.scheduler = scheduler
        self.total_steps = 0
        self.running_loss = {}
        self.writer = None

    def _print_training_status(self):
        metrics_data = [self.running_loss[k] / SUM_FREQ for k in sorted(self.running_loss.keys())]
        training_str = "[{:6d}, {:10.7f}] ".format(self.total_steps + 1, self.scheduler.get_last_lr()[0])
        metrics_str = ("{:10.4f}, " * len(metrics_data)).format(*metrics_data)

        # print the training status
        print(training_str + metrics_str)

        if self.writer is None:
            self.writer = SummaryWriter()

        for k in self.running_loss:
            self.writer.add_scalar(k, self.running_loss[k] / SUM_FREQ, self.total_steps)
            self.running_loss[k] = 0.0

    def push(self, metrics):
        self.total_steps += 1

        for key in metrics:  # metrics中以字典形式存储了epe，1px，3px和5px的数据
            if key not in self.running_loss:
                self.running_loss[key] = 0.0

            self.running_loss[key] += metrics[key]

        if self.total_steps % SUM_FREQ == SUM_FREQ - 1:
            self._print_training_status()
            self.running_loss = {}

    def write_dict(self, results):
        if self.writer is None:
            self.writer = SummaryWriter()

        for key in results:
            self.writer.add_scalar(key, results[key], self.total_steps)

    def close(self):
        self.writer.close()


def train01(args):
    model = nn.DataParallel(PWCNETUL(args), device_ids=args.gpus)
    print("Parameter Count: %d" % count_parameters(model))

    if args.restore_ckpt is not None:
        model.load_state_dict(torch.load(args.restore_ckpt), strict=False)
        # 加载已训练模型参数，strict为False，来自加载模型的keys不需要与torch的state_dict函数方法完全匹配
    model.cuda()
    model.train()  # 将模块设置为训练模式

    # if args.stage != 'chairs':  # and args.stage != 'solar':
    #     model.module.freeze_bn()  # 判断m的类型是否与nn.BatchNorm2d的类型相同，相同执行一个字符串表达式，并返回表达式的值

    train_loader = datasets.fetch_dataloader(args)  # 获取数据加载器
    optimizer, scheduler = fetch_optimizer(args, model)

    total_steps = 0
    scaler = GradScaler(enabled=args.mixed_precision)
    logger = Logger(model, scheduler)

    VAL_FREQ = 100  # 5000
    add_noise = True

    should_keep_training = True
    while should_keep_training:

        for i_batch, data_blob in enumerate(train_loader):
            optimizer.zero_grad()
            image1, image2, flow, valid, image1_org, image2_org = [x.cuda() for x in data_blob]

            if args.add_noise:
                stdv = np.random.uniform(0.0, 5.0)
                image1 = (image1 + stdv * torch.randn(*image1.shape).cuda()).clamp(0.0, 255.0)
                image2 = (image2 + stdv * torch.randn(*image2.shape).cuda()).clamp(0.0, 255.0)

            flow_prediction_fw, occu_mask, _ = model(image1, image2)
            # loss, metrics = sequence_loss(flow_prediction_fw, flow,
            #                               image1, image2, valid, args.gamma, occu_mask=occu_mask)
            loss, metrics = sequence_loss_msy(flow_prediction_fw, flow,
                                              image1, image2, valid, image1_org, image2_org, args.gamma,
                                              occu_mask=occu_mask)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip)

            # 添加assert来判断更新前参数是否为NaN

            scaler.step(optimizer)
            scheduler.step()
            scaler.update()

            logger.push(metrics)

            if total_steps % VAL_FREQ == VAL_FREQ - 1:
                PATH = 'checkpoints/%d_%s.pth' % (total_steps + 1, args.name)
                torch.save(model.state_dict(), PATH)

                results = {}
                for val_dataset in args.validation:
                    if val_dataset == 'chairs':
                        results.update(evaluate.validate_chairs(model.module))
                    elif val_dataset == 'sintel':
                        results.update(evaluate.validate_sintel(model.module))
                    elif val_dataset == 'kitti':
                        results.update(evaluate.validate_kitti(model.module))
                    elif val_dataset == 'solar':
                        results.update(evaluate.validate_solar(model.module))
                        # results.update(evaluate.validate_solarzero(model.module))
                        # results.update(evaluate.validate_optical(model.module))

                logger.write_dict(results)

                model.train()
                # if args.stage != 'chairs':  # and args.stage != 'solar':
                #     model.module.freeze_bn()

            total_steps += 1

            if total_steps > args.num_steps:
                should_keep_training = False
                break

    logger.close()
    PATH = 'checkpoints/%s.pth' % args.name
    torch.save(model.state_dict(), PATH)

    return PATH


if __name__ == '__main__':
    # 训练：--name raft-solar_dis --stage solardis --restore_ckpt checkpoints/raft-solar.pth --validation solardis --gpus 0 --num_steps 60000 --batch_size 2 --lr 0.001 --image_size 384 424 --wdecay 0.0001 --gamma=0.5 --add_bw_flow
    parser = argparse.ArgumentParser()
    parser.add_argument('--name', default='raft', help="name your experiment")
    parser.add_argument('--stage', help="determines which dataset to use for training")
    parser.add_argument('--restore_ckpt', help="restore checkpoint")
    parser.add_argument('--small', action='store_true', help='use small model')  # action当参数在命令行中出现时使用的动作基本类型
    parser.add_argument('--validation', type=str, nargs='+')  # type 命令行参数应当被转换成的类型；nargs 命令行参数应当消耗的数目

    parser.add_argument('--lr', type=float, default=0.00002)
    parser.add_argument('--num_steps', type=int, default=100000)
    parser.add_argument('--batch_size', type=int, default=6)
    parser.add_argument('--image_size', type=int, nargs='+', default=[384, 512])
    parser.add_argument('--gpus', type=int, nargs='+', default=[0, 1])
    parser.add_argument('--mixed_precision', action='store_true', help='use mixed precision')

    parser.add_argument('--iters', type=int, default=8)
    parser.add_argument('--wdecay', type=float, default=.00005)
    parser.add_argument('--epsilon', type=float, default=1e-8)  # ε
    parser.add_argument('--clip', type=float, default=1.0)
    parser.add_argument('--dropout', type=float, default=0.0)
    parser.add_argument('--gamma', type=float, default=0.8, help='exponential weighting')
    parser.add_argument('--add_noise', action='store_true')  # action保存相应布尔值，store_true表示触发action为真，否则为假
    parser.add_argument('--add_bw_flow', action='store_true', help='add backward flow')
    parser.add_argument('--small_bw', action='store_true', help='add small module backward flow')
    args = parser.parse_args()

    random.seed(1234)
    torch.manual_seed(1234)  # 设置CPU生成随机数的种子，方便下次复现实验结果,能使得调用torch.rand(n)的结果一致，
    torch.cuda.manual_seed(1234)
    np.random.seed(1234)  # 注：是每次运行py文件的输出结果一样，而不是每次随机函数生成的结果一样
    torch.backends.cudnn.deterministic = True
    # torch.backends.cudnn.deterministic = True
    #  np.random.seed(n) 用于生成指定随机数，n代表第n堆种子，想每次都能得到相同的随机数，每次产生随机数之前，都需要调用一次seed()
    if not os.path.isdir('checkpoints'):
        os.mkdir('checkpoints')

    train01(args)
