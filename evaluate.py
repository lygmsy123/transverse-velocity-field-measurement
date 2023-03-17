import sys


from core.unet.ablation_experments.extra_loss.pwc import PWCNETUL
from core.utils import frame_utils
from core.utils.utils import InputPadder, forward_interpolate
from evaluationIndicators.warp import flow_warp
from evaluationIndicators.utils import ssim, rv_rm

sys.path.append('core')

from PIL import Image
import argparse
import os
import time
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt

import datasets
par_dir = os.path.dirname(os.path.abspath(__file__))
os.chdir(par_dir)


@torch.no_grad()
def create_sintel_submission(model, iters=32, warm_start=False, output_path='sintel_submission'):
    """ Create submission for the Sintel leaderboard """
    model.eval()
    for dstype in ['clean', 'final']:
        test_dataset = datasets.MpiSintel(split='test', aug_params=None, dstype=dstype)
        
        flow_prev, sequence_prev = None, None
        for test_id in range(len(test_dataset)):
            image1, image2, (sequence, frame) = test_dataset[test_id]
            if sequence != sequence_prev:
                flow_prev = None
            
            padder = InputPadder(image1.shape)
            image1, image2 = padder.pad(image1[None].cuda(), image2[None].cuda())

            flow_low, flow_pr = model(image1, image2, iters=iters, flow_init=flow_prev, test_mode=True)
            flow = padder.unpad(flow_pr[0]).permute(1, 2, 0).cpu().numpy()

            if warm_start:
                flow_prev = forward_interpolate(flow_low[0])[None].cuda()
            
            output_dir = os.path.join(output_path, dstype, sequence)
            output_file = os.path.join(output_dir, 'frame%04d.flo' % (frame+1))

            if not os.path.exists(output_dir):
                os.makedirs(output_dir)

            frame_utils.writeFlow(output_file, flow)
            sequence_prev = sequence


@torch.no_grad()
def create_kitti_submission(model, iters=24, output_path='kitti_submission'):
    """ Create submission for the Sintel leaderboard """
    model.eval()
    test_dataset = datasets.KITTI(split='testing', aug_params=None)

    if not os.path.exists(output_path):
        os.makedirs(output_path)

    for test_id in range(len(test_dataset)):
        image1, image2, (frame_id, ) = test_dataset[test_id]
        padder = InputPadder(image1.shape, mode='kitti')
        image1, image2 = padder.pad(image1[None].cuda(), image2[None].cuda())

        _, flow_pr = model(image1, image2, iters=iters, test_mode=True)
        flow = padder.unpad(flow_pr[0]).permute(1, 2, 0).cpu().numpy()

        output_filename = os.path.join(output_path, frame_id)
        frame_utils.writeFlowKITTI(output_filename, flow)


@torch.no_grad()
def validate_solar(model, iters=2):  # 32
    """ Peform validation using the Sintel (train) split """
    model.eval()
    results = {}
    dstype = 'img'
    val_dataset = datasets.Solar(split='test', dstype=dstype)
    epe_list = []
    epenoc_list = []
    occu_mask_list = []
    flow_u = 0
    flow_v = 0
    total_ssim, total_rm, total_rv = 0, 0, 0
    # count = 0
    # print("len(val_dataset) = ", len(val_dataset))

    for val_id in range(len(val_dataset)):
        # count = count + 1
        # if count % 100 == 0:
        #     print("count = ", count)
        image1, image2, flow_gt = val_dataset[val_id]
        image1 = image1[None].cuda()
        image2 = image2[None].cuda()

        padder = InputPadder(image1.shape)
        image1, image2 = padder.pad(image1, image2)

        flow_low, flow_pr, occu_mask = model(image1, image2, iters=iters, test_mode=True, add_bw=True)
        # metric test
        warp_img = flow_warp(image1, flow_pr)
        res_variance, res_mean = rv_rm(warp_img, image2, 10)
        res_ssim = ssim(warp_img, image2, 10)
        total_ssim += res_ssim
        total_rm += res_mean
        total_rv += res_variance
        # metric test
        flow = padder.unpad(flow_pr[0]).cpu()

        epe = torch.sum((flow - flow_gt)**2, dim=0).sqrt()
        epe = epe[10:-10, 10:-10].contiguous()
        epe_list.append(epe.view(-1).numpy())
        # flow_u += torch.mean(flow[0, 10:-10, 10:-10])
        # flow_v += torch.mean(flow[1, 10:-10, 10:-10])

    epe_all = np.concatenate(epe_list)
    epe = np.mean(epe_all)
    px1 = np.mean(epe_all < 1)
    px3 = np.mean(epe_all<3)
    px5 = np.mean(epe_all<5)
    total_ssim = total_ssim / len(val_dataset)
    total_rm = total_rm / len(val_dataset)
    total_rv = total_rv / len(val_dataset)
    print("Validation (%s) EPE: %f, 1px: %f, 3px: %f, 5px: %f" % (dstype, epe, px1, px3, px5))
    results[dstype] = np.mean(epe_list)
    # print("Validation (%s) u_px: %f, v_px: %f" % (dstype, flow_u / len(val_dataset), flow_v / len(val_dataset)))
    print("Validation (%s) SSIM: %f, RM: %f, RV: %f" % (dstype, total_ssim, total_rm, total_rv))

    # epe in noc area
    # occu_sum = np.concatenate(occu_mask_list).sum()
    # epenoc_all = np.concatenate(epenoc_list)
    # occu_all = np.concatenate(occu_mask_list)
    # epe_noc = np.sum(epenoc_all) / occu_sum
    # px1_noc = np.sum((epe_all < 1) * occu_all) / occu_sum
    # px3_noc = np.sum((epe_all < 3) * occu_all) / occu_sum
    # px5_noc = np.sum((epe_all < 5) * occu_all) / occu_sum
    # print("Validation (%s) EPE_NOC: %f, 1px: %f, 3px: %f, 5px: %f" % (dstype, epe_noc, px1_noc, px3_noc, px5_noc))

    return results

@torch.no_grad()
def validate_solardis(model, iters=2):  # 32
    """ Peform validation using the Sintel (train) split """
    model.eval()
    results = {}
    dstype1 = 'img1'
    dstype2 = 'img2'

    for split in ['test']:  # ['train', 'test']
        val_dataset = datasets.SolarDis(split=split, dstype1=dstype1, dstype2=dstype2, istest='test')
        epe_list = []
        for val_id in range(len(val_dataset)):
            image1, image2, flow_gt = val_dataset[val_id]
            image1 = image1[None].cuda()
            image2 = image2[None].cuda()

            padder = InputPadder(image1.shape)
            image1, image2 = padder.pad(image1, image2)

            flow_low, flow_pr, occu_mask = model(image1, image2, iters=iters, test_mode=True, add_bw=True)
            flow = padder.unpad(flow_pr[0]).cpu()

            epe = torch.sum((flow - flow_gt)**2, dim=0).sqrt()
            epe_list.append(epe.view(-1).numpy())

        epe_all = np.concatenate(epe_list)
        epe = np.mean(epe_all)
        px1 = np.mean(epe_all<1)
        px3 = np.mean(epe_all<3)
        px5 = np.mean(epe_all<5)

        print("Validation (%s) EPE: %f, 1px: %f, 3px: %f, 5px: %f" % (split, epe, px1, px3, px5))
        results[split] = np.mean(epe_list)

    return results


@torch.no_grad()
def validate_solarno(model, iters=2):  # 32
    """ Peform validation using the Sintel (train) split """
    model.eval()
    results = {}
    dstype1 = 'img1'
    dstype2 = 'img2'

    for split in ['train', 'test']:
        val_dataset = datasets.SolarNo(split=split, dstype1=dstype1, dstype2=dstype2, istest='test')
        flow_u = 0
        flow_v = 0
        epe_list = []
        for val_id in range(len(val_dataset)):
            image1, image2, flow_gt = val_dataset[val_id]
            image1 = image1[None].cuda()
            image2 = image2[None].cuda()

            padder = InputPadder(image1.shape)
            image1, image2 = padder.pad(image1, image2)

            flow_low, flow_pr, occu_mask = model(image1, image2, iters=iters, test_mode=True, add_bw=True)
            flow = padder.unpad(flow_pr[0]).cpu()

            epe = torch.sum((flow - flow_gt)**2, dim=0).sqrt()
            epe_list.append(epe.view(-1).numpy())

            flow_u += torch.mean(flow[0])
            flow_v += torch.mean(flow[1])


        epe_all = np.concatenate(epe_list)
        epe = np.mean(epe_all)
        px1 = np.mean(epe_all<1)
        px3 = np.mean(epe_all<3)
        px5 = np.mean(epe_all<5)

        print("Validation (%s) EPE: %f, 1px: %f, 3px: %f, 5px: %f" % (split, epe, px1, px3, px5))
        results[split] = np.mean(epe_list)
        print("Validation (%s) u_px: %f, v_px: %f" % (split, flow_u/len(val_dataset), flow_v/len(val_dataset)))

    return results


@torch.no_grad()
def validate_solarzero(model, iters=2):  # 32
    """ Peform validation using the Sintel (train) split """
    model.eval()
    results = {}
    dstype1 = 'img1'
    dstype2 = 'img2'
    total_ssim, total_rm, total_rv = 0, 0, 0
    for split in ['test']:  # 'train', 'test'
        val_dataset = datasets.SolarZero(split=split, dstype1=dstype1, dstype2=dstype2, istest='test')
        flow_u = 0
        flow_v = 0
        epe_list = []
        for val_id in range(len(val_dataset)):
            image1, image2, flow_gt = val_dataset[val_id]
            image1 = image1[None].cuda()
            image2 = image2[None].cuda()

            padder = InputPadder(image1.shape)
            image1, image2 = padder.pad(image1, image2)

            flow_low, flow_pr, occu_mask = model(image1, image2, iters=iters, test_mode=True, add_bw=True)
            # metric test
            warp_img = flow_warp(image1, flow_pr)
            res_variance, res_mean = rv_rm(warp_img, image2, 10)
            res_ssim = ssim(warp_img, image2, 10)
            total_ssim += res_ssim
            total_rm += res_mean
            total_rv += res_variance
            # metric test
            flow = padder.unpad(flow_pr[0]).cpu()

            epe = torch.sum((flow - flow_gt)**2, dim=0).sqrt()
            epe = epe[10:-10, 10:-10].contiguous()
            epe_list.append(epe.view(-1).numpy())

            flow_u += torch.mean(flow[0, 10:-10, 10:-10])
            flow_v += torch.mean(flow[1, 10:-10, 10:-10])


        epe_all = np.concatenate(epe_list)
        epe = np.mean(epe_all)
        px1 = np.mean(epe_all<1)
        px3 = np.mean(epe_all<3)
        px5 = np.mean(epe_all<5)
        total_ssim = total_ssim / len(val_dataset)
        total_rm = total_rm / len(val_dataset)
        total_rv = total_rv / len(val_dataset)
        print("Validation (%s) EPE: %f, 1px: %f, 3px: %f, 5px: %f" % (split, epe, px1, px3, px5))
        results[split] = np.mean(epe_list)
        print("Validation (%s) u_px: %f, v_px: %f" % (split, flow_u/len(val_dataset), flow_v/len(val_dataset)))
        print("Validation (%s) SSIM: %f, RM: %f, RV: %f" % (split, total_ssim, total_rm, total_rv))

    return results


@torch.no_grad()
def validate_optical(model, iters=2):  # 32
    """ Peform validation using the Sintel (train) split """
    model.eval()
    results = {}
    dstype1 = 'img1'
    dstype2 = 'img2'
    total_ssim, total_rm, total_rv = 0, 0, 0
    for split in ['test']:  # 'train', 'test'
        val_dataset = datasets.Optical(split=split, dstype1=dstype1, dstype2=dstype2, istest='test')
        flow_u = 0
        flow_v = 0
        epe_list = []
        for val_id in range(len(val_dataset)):
            image1, image2, flow_gt = val_dataset[val_id]
            image1 = image1[None].cuda()
            image2 = image2[None].cuda()

            padder = InputPadder(image1.shape)
            image1, image2 = padder.pad(image1, image2)

            flow_low, flow_pr, occu_mask = model(image1, image2, iters=iters, test_mode=True, add_bw=True)
            # metric test
            warp_img = flow_warp(image1, flow_pr)
            res_variance, res_mean = rv_rm(warp_img, image2, 10)
            res_ssim = ssim(warp_img, image2, 10)
            total_ssim += res_ssim
            total_rm += res_mean
            total_rv += res_variance
            # metric test
            flow = padder.unpad(flow_pr[0]).cpu()

            epe = torch.sum((flow - flow_gt)**2, dim=0).sqrt()
            epe = epe[10:-10, 10:-10].contiguous()
            epe_list.append(epe.view(-1).numpy())

            # flow_u += torch.mean(flow[0, 10:-10, 10:-10])
            # flow_v += torch.mean(flow[1, 10:-10, 10:-10])


        epe_all = np.concatenate(epe_list)
        epe = np.mean(epe_all)
        px1 = np.mean(epe_all<1)
        px3 = np.mean(epe_all<3)
        px5 = np.mean(epe_all<5)
        total_ssim = total_ssim / len(val_dataset)
        total_rm = total_rm / len(val_dataset)
        total_rv = total_rv / len(val_dataset)
        print("Validation (%s) EPE: %f, 1px: %f, 3px: %f, 5px: %f" % (split, epe, px1, px3, px5))
        results[split] = np.mean(epe_list)
        # print("Validation (%s) u_px: %f, v_px: %f" % (split, flow_u/len(val_dataset), flow_v/len(val_dataset)))
        print("Validation (%s) SSIM: %f, RM: %f, RV: %f" % (split, total_ssim, total_rm, total_rv))


    return results


@torch.no_grad()
def validate_chairs(model, iters=24):
    """ Perform evaluation on the FlyingChairs (test) split """
    model.eval()
    epe_list = []

    val_dataset = datasets.FlyingChairs(split='validation')
    for val_id in range(len(val_dataset)):
        image1, image2, flow_gt, _ = val_dataset[val_id]
        image1 = image1[None].cuda()
        image2 = image2[None].cuda()

        _, flow_pr = model(image1, image2, iters=iters, test_mode=True)
        epe = torch.sum((flow_pr[0].cpu() - flow_gt)**2, dim=0).sqrt()
        epe_list.append(epe.view(-1).numpy())

    epe = np.mean(np.concatenate(epe_list))
    print("Validation Chairs EPE: %f" % epe)
    return {'chairs': epe}


@torch.no_grad()
def validate_sintel(model, iters=32):
    """ Peform validation using the Sintel (train) split """
    model.eval()
    results = {}

    for dstype in ['clean', 'final']:
        val_dataset = datasets.MpiSintel(split='training', dstype=dstype)
        epe_list = []

        for val_id in range(len(val_dataset)):
            # image1, image2, flow_gt, _ = val_dataset[val_id]
            image1, image2, flow_gt, _, _, _ = val_dataset[val_id]
            image1 = image1[None].cuda()
            image2 = image2[None].cuda()

            padder = InputPadder(image1.shape)
            image1, image2 = padder.pad(image1, image2)

            # flow_low, flow_pr = model(image1, image2, iters=iters, test_mode=True)
            flow_low, flow_pr, occu_mask = model(image1, image2, iters=iters, test_mode=True, add_bw=True)
            flow = padder.unpad(flow_pr[0]).cpu()

            epe = torch.sum((flow - flow_gt)**2, dim=0).sqrt()
            epe_list.append(epe.view(-1).numpy())

        epe_all = np.concatenate(epe_list)
        epe = np.mean(epe_all)
        px1 = np.mean(epe_all<1)
        px3 = np.mean(epe_all<3)
        px5 = np.mean(epe_all<5)

        print("Validation (%s) EPE: %f, 1px: %f, 3px: %f, 5px: %f" % (dstype, epe, px1, px3, px5))
        results[dstype] = np.mean(epe_list)

    return results


@torch.no_grad()
def validate_kitti(model, iters=24):
    """ Peform validation using the KITTI-2015 (train) split """
    model.eval()
    val_dataset = datasets.KITTI(split='training')

    out_list, epe_list = [], []
    for val_id in range(len(val_dataset)):
        image1, image2, flow_gt, valid_gt = val_dataset[val_id]
        image1 = image1[None].cuda()
        image2 = image2[None].cuda()

        padder = InputPadder(image1.shape, mode='kitti')
        image1, image2 = padder.pad(image1, image2)

        flow_low, flow_pr = model(image1, image2, iters=iters, test_mode=True)
        flow = padder.unpad(flow_pr[0]).cpu()

        epe = torch.sum((flow - flow_gt)**2, dim=0).sqrt()
        mag = torch.sum(flow_gt**2, dim=0).sqrt()

        epe = epe.view(-1)
        mag = mag.view(-1)
        val = valid_gt.view(-1) >= 0.5

        out = ((epe > 3.0) & ((epe/mag) > 0.05)).float()
        epe_list.append(epe[val].mean().item())
        out_list.append(out[val].cpu().numpy())

    epe_list = np.array(epe_list)
    out_list = np.concatenate(out_list)

    epe = np.mean(epe_list)
    f1 = 100 * np.mean(out_list)

    print("Validation KITTI: %f, %f" % (epe, f1))
    return {'kitti-epe': epe, 'kitti-f1': f1}


def eva(args):
    # run code : --model=models/raft-solar.pth --dataset=solar
    model = torch.nn.DataParallel(PWCNETUL(args))  # pwcnet
    model.load_state_dict(torch.load(args.model))

    model.cuda()
    model.eval()
    return model


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', help="restore checkpoint")
    parser.add_argument('--dataset', help="dataset for evaluation")
    parser.add_argument('--small', action='store_true', help='use small model')
    parser.add_argument('--mixed_precision', action='store_true', help='use mixed precision')
    parser.add_argument('--alternate_corr', action='store_true', help='use efficent correlation implementation')
    args = parser.parse_args()

    # model = torch.nn.DataParallel(FRAFT(args))
    # model.load_state_dict(torch.load(args.model))
    #
    # model.cuda()
    # model.eval()
    model = eva(args)
    # create_sintel_submission(model.module, warm_start=True)
    # create_kitti_submission(model.module)

    with torch.no_grad():
        if args.dataset == 'chairs':
            validate_chairs(model.module)

        elif args.dataset == 'sintel':
            validate_sintel(model.module)

        elif args.dataset == 'kitti':
            validate_kitti(model.module)

        elif args.dataset == 'solar':
            validate_solar(model.module)
            validate_solarzero(model.module)
            validate_optical(model.module)

        elif args.dataset == 'solardis':
            validate_solardis(model.module)

        elif args.dataset == 'solarno':
            validate_solarno(model.module)

        elif args.dataset == 'solarzero':
            validate_solarzero(model.module)

        elif args.dataset == 'optical':
            validate_optical(model.module)


