import torch
import torch.nn as nn
import torch.nn.functional as F

def gradient(data):
    D_dy = data[:, :, 1:] - data[:, :, :-1]
    D_dx = data[:, :, :, 1:] - data[:, :, :, :-1]
    return D_dx, D_dy


def smooth_grad_1st(flo, image, alpha=10):
    img_dx, img_dy = gradient(image)
    weights_x = torch.exp(-torch.mean(torch.abs(img_dx), 1, keepdim=True) * alpha)
    weights_y = torch.exp(-torch.mean(torch.abs(img_dy), 1, keepdim=True) * alpha)

    dx, dy = gradient(flo)

    loss_x = weights_x * dx.abs() / 2.
    loss_y = weights_y * dy.abs() / 2.

    return loss_x.mean() / 2. + loss_y.mean() / 2.

def smooth_grad_2nd(flo, image, alpha=10):
    img_dx, img_dy = gradient(image)
    weights_x = torch.exp(-torch.mean(torch.abs(img_dx), 1, keepdim=True) * alpha)
    weights_y = torch.exp(-torch.mean(torch.abs(img_dy), 1, keepdim=True) * alpha)

    dx, dy = gradient(flo)
    dx2, dxdy = gradient(dx)
    dydx, dy2 = gradient(dy)

    loss_x = weights_x[:, :, :, 1:] * dx2.abs()
    loss_y = weights_y[:, :, 1:, :] * dy2.abs()

    return loss_x.mean() / 2. + loss_y.mean() / 2.

def loss_smooth(flo, image, alpha=10):
    loss_1 = smooth_grad_1st(flo, image, alpha)
    loss_2 = smooth_grad_2nd(flo, image, alpha)
    return loss_1, loss_2

def loss_photometric(image1, image2, occ=1):
    eps = 0.01
    dif = torch.mean(torch.abs(image1 - image2) * occ, 1, keepdim=True) + eps
    exponent = 0.4
    loss = torch.pow(dif, exponent)
    return torch.mean(loss)
    # loss = torch.mean(torch.abs(image1 - image2))
    # return loss

def loss_gradient(image1, image2, occ=1):
    exponent = 0.4
    eps = 0.01
    dx1, dy1 = gradient(image1)
    dx2, dy2 = gradient(image2)
    dif_x = torch.mean(torch.abs(dx1 - dx2) * occ, 1, keepdim=True) + eps
    dif_y = torch.mean(torch.abs(dy1 - dy2) * occ, 1, keepdim=True) + eps
    loss_x = torch.pow(dif_x, exponent)
    loss_y = torch.pow(dif_y, exponent)
    return torch.mean(loss_x) + torch.mean(loss_y)
    # dx1, dy1 = gradient(image1)
    # dx2, dy2 = gradient(image2)
    # loss_x = torch.mean(torch.abs(dx1 - dx2))
    # loss_y = torch.mean(torch.abs(dy1 - dy2))
    # return loss_x + loss_y


def loss_gradient_oc(image1, image2, occ=1):
    exponent = 0.4
    eps = 0.01
    dx1, dy1 = gradient(image1)
    dx2, dy2 = gradient(image2)
    occ_y = occ[:, :, :-1]
    occ_x = occ[:, :, :, :-1]
    dif_x = torch.mean(torch.abs(dx1 - dx2) * occ_x, 1, keepdim=True) + eps
    dif_y = torch.mean(torch.abs(dy1 - dy2) * occ_y, 1, keepdim=True) + eps
    loss_x = torch.pow(dif_x, exponent)
    loss_y = torch.pow(dif_y, exponent)
    return torch.mean(loss_x) + torch.mean(loss_y)
