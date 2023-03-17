
import torch
import torch.nn as nn
import torch.nn.functional as F
from core.unet.ablation_experments.extra_loss.res_bottle import ResidualBlock, Block
from core.utils.add_module import SELayer
# from spatial_correlation_sampler import spatial_correlation_sample
from core.pwc_net.correlation import FunctionCorrelation
from core.utils.warp_occ_utils import flow_warp


class ShareEncoder(nn.Module):
    def __init__(self, output_dim=256, norm_fn='batch', dropout=0.0):
        super(ShareEncoder, self).__init__()
        self.norm_fn = norm_fn

        if self.norm_fn == 'group':
            self.norm1_1 = nn.GroupNorm(num_groups=8, num_channels=64)
            self.norm1_2 = nn.GroupNorm(num_groups=8, num_channels=64)
            self.norm_conv2 = nn.GroupNorm(num_groups=8, num_channels=96)
            self.norm_conv3 = nn.GroupNorm(num_groups=8, num_channels=128)
            self.norm_conv4 = nn.GroupNorm(num_groups=8, num_channels=256)

        elif self.norm_fn == 'batch':
            self.norm1_1 = nn.BatchNorm2d(64)
            self.norm1_2 = nn.BatchNorm2d(64)
            self.norm_conv2 = nn.BatchNorm2d(96)
            self.norm_conv3 = nn.BatchNorm2d(128)
            self.norm_conv4 = nn.BatchNorm2d(256)

        elif self.norm_fn == 'instance':
            self.norm1_1 = nn.InstanceNorm2d(64)
            self.norm1_2 = nn.InstanceNorm2d(64)
            self.norm_conv2 = nn.InstanceNorm2d(96)
            self.norm_conv3 = nn.InstanceNorm2d(128)
            self.norm_conv4 = nn.InstanceNorm2d(256)

        elif self.norm_fn == 'none':
            self.norm1_1 = nn.Sequential()
            self.norm1_1 = nn.Sequential()
            self.norm_conv2 = nn.Sequential()
            self.norm_conv3 = nn.Sequential()
            self.norm_conv4 = nn.Sequential()

        self.relu = nn.ReLU(inplace=True)
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1), self.norm1_1, self.relu,   # 22-09-29 暂时把kernel_size=3改为7
            # nn.Conv2d(64, 64, kernel_size=3, padding=1), self.norm1_2, self.relu,
            self._make_layer(64, 64, stride=1),
            self._se_block(64)
        )
        self.conv2 = nn.Sequential(
            nn.Sequential(nn.Conv2d(64, 96, kernel_size=2, stride=2), self.norm_conv2, self.relu),
            self._make_layer(96, 96, stride=1),
            self._make_layer(96, 96, stride=1),
            self._se_block(96)
        )
        self.conv3 = nn.Sequential(
            nn.Sequential(nn.Conv2d(96, 128, kernel_size=2, stride=2), self.norm_conv3, self.relu),
            self._make_layer(128, 128, stride=1),
            self._make_layer(128, 128, stride=1),
            self._se_block(128)
        )
        self.conv4 = nn.Sequential(
            nn.Conv2d(128, output_dim, kernel_size=2, stride=2), self.norm_conv4, self.relu,
            self._make_layer(output_dim, output_dim, stride=1),  #
            self._se_block(output_dim)
        )

        self.dropout = None
        if dropout > 0:
            self.dropout = nn.Dropout2d(p=dropout)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.InstanceNorm2d, nn.GroupNorm)):
                if m.weight is not None:
                    nn.init.constant_(m.weight, 1)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def _make_layer(self, input_dim, output_dim, stride=1):
        # layer = ResidualBlock(input_dim, output_dim, self.norm_fn, stride=stride)
        layer = Block(output_dim)
        return nn.Sequential(layer)

    def _se_block(self, output_dim):
        se_layer = SELayer(output_dim)
        return nn.Sequential(se_layer)

    def forward(self, x):
        # if input is list, combine batch dimension
        is_list = isinstance(x, tuple) or isinstance(x, list)
        if is_list:
            batch_dim = x[0].shape[0]
            x = torch.cat(x, dim=0)

        x1_pyramid = []
        x2_pyramid = []

        layer1 = self.conv1(x)
        layer2 = self.conv2(layer1)
        layer3 = self.conv3(layer2)
        layer4 = self.conv4(layer3)

        if self.training and self.dropout is not None:
            layer4 = self.dropout(layer4)

        layers = (layer1, layer2, layer3, layer4)

        if is_list:
            for i in range(4):
                layer = torch.split(layers[i], [batch_dim, batch_dim], dim=0)
                x1_pyramid.append(layer[0])
                x2_pyramid.append(layer[1])
            return x1_pyramid, x2_pyramid
        else:
            for i in range(4):
                x1_pyramid.append(layers[i])
            return x1_pyramid, None


class FRFlowDecoder(nn.Module):
    def __init__(self, output_dim=2, dropout=0.0):
        super(FRFlowDecoder, self).__init__()
        self.relu = nn.ReLU(inplace=True)

        self.conv3 = nn.Sequential(nn.Conv2d(81 + 256, 256, kernel_size=1), self.relu,)
        self.conv2 = nn.Sequential(nn.Conv2d(81 + 128 + 2, 128, kernel_size=1), self.relu,)
        self.conv1 = nn.Sequential(nn.Conv2d(81 + 96 + 2, 96, kernel_size=1), self.relu,)
        self.conv0 = nn.Sequential(nn.Conv2d(81 + 64 + 2, 64, kernel_size=1), self.relu,)

        self.estimate = nn.ModuleList([
            nn.Sequential(nn.Conv2d(64 + 64, 64, kernel_size=3, padding=1), self.relu),
            nn.Sequential(nn.Conv2d(96 + 96, 96, kernel_size=3, padding=1), self.relu),
            nn.Sequential(nn.Conv2d(128 + 128, 128, kernel_size=3, padding=1), self.relu),
            nn.Sequential(nn.Conv2d(256 + 256, 256, kernel_size=3, padding=1), self.relu)
        ])

        self.multi_1 = nn.ModuleList([
            nn.Sequential(nn.Conv2d(64, 64, kernel_size=3, padding=1), self.relu,
                          nn.Conv2d(64, 32, kernel_size=1), self.relu,
                          nn.Conv2d(32, 2, kernel_size=3, padding=1)),
            nn.Sequential(nn.Conv2d(96, 96, kernel_size=3, padding=1), self.relu,
                          nn.Conv2d(96, 48, kernel_size=1), self.relu,
                          nn.Conv2d(48, 2, kernel_size=3, padding=1)),
            nn.Sequential(nn.Conv2d(128, 128, kernel_size=3, padding=1), self.relu,
                          nn.Conv2d(128, 64, kernel_size=1), self.relu,
                          nn.Conv2d(64, 2, kernel_size=3, padding=1)),
            nn.Sequential(nn.Conv2d(256, 256, kernel_size=3, padding=1), self.relu,
                          nn.Conv2d(256, 128, kernel_size=1), self.relu,
                          nn.Conv2d(128, 2, kernel_size=3, padding=1)),
        ])
        self.multi_2 = nn.ModuleList([
            nn.Sequential(nn.Conv2d(64, 64, kernel_size=3, padding=2, dilation=2), self.relu,
                          nn.Conv2d(64, 32, kernel_size=1), self.relu,
                          nn.Conv2d(32, 2, kernel_size=3, padding=1)),
            nn.Sequential(nn.Conv2d(96, 96, kernel_size=3, padding=2, dilation=2), self.relu,
                          nn.Conv2d(96, 48, kernel_size=1), self.relu,
                          nn.Conv2d(48, 2, kernel_size=3, padding=1)),
            nn.Sequential(nn.Conv2d(128, 128, kernel_size=3, padding=2, dilation=2), self.relu,
                          nn.Conv2d(128, 64, kernel_size=1), self.relu,
                          nn.Conv2d(64, 2, kernel_size=3, padding=1)),
            nn.Sequential(nn.Conv2d(256, 256, kernel_size=3, padding=2, dilation=2), self.relu,
                          nn.Conv2d(256, 128, kernel_size=1), self.relu,
                          nn.Conv2d(128, 2, kernel_size=3, padding=1)),
        ])

        self.dropout = None
        if dropout > 0:
            self.dropout = nn.Dropout2d(p=dropout)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.InstanceNorm2d, nn.GroupNorm)):
                if m.weight is not None:
                    nn.init.constant_(m.weight, 1)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def _se_block(self, output_dim):
        se_layer = SELayer(output_dim)
        return nn.Sequential(se_layer)

    def forward(self, f1_pyramid, f2_pyramid, cnet_pyramid):
        corr3 = FunctionCorrelation(f1_pyramid[3], f2_pyramid[3])
        corr3 = self.relu(corr3)
        x = torch.cat([corr3, f2_pyramid[3]], dim=1)
        x = self.conv3(x)  # 81 + 256
        x = self.estimate[3](torch.cat([x, cnet_pyramid[3]], dim=1))
        multi1_3 = self.multi_1[3](x)
        multi2_3 = self.multi_2[3](x)
        output_3 = 0.8 * multi1_3 + 0.2 * multi2_3

        init_flow2 = F.interpolate(output_3 * 2.0, scale_factor=2, mode='bilinear', align_corners=True)
        warp_f1_2 = flow_warp(f1_pyramid[2], init_flow2)
        corr2 = FunctionCorrelation(warp_f1_2, f2_pyramid[2])
        x = torch.cat([corr2, f2_pyramid[2], init_flow2], dim=1)
        x = self.conv2(x)  # 81 + 128 + 2
        x = self.estimate[2](torch.cat([x, cnet_pyramid[2]], dim=1))
        multi1_2 = self.multi_1[2](x)
        multi2_2 = self.multi_2[2](x)
        output_2 = 0.8 * multi1_2 + 0.2 * multi2_2 + init_flow2

        init_flow1 = F.interpolate(output_2 * 2.0, scale_factor=2, mode='bilinear', align_corners=True)
        warp_f1_1 = flow_warp(f1_pyramid[1], init_flow1)
        corr1 = FunctionCorrelation(warp_f1_1, f2_pyramid[1])
        x = torch.cat([corr1, f2_pyramid[1], init_flow1], dim=1)
        x = self.conv1(x)  # 81 + 96 + 2
        x = self.estimate[1](torch.cat([x, cnet_pyramid[1]], dim=1))
        multi1_1 = self.multi_1[1](x)
        multi2_1 = self.multi_2[1](x)
        output_1 = 0.8 * multi1_1 + 0.2 * multi2_1 + init_flow1

        init_flow0 = F.interpolate(output_1 * 2.0, scale_factor=2, mode='bilinear', align_corners=True)
        warp_f1_0 = flow_warp(f1_pyramid[0], init_flow0)
        corr0 = FunctionCorrelation(warp_f1_0, f2_pyramid[0])
        x = torch.cat([corr0, f2_pyramid[0], init_flow0], dim=1)
        x = self.conv0(x)  # 81 + 64 + 2
        x = self.estimate[0](torch.cat([x, cnet_pyramid[0]], dim=1))
        multi1_0 = self.multi_1[0](x)
        multi2_0 = self.multi_2[0](x)
        output_0 = 0.8 * multi1_0 + 0.2 * multi2_0 + init_flow0

        # if self.training and self.dropout is not None:
        #     x = self.dropout(x)

        return [output_3, output_2, output_1, output_0]
