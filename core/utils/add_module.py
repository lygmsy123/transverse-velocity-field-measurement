import torch
import torch.nn as nn


class ResidualBlock(nn.Module):
    def __init__(self, in_planes, planes, norm_fn='batch', stride=1):
        super(ResidualBlock, self).__init__()

        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, padding=1, stride=stride)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, padding=1)
        self.relu = nn.ReLU(inplace=True)
        self.conv3 = nn.Conv2d(in_planes, planes, kernel_size=1, stride=stride)

        num_groups = planes // 8

        if norm_fn == 'group':
            self.norm1 = nn.GroupNorm(num_groups=num_groups, num_channels=planes)
            self.norm2 = nn.GroupNorm(num_groups=num_groups, num_channels=planes)
            self.norm4 = nn.GroupNorm(num_groups=num_groups, num_channels=planes)
            if not stride == 1:
                self.norm3 = nn.GroupNorm(num_groups=num_groups, num_channels=planes)

        elif norm_fn == 'batch':
            self.norm1 = nn.BatchNorm2d(planes)
            self.norm2 = nn.BatchNorm2d(planes)
            self.norm4 = nn.BatchNorm2d(planes)
            if not stride == 1:
                self.norm3 = nn.BatchNorm2d(planes)

        elif norm_fn == 'instance':
            self.norm1 = nn.InstanceNorm2d(planes)
            self.norm2 = nn.InstanceNorm2d(planes)
            self.norm4 = nn.InstanceNorm2d(planes)
            if not stride == 1:
                self.norm3 = nn.InstanceNorm2d(planes)

        elif norm_fn == 'none':
            self.norm1 = nn.Sequential()
            self.norm2 = nn.Sequential()
            self.norm4 = nn.Sequential()
            if not stride == 1:
                self.norm3 = nn.Sequential()

        if stride == 1:
            self.downsample = None

        else:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_planes, planes, kernel_size=1, stride=stride), self.norm3)

    def forward(self, x):
        y = x
        y = self.relu(self.norm1(self.conv1(y)))
        y = self.relu(self.norm2(self.conv2(y)))

        if self.downsample is not None:
            x = self.downsample(x)
        x = self.norm4(self.conv3(x))

        return self.relu(x+y)


class SELayer(nn.Module):
    def __init__(self, channel, reduction=16):
        super(SELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )
        # self.conv1 = nn.Sequential(
        #     nn.Conv2d(channel, 2, kernel_size=1, stride=1),
        #     nn.BatchNorm2d(2),
        #     nn.ReLU(inplace=True)
        # )

    def forward(self, x):
        is_list = isinstance(x, tuple) or isinstance(x, list)
        if is_list:
            # batch_dim = x[0].shape[0]
            x = torch.cat(x, dim=1)
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        x = x * y.expand_as(x)
        # x = self.conv1(x)
        return x


class FlowConv(nn.Module):
    def __init__(self):
        super(FlowConv, self).__init__()
        self.conv_y = nn.Conv2d(2, 1, kernel_size=1, stride=1)
        self.norm_y = nn.BatchNorm2d(1)
        self.relu_y = nn.ReLU(inplace=True)
        self.conv_x = nn.Conv2d(2, 1, kernel_size=1, stride=1)
        self.norm_x = nn.BatchNorm2d(1)
        self.relu_x = nn.ReLU(inplace=True)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.InstanceNorm2d, nn.GroupNorm)):
                if m.weight is not None:
                    nn.init.constant_(m.weight, 1)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, flow_1, flow_2):
        y1, y2 = flow_1[:, 0, :, :].unsqueeze(1), flow_2[:, 0, :, :].unsqueeze(1)
        x1, x2 = flow_1[:, 1, :, :].unsqueeze(1), flow_2[:, 1, :, :].unsqueeze(1)
        y = torch.cat([y1, y2], dim=1)
        x = torch.cat([x1, x2], dim=1)
        y = self.relu_y(self.norm_y(self.conv_y(y)))
        x = self.relu_x(self.norm_x(self.conv_x(x)))
        return torch.cat([y, x], dim=1)


class SmallFlow(nn.Module):
    def __init__(self, output_dim=2, norm_fn='batch', dropout=0.0):
        super(SmallFlow, self).__init__()
        self.norm_fn = norm_fn

        if self.norm_fn == 'batch':
            self.norm1 = nn.BatchNorm2d(64)
        else:
            self.norm1 = nn.Sequential()

        self.conv1 = nn.Conv2d(6, 64, kernel_size=3, stride=1, padding=1)
        self.relu1 = nn.ReLU(inplace=True)

        self.in_planes = 64
        self.layer1 = self._make_layer(64, stride=1)
        self.layer2 = self._make_layer(96, stride=1)
        self.layer3 = self._make_layer(128, stride=1)

        self.conv2 = nn.Conv2d(128, 32, kernel_size=1)
        # self.norm2 = nn.BatchNorm2d(64)
        # self.relu2 = nn.ReLU(inplace=True)
        # output convolution
        self.conv3 = nn.Conv2d(32, output_dim, kernel_size=3, padding=1)

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

    def _make_layer(self, dim, stride=1):
        layer1 = ResidualBlock(self.in_planes, dim, self.norm_fn, stride=stride)
        layer2 = ResidualBlock(dim, dim, self.norm_fn, stride=1)
        layers = (layer1, layer2)

        self.in_planes = dim
        return nn.Sequential(*layers)

    def forward(self, x):
        # if input is list, combine batch dimension
        is_list = isinstance(x, tuple) or isinstance(x, list)
        if is_list:
            # batch_dim = x[0].shape[0]
            x = torch.cat(x, dim=1)

        x = self.conv1(x)
        x = self.norm1(x)
        x = self.relu1(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)

        x = self.conv2(x)
        x = self.conv3(x)

        if self.training and self.dropout is not None:
            x = self.dropout(x)

        # if is_list:
        #     x = torch.split(x, [batch_dim, batch_dim], dim=0)

        return x


class SmallFlow02(nn.Module):
    def __init__(self, output_dim=2, norm_fn='batch', dropout=0.0):
        super(SmallFlow02, self).__init__()
        self.norm_fn = norm_fn

        if self.norm_fn == 'batch':
            self.norm1 = nn.BatchNorm2d(64)
        else:
            self.norm1 = nn.Sequential()

        self.conv1 = nn.Conv2d(6, 64, kernel_size=3, stride=1, padding=1)
        self.relu1 = nn.ReLU(inplace=True)

        self.in_planes = 64
        self.layer1 = self._make_layer(64, stride=1)
        self.layer2 = self._make_layer(128, stride=1)

        self.conv2 = nn.Conv2d(128, 64, kernel_size=1)
        self.conv2_5_1 = nn.Conv2d(64, 64, kernel_size=(5, 1), padding=(2, 0))
        self.conv2_1_5 = nn.Conv2d(64, 64, kernel_size=(1, 5), padding=(0, 2))
        self.norm2 = nn.BatchNorm2d(64)
        self.norm2_5_1 = nn.BatchNorm2d(64)
        self.norm2_1_5 = nn.BatchNorm2d(64)
        self.relu2 = nn.ReLU(inplace=True)
        # output convolution
        self.conv3_1_1 = nn.Conv2d(64, 32, kernel_size=1)
        self.conv3 = nn.Conv2d(32, output_dim, kernel_size=3, padding=1)
        self.norm3 = nn.BatchNorm2d(32)
        self.relu3 = nn.ReLU(inplace=True)

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

    def _make_layer(self, dim, stride=1):
        layer1 = ResidualBlock(self.in_planes, dim, self.norm_fn, stride=stride)
        layer2 = ResidualBlock(dim, dim, self.norm_fn, stride=1)
        layers = (layer1, layer2)

        self.in_planes = dim
        return nn.Sequential(*layers)

    def forward(self, x):
        # if input is list, combine batch dimension
        is_list = isinstance(x, tuple) or isinstance(x, list)
        if is_list:
            # batch_dim = x[0].shape[0]
            x = torch.cat(x, dim=1)

        x = self.conv1(x)
        x = self.norm1(x)
        x = self.relu1(x)

        x = self.layer1(x)
        x = self.layer2(x)

        x = self.relu2(self.norm2(self.conv2(x)))
        x = self.relu2(self.norm2_5_1(self.conv2_5_1(x)))
        x = self.relu2(self.norm2_1_5(self.conv2_1_5(x)))

        x = self.relu3(self.norm3(self.conv3_1_1(x)))
        x = self.conv3(x)

        if self.training and self.dropout is not None:
            x = self.dropout(x)

        # if is_list:
        #     x = torch.split(x, [batch_dim, batch_dim], dim=0)

        return x


class Dilation_conv(nn.Module):
    def __init__(self, input_dim, output_dim, b1_dim, b2_dim, b3_dim, is_conv1=False):
        super(Dilation_conv, self).__init__()
        self.is_conv1 = is_conv1

        self.branch_1 = nn.Conv2d(output_dim, b1_dim, kernel_size=3, stride=1, padding=1)
        self.branch_2 = nn.Conv2d(output_dim, b2_dim, kernel_size=3, stride=1, dilation=3, padding=3)
        self.branch_3 = nn.Conv2d(output_dim, b3_dim, kernel_size=3, stride=1, dilation=5, padding=5)
        self.conv_1_1 = nn.Conv2d(input_dim, output_dim, kernel_size=1, stride=1)
        # self.relu1 = nn.ReLU(inplace=True)
        # self.norm1 = nn.BatchNorm2d(32+16+16)

    def forward(self, x):
        if self.is_conv1:
            x = self.conv_1_1(x)
        x1 = self.branch_1(x)
        x2 = self.branch_2(x)
        x3 = self.branch_3(x)

        x = torch.cat([x1, x2, x3], dim=1)
        # x = self.relu1(self.norm1(x))
        return x


class SmallFlow03(nn.Module):
    def __init__(self, output_dim=2, norm_fn='batch', dropout=0.0):
        super(SmallFlow03, self).__init__()
        self.norm_fn = norm_fn

        if self.norm_fn == 'batch':
            self.norm1 = nn.BatchNorm2d(64)
        else:
            self.norm1 = nn.Sequential()

        self.conv1 = nn.Conv2d(6, 64, kernel_size=3, stride=1, padding=1)
        # self.conv1 = Dilation_conv(64, 64, 32, 16, 16, is_conv1=False)
        self.relu1 = nn.ReLU(inplace=True)

        self.in_planes = 64
        self.layer1 = self._make_layer(64, stride=1)
        self.layer2 = self._make_layer(128, stride=1)

        self.conv2 = Dilation_conv(128, 64, 32, 16, 16, is_conv1=True)
        self.norm2 = nn.BatchNorm2d(64)
        self.relu2 = nn.ReLU(inplace=True)
        # output convolution

        self.conv3 = Dilation_conv(64, 32, 16, 8, 8, is_conv1=True)
        self.norm3 = nn.BatchNorm2d(32)
        self.relu3 = nn.ReLU(inplace=True)

        self.conv4 = nn.Conv2d(32, output_dim, kernel_size=3, stride=1, padding=1)
        self.norm4 = nn.BatchNorm2d(output_dim)
        self.relu4 = nn.ReLU(inplace=True)

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

    def _make_layer(self, dim, stride=1):
        layer1 = ResidualBlock(self.in_planes, dim, self.norm_fn, stride=stride)
        layer2 = ResidualBlock(dim, dim, self.norm_fn, stride=1)
        layers = (layer1, layer2)

        self.in_planes = dim
        return nn.Sequential(*layers)


    def forward(self, x):
        # if input is list, combine batch dimension
        is_list = isinstance(x, tuple) or isinstance(x, list)
        if is_list:
            # batch_dim = x[0].shape[0]
            x = torch.cat(x, dim=1)

        # x = self.relu1(self.norm1(self.conv0(x)))
        x = self.relu1(self.norm1(self.conv1(x)))

        x = self.layer1(x)
        x = self.layer2(x)

        x = self.relu2(self.norm2(self.conv2(x)))

        x = self.relu3(self.norm3(self.conv3(x)))

        x = self.relu4(self.norm4(self.conv4(x)))

        if self.training and self.dropout is not None:
            x = self.dropout(x)

        # if is_list:
        #     x = torch.split(x, [batch_dim, batch_dim], dim=0)

        return x


# 对应模型19，记住raft的basicencoder的se模块需要加上
class SmallFlow04(nn.Module):
    def __init__(self, output_dim=2, norm_fn='batch', dropout=0.0):
        super(SmallFlow04, self).__init__()
        self.norm_fn = norm_fn

        if self.norm_fn == 'batch':
            self.norm1 = nn.BatchNorm2d(64)
        else:
            self.norm1 = nn.Sequential()

        self.conv1 = nn.Conv2d(6, 64, kernel_size=3, stride=1, padding=1)
        self.relu1 = nn.ReLU(inplace=True)

        self.in_planes = 64
        self.layer1 = self._make_layer(64, stride=1)
        self.layer2 = self._make_layer(96, stride=1)
        self.layer3 = self._make_layer(128, stride=1)

        self.conv2 = nn.Conv2d(128, 64, kernel_size=1)
        self.norm2 = nn.BatchNorm2d(64)
        self.relu2 = nn.ReLU(inplace=True)
        # output convolution
        self.conv3 = nn.Conv2d(64, output_dim, kernel_size=3, padding=1)
        # small_bw
        # self.norm3 = nn.BatchNorm2d(output_dim)

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

    def _make_layer(self, dim, stride=1):
        layer1 = ResidualBlock(self.in_planes, dim, self.norm_fn, stride=stride)
        layer2 = ResidualBlock(dim, dim, self.norm_fn, stride=1)
        layers = (layer1, layer2)

        self.in_planes = dim
        return nn.Sequential(*layers)

    def forward(self, x):
        # if input is list, combine batch dimension
        is_list = isinstance(x, tuple) or isinstance(x, list)
        if is_list:
            # batch_dim = x[0].shape[0]
            x = torch.cat(x, dim=1)

        x = self.conv1(x)
        x = self.norm1(x)
        x = self.relu1(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)

        x = self.relu2(self.norm2(self.conv2(x)))
        x = self.conv3(x)

        if self.training and self.dropout is not None:
            x = self.dropout(x)

        # if is_list:
        #     x = torch.split(x, [batch_dim, batch_dim], dim=0)

        return x