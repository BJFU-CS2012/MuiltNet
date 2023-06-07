import torch
from torch import nn
import torchvision
class h_sigmoid(nn.Module):
    def __init__(self, inplace=True):
        super(h_sigmoid, self).__init__()
        self.relu = nn.ReLU6(inplace=inplace)

    def forward(self, x):
        return self.relu(x + 3) / 6


class h_swish(nn.Module):
    def __init__(self, inplace=True):
        super(h_swish, self).__init__()
        self.sigmoid = h_sigmoid(inplace=inplace)

    def forward(self, x):
        return x * self.sigmoid(x)


class CoordAttention(nn.Module):

    def __init__(self, in_channels, out_channels, reduction=32):
        super(CoordAttention, self).__init__()
        self.pool_w, self.pool_h = nn.AdaptiveAvgPool2d((1, None)), nn.AdaptiveAvgPool2d((None, 1))
        temp_c = max(8, in_channels // reduction)
        self.conv1 = nn.Conv2d(in_channels, temp_c, kernel_size=1, stride=1, padding=0)

        self.bn1 = nn.BatchNorm2d(temp_c)
        self.act1 = h_swish()

        self.conv2 = nn.Conv2d(temp_c, out_channels, kernel_size=1, stride=1, padding=0)
        self.conv3 = nn.Conv2d(temp_c, out_channels, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        short = x
        n, c, H, W = x.shape
        x_h, x_w = self.pool_h(x), self.pool_w(x).permute(0, 1, 3, 2)
        x_cat = torch.cat([x_h, x_w], dim=2)
        out = self.act1(self.bn1(self.conv1(x_cat)))
        x_h, x_w = torch.split(out, [H, W], dim=2)
        x_w = x_w.permute(0, 1, 3, 2)
        out_h = torch.sigmoid(self.conv2(x_h))
        out_w = torch.sigmoid(self.conv3(x_w))
        return short * out_w * out_h

# 残差层数 50和101
class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_channel, out_channel, stride=1, downsample=None,
                 groups=1, width_per_group=64):
        super(Bottleneck, self).__init__()

        width = int(out_channel * (width_per_group / 64.)) * groups

        self.conv1 = nn.Conv2d(in_channels=in_channel, out_channels=width,
                               kernel_size=1, stride=1, bias=False)  # squeeze channels
        self.bn1 = nn.BatchNorm2d(width)
        # -----------------------------------------
        self.conv2 = nn.Conv2d(in_channels=width, out_channels=width, groups=groups,
                               kernel_size=3, stride=stride, bias=False, padding=1)
        self.bn2 = nn.BatchNorm2d(width)
        # -----------------------------------------
        self.conv3 = nn.Conv2d(in_channels=width, out_channels=out_channel * self.expansion,
                               kernel_size=1, stride=1, bias=False)  # unsqueeze channels
        self.bn3 = nn.BatchNorm2d(out_channel * self.expansion)
        self.relu = nn.ReLU(inplace=True)

        self.downsample = downsample

        self.stride = stride
        self.ca = CoordAttention(in_channels=out_channel * self.expansion, out_channels=out_channel * self.expansion)

    def forward(self, x):
        identity = x
        if self.downsample is not None:
            identity = self.downsample(x)

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        out = self.ca(out)  # add CA

        out += identity
        out = self.relu(out)

        return out


class CANet(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        bits = self.config.code_length
        classlen = self.config.classlen

        if self.config.model_name == 'resnet50':
            self.backbone = torchvision.models.resnet50(pretrained=True)
            self.backbone.avgpool = nn.AdaptiveAvgPool2d(output_size=(1, 1))
            # Adaptive hyper
            self.alpha1 = nn.Parameter(torch.ones(1) * 1, requires_grad=True)
            self.alpha2 = nn.Parameter(torch.ones(1) * 1, requires_grad=True)

            self.backbone.b1 = nn.Linear(1024, bits)
            self.backbone.b2 = nn.Linear(1024, bits)
            self.backbone.b3 = nn.Linear(1024, bits)
            self.backbone.b_cat = nn.Linear(3072, bits)

            self.backbone.fc = nn.Linear(bits, classlen)

            self.relu = nn.ReLU(inplace=True)
            self.num_ftrs = 2048
            self.feature_size = 512
            self.backbone.fc_x = nn.Linear(self.num_ftrs, classlen)

            self.backbone.layer1 = self._make_layer(Bottleneck, 64, 3)
            self.backbone.layer2 = self._make_layer(Bottleneck, 128, 4, stride=2)
            self.backbone.layer3 = self._make_layer(Bottleneck, 256, 6, stride=2)
            self.backbone.layer4 = self._make_layer(Bottleneck, 512, 3, stride=2)
            # stage 1
            self.backbone.conv_block1 = nn.Sequential(
                BasicConv(self.num_ftrs // 4, self.feature_size, kernel_size=1, stride=1, padding=0, relu=True),
                BasicConv(self.feature_size, self.num_ftrs // 2, kernel_size=3, stride=1, padding=1, relu=True),
                nn.AdaptiveMaxPool2d(1),
            )

            # stage 2
            self.backbone.conv_block2 = nn.Sequential(
                BasicConv(self.num_ftrs // 2, self.feature_size, kernel_size=1, stride=1, padding=0, relu=True),
                BasicConv(self.feature_size, self.num_ftrs // 2, kernel_size=3, stride=1, padding=1, relu=True),
                nn.AdaptiveMaxPool2d(1),
            )

            # stage 3
            self.backbone.conv_block3 = nn.Sequential(
                BasicConv(self.num_ftrs, self.feature_size, kernel_size=1, stride=1, padding=0, relu=True),
                BasicConv(self.feature_size, self.num_ftrs // 2, kernel_size=3, stride=1, padding=1, relu=True),
                nn.AdaptiveMaxPool2d(1),
            )

            # concat features from different stages
            self.backbone.hashing_concat = nn.Sequential(
                nn.BatchNorm1d(self.num_ftrs // 2 * 3, affine=True),
                nn.Linear(self.num_ftrs // 2 * 3, self.feature_size),
                nn.BatchNorm1d(self.feature_size, affine=True),
                nn.ELU(inplace=True),
                nn.Linear(self.feature_size, bits),
            )

    def _make_layer(self, block, channel, block_num, stride=1):
        downsample = None
        if stride != 1 or self.in_channel != channel * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.in_channel, channel * block.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(channel * block.expansion))

        layers = []
        layers.append(block(self.in_channel,
                            channel,
                            downsample=downsample,
                            stride=stride,
                            groups=self.groups,
                            width_per_group=self.width_per_group))
        self.in_channel = channel * block.expansion

        for _ in range(1, block_num):
            layers.append(block(self.in_channel,
                                channel,
                                groups=self.groups,
                                width_per_group=self.width_per_group))

        return nn.Sequential(*layers)
    def forward(self, x):
        return self.forward_vanilla(x)

    def forward_vanilla(self, x):
        x = self.backbone.conv1(x)
        x = self.backbone.bn1(x)
        x = self.backbone.relu(x)
        x1 = self.backbone.maxpool(x)
        x2 = self.backbone.layer1(x1)
        f1 = self.backbone.layer2(x2)
        f2 = self.backbone.layer3(f1)
        f3 = self.backbone.layer4(f2)

        feats = f3

        f11 = self.backbone.conv_block1(f1).view(-1, self.num_ftrs // 2)
        f11_b = self.backbone.b1(f11)

        f22 = self.backbone.conv_block2(f2).view(-1, self.num_ftrs // 2)
        f22_b = self.backbone.b2(f22)

        f33 = self.backbone.conv_block3(f3).view(-1, self.num_ftrs // 2)
        f33_b = self.backbone.b3(f33)
        y33 = self.backbone.fc(f33_b)

        f44 = torch.cat((f11, f22, f33), -1)
        f44_b = self.backbone.hashing_concat(f44)

        # x = self.backbone.avgpool(feats)
        # x = torch.flatten(x, 1)
        # y_x = self.backbone.fc_x(x)

        return self.alpha1, self.alpha2, f44_b, y33, feats



# Model Setting
class BasicConv(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, dilation=1, groups=1, relu=True, bn=True, bias=False):
        super(BasicConv, self).__init__()
        self.out_channels = out_planes
        self.conv = nn.Conv2d(in_channels=in_planes, out_channels=out_planes, kernel_size=kernel_size,
                                stride=stride, padding=padding, dilation=dilation, groups=groups, bias=bias)
        self.bn = nn.BatchNorm2d(out_planes, momentum=0.01, affine=True) if bn else None
        self.relu = nn.ReLU() if relu else None

    def forward(self, x):
        x = self.conv(x)
        if self.bn is not None:
            x = self.bn(x)
        if self.relu is not None:
            x = self.relu(x)
        return x