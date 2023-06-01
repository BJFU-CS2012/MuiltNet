import torch
from torch import nn
import torchvision
from utils.channel_exchange import NewExchange
import torch.nn.functional as F
# Bilinear Attention Pooling
EPSILON = 1e-6

class BottleneckFusion(nn.Module):
    expansion = 4

    def __init__(self, in_planes, planes, stride=1, bn_threshold=2e-2):
        super(BottleneckFusion, self).__init__()
        self.bn_threshold = bn_threshold

        # first layer -- T
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        # first layer -- F
        self.conv1_f = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
        self.bn1_f = nn.BatchNorm2d(planes)
        # second layer --T
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        # second layer --F
        self.conv2_f = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2_f = nn.BatchNorm2d(planes)
        # three layer -- T
        self.conv3 = nn.Conv2d(planes, self.expansion * planes, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(self.expansion * planes)
        # three layer -- F
        self.conv3_f = nn.Conv2d(planes, self.expansion * planes, kernel_size=1, bias=False)
        self.bn3_f = nn.BatchNorm2d(self.expansion * planes)
        # down sample -- T
        self.downsample = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion * planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion * planes)
            )

        # down sample -- F
        self.downsample_f = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.downsample_f = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion * planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion * planes)
            )

        # the list of bn.weNewExchangeight()
        self.t_f_bn_layer1 = [self.bn1, self.bn1_f]
        self.t_f_bn_layer2 = [self.bn2, self.bn2_f]
        self.t_f_bn_layer3 = [self.bn3, self.bn3_f]

        # exchange function
        self.exchange = NewExchange()

    def forward(self, x_t_f_list):
        out_t = F.relu(self.bn1(self.conv1(x_t_f_list[0])))
        out_f = F.relu(self.bn1_f(self.conv1_f(x_t_f_list[1])))

        out_t, out_f = self.bn2(self.conv2(out_t)), self.bn2_f(self.conv2_f(out_f))

        out_t, out_f = self.exchange([out_t, out_f], self.t_f_bn_layer2, self.bn_threshold)  # Local fusion

        out_t, out_f = F.relu(out_t), F.relu(out_f)

        out_t, out_f = self.bn3(self.conv3(out_t)), self.bn3_f(self.conv3_f(out_f))
        out_t += self.downsample(x_t_f_list[0])
        out_f += self.downsample_f(x_t_f_list[1])
        out_t, out_f = F.relu(out_t), F.relu(out_f)
        return [out_t, out_f]

class CANet(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        bits = self.config.code_length
        classlen = self.config.classlen

        if self.config.model_name == 'resnet50':
            self.backbone = torchvision.models.resnet50(pretrained=True)
            # print('before\n', self.backbone)
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
            # print('after\n',self.backbone)

            self.upsample = nn.UpsamplingNearest2d(scale_factor=2)
            self.backbone.layer1 = self._make_layer(BottleneckFusion, 64, 3, stride=1, bn_threshold=0.02)
            self.backbone.layer2 = self._make_layer(BottleneckFusion, 64, 4, stride=1, bn_threshold=0.02)
            self.backbone.layer3 = self._make_layer(BottleneckFusion, 64, 6, stride=1, bn_threshold=0.02)
            self.backbone.layer4 = self._make_layer(BottleneckFusion, 64, 3, stride=1, bn_threshold=0.02)
            # training parameters
            self.alpha_beta1 = nn.Parameter(torch.ones(2, requires_grad=True))
            self.register_parameter('alpha_beta1', self.alpha_beta1)
            self.alpha_beta2 = nn.Parameter(torch.ones(2, requires_grad=True))
            self.register_parameter('alpha_beta1', self.alpha_beta2)
            self.alpha_beta3 = nn.Parameter(torch.ones(2, requires_grad=True))
            self.register_parameter('alpha_beta1', self.alpha_beta3)
            self.alpha_beta4 = nn.Parameter(torch.ones(2, requires_grad=True))
            self.register_parameter('alpha_beta1', self.alpha_beta4)
    def forward(self, x):
        return self.forward_vanilla(x)

    def forward_vanilla(self, x):
        x_t, x_f = input
        x_f = self.upsample(x_f)

        x = self.backbone.conv1(x_t)
        x = self.backbone.bn1(x)
        x = self.backbone.relu(x)
        out_t = self.backbone.maxpool(x)

        out_t, out_f = self.backbone.layer1([out_t, x_f])
        alpha_beta1 = F.softmax(self.alpha_beta, dim=0)
        x1 = alpha_beta1[0] * out_t.detach() + alpha_beta1[1] * out_f.detach()

        out_t, out_f = self.backbone.layer2([out_t, out_f])
        alpha_beta2 = F.softmax(self.alpha_beta, dim=0)
        f1 = alpha_beta2[0] * out_t.detach() + alpha_beta2[1] * out_f.detach()

        out_t, out_f = self.backbone.layer3([out_t, out_f])
        alpha_beta3 = F.softmax(self.alpha_beta, dim=0)
        f2 = alpha_beta3[0] * out_t.detach() + alpha_beta3[1] * out_f.detach()

        out_t, out_f = self.backbone.layer4([out_t, out_f])
        alpha_beta4 = F.softmax(self.alpha_beta, dim=0)
        f3 = alpha_beta4[0] * out_t.detach() + alpha_beta4[1] * out_f.detach()

        feats = f3

        f11 = self.backbone.conv_block1(f1).view(-1, self.num_ftrs // 2)
        f11_c = self.backbone.conv_block1(f1)
        print('f11_c.shape', f11_c.shape)  # torch.Size([32, 1024, 1, 1])
        print('f11.shape',f11.shape) # torch.Size([32, 1024])
        f11_b = self.backbone.b1(f11)

        f22 = self.backbone.conv_block2(f2).view(-1, self.num_ftrs // 2)
        print('f22.shape', f22.shape) # torch.Size([32, 1024])

        f22_b = self.backbone.b2(f22)

        f33 = self.backbone.conv_block3(f3).view(-1, self.num_ftrs // 2)
        print('f33.shape', f33.shape) # torch.Size([32, 1024])
        f33_b = self.backbone.b3(f33)
        y33 = self.backbone.fc(f33_b)


        f44 = torch.cat((f11, f22, f33), -1)
        f44_b = self.backbone.hashing_concat(f44)

        # x = self.backbone.avgpool(feats)
        # x = torch.flatten(x, 1)
        # y_x = self.backbone.fc_x(x)

        return self.alpha1, self.alpha2, f44_b, y33, feats

    def _make_layer(self, block, planes, num_blocks, stride, bn_threshold):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride, bn_threshold))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)


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

class BasicConv2d(nn.Module):

    def __init__(self, in_channels, out_channels, **kwargs):
        super(BasicConv2d, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, bias=False, **kwargs)
        self.bn = nn.BatchNorm2d(out_channels, eps=0.001)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        return nn.functional.relu(x, inplace=True)