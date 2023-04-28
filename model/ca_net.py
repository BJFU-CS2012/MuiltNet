import torch
from torch import nn
import torchvision
import numpy as np
# Bilinear Attention Pooling
EPSILON = 1e-6
class BAP(nn.Module):
    def __init__(self, pool='GAP'):
        super(BAP, self).__init__()
        assert pool in ['GAP', 'GMP']
        if pool == 'GAP':
            self.pool = None
        else:
            self.pool = nn.AdaptiveMaxPool2d(1)

    def forward(self, features, attentions):
        B, C, H, W = features.size()
        _, M, AH, AW = attentions.size()

        # match size
        if AH != H or AW != W:
            attentions = nn.functional.upsample_bilinear(attentions, size=(H, W))

        # feature_matrix: (B, M, C) -> (B, M * C)
        if self.pool is None:
            feature_matrix = (torch.einsum('imjk,injk->imn', (attentions, features)) / float(H * W)).view(B, -1)
        else:
            feature_matrix = []
            for i in range(M):
                AiF = self.pool(features * attentions[:, i:i + 1, ...]).view(B, -1)
                feature_matrix.append(AiF)
            feature_matrix = torch.cat(feature_matrix, dim=1)

        # sign-sqrt
        feature_matrix_raw = torch.sign(feature_matrix) * torch.sqrt(torch.abs(feature_matrix) + EPSILON)

        # l2 normalization along dimension M and C
        feature_matrix = nn.functional.normalize(feature_matrix_raw, dim=-1)

        if self.training:
            fake_att = torch.zeros_like(attentions).uniform_(0, 2)
        else:
            fake_att = torch.ones_like(attentions)
        counterfactual_feature = (torch.einsum('imjk,injk->imn', (fake_att, features)) / float(H * W)).view(B, -1)

        counterfactual_feature = torch.sign(counterfactual_feature) * torch.sqrt(torch.abs(counterfactual_feature) + EPSILON)

        counterfactual_feature = nn.functional.normalize(counterfactual_feature, dim=-1)
        return feature_matrix, counterfactual_feature

class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=8):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        # 利用1x1卷积代替全连接
        self.fc1   = nn.Conv2d(in_planes, in_planes // ratio, 1, bias=False)
        self.relu1 = nn.ReLU()
        self.fc2   = nn.Conv2d(in_planes // ratio, in_planes, 1, bias=False)

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc2(self.relu1(self.fc1(self.avg_pool(x))))
        max_out = self.fc2(self.relu1(self.fc1(self.max_pool(x))))
        out = avg_out + max_out
        return self.sigmoid(out)

class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()

        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        padding = 3 if kernel_size == 7 else 1
        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv1(x)
        return self.sigmoid(x)
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


class CANet_cal(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        bits = self.config.code_length
        classlen = self.config.classlen
        self.M = 32
        if self.config.model_name == 'resnet50':
            self.backbone = torchvision.models.resnet50(pretrained=True)
            # print('before\n', self.backbone)
            self.num_features = 512 * 4

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
        else:
            raise ValueError('Unsupported net: %s' % self.config.model_name)
        # Attention Maps
        self.attentions = BasicConv2d(self.num_features, self.M, kernel_size=1)
        # Bilinear Attention Pooling
        self.bap = BAP(pool='GAP')
        # Classification Layer
        self.fc = nn.Linear(self.M * self.num_features, classlen, bias=False)
    def forward(self, x):
        return self.forward_vanilla(x)

    def forward_vanilla(self, x):
        batch_size = x.size(0)
        x = self.backbone.conv1(x)
        x = self.backbone.bn1(x)
        x = self.backbone.relu(x)
        x1 = self.backbone.maxpool(x)
        x2 = self.backbone.layer1(x1)
        f1 = self.backbone.layer2(x2)
        f2 = self.backbone.layer3(f1)
        f3 = self.backbone.layer4(f2)
        #注意力图
        feats = f3
        attention_maps = self.attentions(feats)
        feature_matrix, feature_matrix_hat = self.bap(feats, attention_maps)

        p = self.fc(feature_matrix * 100.)
        # Generate Attention Map
        if self.training:
            # Randomly choose one of attention maps Ak
            attention_map = []
            for i in range(batch_size):
                attention_weights = torch.sqrt(attention_maps[i].sum(dim=(1, 2)).detach() + EPSILON)
                attention_weights = nn.functional.normalize(attention_weights, p=1, dim=0)
                k_index = np.random.choice(self.M, 2, p=attention_weights.cpu().numpy())
                attention_map.append(attention_maps[i, k_index, ...])
            attention_map = torch.stack(attention_map)  # (B, 2, H, W) - one for cropping, the other for dropping
        else:
            attention_map = torch.mean(attention_maps, dim=1, keepdim=True)  # (B, 1, H, W)



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

        return self.alpha1, self.alpha2, f44_b, y33, attention_map, p, p - self.fc(feature_matrix_hat * 100.),feature_matrix
class CANet_att(nn.Module):
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
            self.channelattention_f1 = ChannelAttention(512, ratio=8)
            self.channelattention_f2 = ChannelAttention(1024, ratio=8)
            self.channelattention_f3 = ChannelAttention(2048, ratio=8)
            self.spatialattention = SpatialAttention(kernel_size=7)
            # print('after\n',self.backbone)

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
        f3 = f3 * self.channelattention_f3(f3)
        f3 = f3 * self.spatialattention(f3)
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

class BasicConv2d(nn.Module):

    def __init__(self, in_channels, out_channels, **kwargs):
        super(BasicConv2d, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, bias=False, **kwargs)
        self.bn = nn.BatchNorm2d(out_channels, eps=0.001)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        return nn.functional.relu(x, inplace=True)