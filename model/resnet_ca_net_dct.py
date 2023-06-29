import torch

from torch import nn
import torchvision
from utils.dct_utils import PreNorm, FeedForward, Attention
from einops import rearrange
from torchvision import models

# Bilinear Attention Pooling
EPSILON = 1e-6
class eca_layer(nn.Module):
    """Constructs a ECA module.
    Args:
        channel: Number of channels of the input feature map
        k_size: Adaptive selection of kernel size
    """

    def __init__(self, channel, k_size=3):
        super(eca_layer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Conv1d(1, 1, kernel_size=k_size, padding=(k_size - 1) // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # feature descriptor on the global spatial information
        y = self.avg_pool(x)

        # Two different branches of ECA module
        y = self.conv(y.squeeze(-1).transpose(-1, -2)).transpose(-1, -2).unsqueeze(-1)

        # Multi-scale information fusion
        y = self.sigmoid(y)
        return x * y.expand_as(x)

def norm(x):
    return (1 - torch.exp(-x)) / (1 + torch.exp(-x))

def norm_(x):
    import numpy as np
    return (1 - np.exp(-x)) / (1 + np.exp(-x))

class Transformer(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout=0.):
        super().__init__()
        self.layers = nn.ModuleList([])
        self.norm = nn.LayerNorm(dim)
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                PreNorm(dim, Attention(dim, heads=heads, dim_head=dim_head, dropout=dropout)),
                PreNorm(dim, FeedForward(dim, mlp_dim, dropout=dropout))
            ]))

    def forward(self, x):
        for attn, ff in self.layers:
            x = attn(x) + x
            x = ff(x) + x
        return self.norm(x)


class HOR(nn.Module):
    def __init__(self):
        super(HOR, self).__init__()
        self.high = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=1)
        self.low = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=1)

        self.value = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=1)
        self.softmax = nn.Softmax(dim=1)

        self.e_conv = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=1)
        self.mid = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=1)
        self.latter = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=1)

    def forward(self, x_latter, x):
        b, c, h, w = x_latter.shape
        _, c_, _, _ = x.shape
        x_latter_ = self.high(x_latter).reshape(b, c, h * w).contiguous()
        x_ = self.low(x).reshape(b, c_, h * w).permute(0, 2, 1).contiguous()

        p = torch.bmm(x_, x_latter_).contiguous()
        p = self.softmax(p).contiguous()

        e_ = torch.bmm(p, self.value(x).reshape(b, c, h * w).permute(0, 2, 1)).contiguous()
        e = e_ + x_
        e = e.permute(0, 2, 1).contiguous()
        e = self.e_conv(e.reshape(b, c, h, w)).reshape(b, c, h * w).contiguous()

        # e = e.permute(0, 2, 1)
        x_latter_ = self.latter(x_latter).reshape(b, c, h * w).permute(0, 2, 1).contiguous()
        t = torch.bmm(e, x_latter_).contiguous()
        t = self.softmax(t).contiguous()

        x_ = self.mid(x).view(b, c_, h * w).permute(0, 2, 1).contiguous()
        out = torch.bmm(x_, t).permute(0, 2, 1).reshape(b, c, h, w).contiguous()

        return out


class channel_shuffle(nn.Module):
    def __init__(self,groups=4):
        super(channel_shuffle,self).__init__()
        self.groups=groups
    def forward(self,x):
        batchsize, num_channels, height, width = x.data.size()
        channels_per_group = num_channels // self.groups

        # reshape
        x = x.view(batchsize, self.groups,
               channels_per_group, height, width)
        x = torch.transpose(x, 1, 2).contiguous()

        # flatten
        x = x.view(batchsize, -1, height, width)
        return x

class two_ConvBnRule(nn.Module):

    def __init__(self, in_chan, out_chan= 64):
        super(two_ConvBnRule, self).__init__()

        self.conv1 = nn.Conv2d(
            in_channels=in_chan,
            out_channels=out_chan,
            kernel_size=3,
            padding=1
        )
        self.BN1 = nn.BatchNorm2d(out_chan)
        self.relu1 = nn.ReLU(inplace=True)

        self.conv2 = nn.Conv2d(
            in_channels=out_chan,
            out_channels=out_chan,
            kernel_size=3,
            padding=1
        )
        self.BN2 = nn.BatchNorm2d(out_chan)
        self.relu2 = nn.ReLU(inplace=True)

    def forward(self, x, mid=False):
        feat = self.conv1(x)
        feat = self.BN1(feat)
        feat = self.relu1(feat)

        if mid:
            feat_mid = feat

        feat = self.conv2(feat)
        feat = self.BN2(feat)
        feat = self.relu2(feat)

        if mid:
            return feat, feat_mid
        return feat
class two_ConvBnRule_new(nn.Module):

    def __init__(self, in_chan, out_chan):
        super(two_ConvBnRule_new, self).__init__()

        self.conv1 = nn.Conv2d(
            in_channels=in_chan,
            out_channels=out_chan,
            kernel_size=3,
            padding=1
        )
        self.BN1 = nn.BatchNorm2d(out_chan)
        self.relu1 = nn.ReLU(inplace=True)

        self.conv2 = nn.Conv2d(
            in_channels=out_chan,
            out_channels=out_chan,
            kernel_size=3,
            padding=1
        )
        self.BN2 = nn.BatchNorm2d(out_chan)
        self.relu2 = nn.ReLU(inplace=True)

    def forward(self, x, mid=False):
        feat = self.conv1(x)
        feat = self.BN1(feat)
        feat = self.relu1(feat)

        if mid:
            feat_mid = feat

        feat = self.conv2(feat)
        feat = self.BN2(feat)
        feat = self.relu2(feat)

        if mid:
            return feat, feat_mid
        return feat

class two_ConvBnRule_back(nn.Module):

    def __init__(self, out_chan):
        super(two_ConvBnRule_back, self).__init__()

        self.conv1 = nn.Conv2d(
            in_channels=64,
            out_channels=out_chan,
            kernel_size=3,
            padding=1
        )
        self.BN1 = nn.BatchNorm2d(out_chan)
        self.relu1 = nn.ReLU(inplace=True)

        self.conv2 = nn.Conv2d(
            in_channels=out_chan,
            out_channels=out_chan,
            kernel_size=3,
            padding=1
        )
        self.BN2 = nn.BatchNorm2d(out_chan)
        self.relu2 = nn.ReLU(inplace=True)
    def forward(self, x, mid=False):
        feat = self.conv1(x)
        feat = self.BN1(feat)
        feat = self.relu1(feat)
        feat = self.conv2(feat)
        feat = self.BN2(feat)
        feat = self.relu2(feat)

        return feat

def Seg():

    dict = {     0: 0, 1: 1, 2: 8, 3: 16, 4: 9, 5: 2, 6: 3, 7: 10, 8: 17,
                 9: 24, 10: 32, 11: 25, 12: 18, 13: 11, 14: 4, 15: 5, 16: 12,
                 17: 19, 18: 26, 19: 33, 20: 40, 21: 48, 22: 41, 23: 34, 24: 27,
                 25: 20, 26: 13, 27: 6, 28: 7, 29: 14, 30: 21, 31: 28, 32: 35,
                 33: 42, 34: 49, 35: 56, 36: 57, 37: 50, 38: 43, 39: 36, 40: 29,
                 41: 22, 42: 15, 43: 23, 44: 30, 45: 37, 46: 44, 47: 51, 48: 58,
                 49: 59, 50: 52, 51: 45, 52: 38, 53: 31, 54: 39, 55: 46, 56: 53,
                 57: 60, 58: 61, 59: 54, 60: 47, 61: 55, 62: 62, 63: 63}
    a = torch.zeros(1, 64, 1, 1)

    for i in range(0, 32):
        a[0, dict[i+32], 0, 0] = 1
        # a[0, dict[i], 0, 0] = 0.5

    return a


class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.fc1   = nn.Conv2d(in_planes, in_planes // 16, 1, bias=False)
        self.relu1 = nn.ReLU()
        self.fc2   = nn.Conv2d(in_planes // 16, in_planes, 1, bias=False)

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
class cbam_block(nn.Module):
    def __init__(self, channel, ratio=8, kernel_size=7):
        super(cbam_block, self).__init__()
        self.channelattention = ChannelAttention(channel, ratio=ratio)
        self.spatialattention = SpatialAttention(kernel_size=kernel_size)

    def forward(self, x):
        x = x * self.channelattention(x)
        x = x * self.spatialattention(x)
        return x


class PAM(nn.Module):

    def __init__(self, in_dim):

        super(PAM, self).__init__()
        self.conv = nn.Conv2d(
            in_channels=in_dim*2,
            out_channels= 2,
            kernel_size=3,
            padding=1
        )

        self.v_rgb = nn.Parameter(torch.randn((1,in_dim,1,1)),requires_grad=True)
        self.v_freq = nn.Parameter(torch.randn((1,in_dim,1,1)),requires_grad=True)

    def forward(self, rgb, freq):

        attmap = self.conv( torch.cat( (rgb,freq),1) )
        attmap = torch.sigmoid(attmap)

        rgb = attmap[:,0:1,:,:] * rgb * self.v_rgb
        freq = attmap[:,1:,:,:] * freq * self.v_freq
        out = rgb + freq

        return out

class CANet(nn.Module):
    def __init__(self,
                 config,
                 block,
                 blocks_num,
                 groups=1,
                 width_per_group=64,
                 ):
        super().__init__()
        #-----------------初始参数
        self.config = config
        bits = self.config.code_length
        num_classes = self.config.classlen

        self.alpha1 = nn.Parameter(torch.ones(1) * 1, requires_grad=True)
        self.alpha2 = nn.Parameter(torch.ones(1) * 1, requires_grad=True)

        self.b1 = nn.Linear(1024, bits)
        self.b2 = nn.Linear(1024, bits)
        self.b3 = nn.Linear(1024, bits)
        self.b_cat = nn.Linear(3072, bits)
        self.num_ftrs = 2048
        self.feature_size = 512
        self.fc_x = nn.Linear(self.num_ftrs, num_classes)

        self.in_channel = 64
        self.groups = groups
        self.width_per_group = width_per_group
        self.conv1 = nn.Conv2d(3, self.in_channel, kernel_size=7, stride=2,
                               padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(self.in_channel)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, blocks_num[0])
        self.layer2 = self._make_layer(block, 128, blocks_num[1], stride=2)
        self.layer3 = self._make_layer(block, 256, blocks_num[2], stride=2)
        self.layer4 = self._make_layer(block, 512, blocks_num[3], stride=2)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))  # output size = (1, 1)
        self.fc = nn.Linear(bits, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')

        # 网络的第一层加入注意力机制
        self.ca = ChannelAttention(self.in_channel)
        self.sa = SpatialAttention()
        # 网络的卷积层的最后一层加入注意力机制
        self.ca1 = ChannelAttention(2048)
        self.sa1 = SpatialAttention()

        self.relu = nn.ReLU(inplace=True)
        self.num_ftrs = 2048
        self.feature_size = 512
        self.fc_x = nn.Linear(self.num_ftrs, num_classes)
        self.seg = Seg()
        self.hor = HOR()
        self.con1_2 = nn.Conv2d(in_channels=192, out_channels=64, kernel_size=1)
        self.con1_3 = nn.Conv2d(in_channels=192, out_channels=64, kernel_size=1)
        self.con1_4 = nn.Conv2d(in_channels=192, out_channels=64, kernel_size=1)
        self.con1_5 = nn.Conv2d(in_channels=192, out_channels=64, kernel_size=1)

        self.shuffle = channel_shuffle()
        self.high_band = Transformer(dim=256, depth=1, heads=2, dim_head=128, mlp_dim=128 * 2, dropout=0)
        self.low_band = Transformer(dim=256, depth=1, heads=2, dim_head=128, mlp_dim=128 * 2, dropout=0)

        self.PAM2 = PAM(in_dim=64)
        self.PAM3 = PAM(in_dim=64)
        self.PAM4 = PAM(in_dim=64)
        self.PAM5 = PAM(in_dim=64)

        self.conv_r2 = two_ConvBnRule_back(64)
        self.conv_r3 = two_ConvBnRule_back(512)
        self.conv_r4 = two_ConvBnRule_back(1024)
        self.conv_r5 = two_ConvBnRule_back(2048)

        self.conv_l2 = two_ConvBnRule(256)
        self.conv_l3 = two_ConvBnRule(512)
        self.conv_l4 = two_ConvBnRule(1024)
        self.conv_l5 = two_ConvBnRule(2048)

        # decoder_convlution#
        "chanal_decoder1 = chanal_feat5 + 64 = 1028 + 64 =1092"
        self.conv_decoder1 = two_ConvBnRule_new(3072,1024)
        self.conv_decoder2 = two_ConvBnRule_new(1536,512)
        self.conv_decoder3 = two_ConvBnRule_new(576,64)

        self.band = Transformer(dim=256, depth=1, heads=2, dim_head=128, mlp_dim=128 * 2, dropout=0)
        self.spatial = Transformer(dim=192, depth=1, heads=2, dim_head=64, mlp_dim=64 * 2, dropout=0)
        # stage 0
        self.conv_block0 = nn.Sequential(
            BasicConv(self.num_ftrs // 32, self.feature_size, kernel_size=1, stride=1, padding=0, relu=True),
            BasicConv(self.feature_size, 1024, kernel_size=3, stride=1, padding=1, relu=True),
            nn.AdaptiveMaxPool2d(1),
        )
        #

        # stage 1
        self.conv_block1 = nn.Sequential(
            BasicConv(self.num_ftrs // 4, self.feature_size, kernel_size=1, stride=1, padding=0, relu=True),
            BasicConv(self.feature_size, self.num_ftrs // 2, kernel_size=3, stride=1, padding=1, relu=True),
            nn.AdaptiveMaxPool2d(1),
        )

        # stage 2
        self.conv_block2 = nn.Sequential(
            BasicConv(self.num_ftrs // 2, self.feature_size, kernel_size=1, stride=1, padding=0, relu=True),
            BasicConv(self.feature_size, self.num_ftrs // 2, kernel_size=3, stride=1, padding=1, relu=True),
            nn.AdaptiveMaxPool2d(1),
        )

        # stage 3
        self.conv_block3 = nn.Sequential(
            BasicConv(self.num_ftrs, self.feature_size, kernel_size=1, stride=1, padding=0, relu=True),
            BasicConv(self.feature_size, self.num_ftrs // 2, kernel_size=3, stride=1, padding=1, relu=True),
            nn.AdaptiveMaxPool2d(1),
        )

        # concat features from different stages
        self.hashing_concat = nn.Sequential(
            nn.BatchNorm1d(self.num_ftrs // 2 * 4, affine=True),
            nn.Linear(self.num_ftrs // 2 * 4, self.feature_size),
            nn.BatchNorm1d(self.feature_size, affine=True),
            nn.ELU(inplace=True),
            nn.Linear(self.feature_size, bits),
        )
#-------------------------------
        self.eca = eca_layer(64)
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
    def forward_vanilla(self, input):
        x,DCT_x = input
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        # x = self.ca(x) * x
        # x = self.sa(x) * x

        x1 = self.maxpool(x)

        x2 = self.layer1(x1)
        f1_ori = self.backbone.layer2(x2)
        f1_eca = self.eca(f1_ori)

        f2_ori = self.backbone.layer3(f1_ori)
        f2_eca = self.eca(f2_ori)

        conv5_b = self.backbone.layer4[:2](f2_ori)

        f3_ori = self.backbone.layer4[2](conv5_b)
        # f3_ori = self.backbone.layer4(f2_ori)
        f3_eca = self.eca(f3_ori)

        feats = torch.add(f3_eca,f3_ori)


        f1 = torch.add(f1_eca,f1_ori)
        f2 = torch.add(f2_eca,f2_ori)
        f3 = torch.add(f3_eca,f3_ori)

        # f3 = self.ca1(f3) * f3
        # f3 = self.sa1(f3) * f3

        x2 = self.conv_l2(x2)
        f1 = self.conv_l3(f1)
        f2 = self.conv_l4(f2)
        f3,f3_mid = self.conv_l5(f3, mid=True)
        f3 = self.hor(f3, f3_mid)

        # ----------------------------这里的缩减channel是为什么？？？这里需要参考dct的论文
        self.seg = self.seg.to(DCT_x.device)
        feat_y = DCT_x[:, 0:64, :, :] * self.seg
        feat_Cb = DCT_x[:, 64:128, :, :] * self.seg
        feat_Cr = DCT_x[:, 128:192, :, :] * self.seg
        origin_feat_DCT = torch.cat((torch.cat((feat_y, feat_Cb), 1), feat_Cr), 1)
        origin_feat_DCT = self.shuffle(origin_feat_DCT)

        high = torch.cat([feat_y[:, 32:, :, :], feat_Cb[:, 32:, :, :], feat_Cr[:, 32:, :, :]], 1)
        low = torch.cat([feat_y[:, :32, :, :], feat_Cb[:, :32, :, :], feat_Cr[:, :32, :, :]], 1)

        b, n, h, w = high.shape
        high = torch.nn.functional.interpolate(high, size=(16, 16))
        low = torch.nn.functional.interpolate(low, size=(16, 16))
        high = rearrange(high, 'b n h w -> b n (h w)')
        low = rearrange(low, 'b n h w -> b n (h w)')

        high = self.high_band(high)
        low = self.low_band(low)

        y_h, b_h, r_h = torch.split(high, 32, 1)
        y_l, b_l, r_l = torch.split(low, 32, 1)

        feat_y = torch.cat([y_l, y_h], 1)
        feat_Cb = torch.cat([b_l, b_h], 1)
        feat_Cr = torch.cat([r_l, r_h], 1)
        feat_DCT = torch.cat((torch.cat((feat_y, feat_Cb), 1), feat_Cr), 1)


        feat_DCT = self.band(feat_DCT)                      # Figur3 step2 left output
        feat_DCT = feat_DCT.transpose(1, 2)
        feat_DCT = self.spatial(feat_DCT)                   # Figur3 step2 right
        feat_DCT = feat_DCT.transpose(1, 2)
        feat_DCT = rearrange(feat_DCT, 'b n (h w) -> b n h w', h=16)
        feat_DCT = torch.nn.functional.interpolate(feat_DCT, size=(h, w))

        feat_DCT = origin_feat_DCT + feat_DCT               # rgb + freq

        #using 1*1conv to change the numbers of the channel of DCT_x
        feat_DCT2 = self.con1_2(feat_DCT)
        feat_DCT3 = self.con1_3(feat_DCT)
        feat_DCT4 = self.con1_4(feat_DCT)
        feat_DCT5 = self.con1_5(feat_DCT)



        feat_DCT2 = torch.nn.functional.interpolate(feat_DCT2,size=x2.size()[2:],mode='bilinear',align_corners=True)
        feat_DCT3 = torch.nn.functional.interpolate(feat_DCT3,size=f1.size()[2:],mode='bilinear',align_corners=True)
        feat_DCT4 = torch.nn.functional.interpolate(feat_DCT4,size=f2.size()[2:],mode='bilinear',align_corners=True)
        feat_DCT5 = torch.nn.functional.interpolate(feat_DCT5,size=f3.size()[2:],mode='bilinear',align_corners=True)

        #feature fusion
        x2 = self.PAM2(x2, feat_DCT2)
        f1 = self.PAM3(f1, feat_DCT3)
        f2 = self.PAM4(f2, feat_DCT4)
        f3 = self.PAM5(f3, feat_DCT5)

        feat2 = self.conv_r2(x2)
        feat3 = self.conv_r3(f1)
        feat4 = self.conv_r4(f2)
        feat5 = self.conv_r5(f3)

        # connect feat5 and feat4#
        size4 = feat4.size()[2:]
        feat5 = torch.nn.functional.interpolate(feat5, size=size4, mode='bilinear', align_corners=True)
        feat4 = torch.cat((feat4, feat5), 1)
        feat4 = self.conv_decoder1(feat4)

        size3 = feat3.size()[2:]
        feat4 = torch.nn.functional.interpolate(feat4, size=size3, mode='bilinear', align_corners=True)
        feat3 = torch.cat((feat3, feat4), 1)
        feat3 = self.conv_decoder2(feat3)

        # connect feat3 and feat2#
        size2 = feat2.size()[2:]
        feat3 = torch.nn.functional.interpolate(feat3, size=size2, mode='bilinear', align_corners=True)
        feat2 = torch.cat((feat2, feat3), 1)
        feat2 = self.conv_decoder3(feat2)

        # -------------------------------------------------------------------------------------------------
        f00 = self.conv_block0(feat2).view(-1, self.num_ftrs // 2)
        f11 = self.conv_block1(feat3).view(-1, self.num_ftrs // 2)
        f22 = self.conv_block2(feat4).view(-1, self.num_ftrs // 2)
        f33 = self.conv_block3(feat5).view(-1, self.num_ftrs // 2)

        f33_b = self.b3(f33)
        output = self.fc(f33_b)
        f44 = torch.cat((f00, f11, f22, f33), -1)

        f44_b = self.hashing_concat(f44)
        return self.alpha1, self.alpha2, f44_b, output, feats,f44

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

# 残差层 18和34层数
class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_channel, out_channel, stride=1, downsample=None, **kwargs):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=in_channel, out_channels=out_channel,
                               kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channel)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv2d(in_channels=out_channel, out_channels=out_channel,
                               kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channel)
        self.downsample = downsample

    def forward(self, x):
        identity = x
        if self.downsample is not None:
            identity = self.downsample(x)

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        out += identity
        out = self.relu(out)

        return out

# 残差层数 50和101
class Bottleneck(nn.Module):
    """
    注意：原论文中，在虚线残差结构的主分支上，第一个1x1卷积层的步距是2，第二个3x3卷积层步距是1。
    但在pytorch官方实现过程中是第一个1x1卷积层的步距是1，第二个3x3卷积层步距是2，
    这么做的好处是能够在top1上提升大概0.5%的准确率。
    可参考Resnet v1.5 https://ngc.nvidia.com/catalog/model-scripts/nvidia:resnet_50_v1_5_for_pytorch
    """
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
        self.conv3 = nn.Conv2d(in_channels=width, out_channels=out_channel*self.expansion,
                               kernel_size=1, stride=1, bias=False)  # unsqueeze channels
        self.bn3 = nn.BatchNorm2d(out_channel*self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        # self.cbam = cbam_block(out_channel * self.expansion)  # 这里的问题

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

        # out = self.cbam(out)

        out += identity
        out = self.relu(out)

        return out
def resnet50(config):
    # https://download.pytorch.org/models/resnet50-19c8e357.pth
    return CANet(config, Bottleneck, [3, 4, 6, 3])
