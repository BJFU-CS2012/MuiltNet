import torch

from torch import nn
import torchvision
from utils.dct_utils import PreNorm, FeedForward, Attention
from einops import rearrange


# Bilinear Attention Pooling
EPSILON = 1e-6

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

        if mid:
            feat_mid = feat

        feat = self.conv2(feat)
        feat = self.BN2(feat)
        feat = self.relu2(feat)

        if mid:
            return feat, feat_mid
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

    return a


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
            self.seg = Seg()
            self.hor = HOR()
            self.con1_2 = nn.Conv2d(in_channels=192, out_channels=64, kernel_size=1)
            self.con1_3 = nn.Conv2d(in_channels=192, out_channels=64, kernel_size=1)
            self.con1_4 = nn.Conv2d(in_channels=192, out_channels=64, kernel_size=1)
            self.con1_5 = nn.Conv2d(in_channels=192, out_channels=64, kernel_size=1)
            self.vector_y = nn.Parameter(torch.FloatTensor(1, 64, 1, 1), requires_grad=True)
            self.vector_cb = nn.Parameter(torch.FloatTensor(1, 64, 1, 1), requires_grad=True)
            self.vector_cr = nn.Parameter(torch.FloatTensor(1, 64, 1, 1), requires_grad=True)
            self.freq_out_1 = nn.Conv2d(64, 1, 1, 1, 0)
            self.freq_out_2 = nn.Conv2d(64, 1, 1, 1, 0)
            self.freq_out_3 = nn.Conv2d(64, 1, 1, 1, 0)
            self.freq_out_4 = nn.Conv2d(64, 1, 1, 1, 0)
            self.shuffle = channel_shuffle()
            self.high_band = Transformer(dim=256, depth=1, heads=2, dim_head=128, mlp_dim=128 * 2, dropout=0)
            self.low_band = Transformer(dim=256, depth=1, heads=2, dim_head=128, mlp_dim=128 * 2, dropout=0)
            # output
            self.conv_out = nn.Conv2d(
                in_channels=64,
                out_channels=1,
                padding=1,
                kernel_size=3
            )
            self.conv_out_2 = nn.Conv2d(
                in_channels=64,
                out_channels=1,
                padding=1,
                kernel_size=3
            )
            self.conv_out_3 = nn.Conv2d(
                in_channels=64,
                out_channels=1,
                padding=1,
                kernel_size=3
            )
            self.conv_out_4 = nn.Conv2d(
                in_channels=64,
                out_channels=1,
                padding=1,
                kernel_size=3
            )
            self.conv_decoder1 = two_ConvBnRule(128)
            self.conv_decoder2 = two_ConvBnRule(128)
            self.conv_decoder3 = two_ConvBnRule(128)

            self.PAM2 = PAM(in_dim=64)
            self.PAM3 = PAM(in_dim=64)
            self.PAM4 = PAM(in_dim=64)
            self.PAM5 = PAM(in_dim=64)

            self.conv_r2 = two_ConvBnRule(64)
            self.conv_r3 = two_ConvBnRule_back(512)
            self.conv_r4 = two_ConvBnRule_back(1024)
            self.conv_r5 = two_ConvBnRule_back(2048)

            self.conv_l2 = two_ConvBnRule(256)
            self.conv_l3 = two_ConvBnRule(512)
            self.conv_l4 = two_ConvBnRule(1024)
            self.conv_l5 = two_ConvBnRule(2048)
            self.band = Transformer(dim=256, depth=1, heads=2, dim_head=128, mlp_dim=128 * 2, dropout=0)
            self.spatial = Transformer(dim=192, depth=1, heads=2, dim_head=64, mlp_dim=64 * 2, dropout=0)
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
            # stage 1
            self.backbone.conv_block4 = nn.Sequential(
                BasicConv(64, self.feature_size, kernel_size=1, stride=1, padding=0, relu=True),
                BasicConv(self.feature_size, 96, kernel_size=3, stride=1, padding=1, relu=True),
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

            # model_s#
            self.conv_l2 = two_ConvBnRule(256)
            self.conv_l3 = two_ConvBnRule(512)
            self.conv_l4 = two_ConvBnRule(1024)
            self.conv_l5 = two_ConvBnRule(2048)

            self.freq_out_1 = nn.Conv2d(64, 1, 1, 1, 0)
            self.freq_out_2 = nn.Conv2d(64, 1, 1, 1, 0)
            self.freq_out_3 = nn.Conv2d(64, 1, 1, 1, 0)
            self.freq_out_4 = nn.Conv2d(64, 1, 1, 1, 0)

            # print('after\n',self.backbone)

    def forward(self, x):
        return self.forward_vanilla(x)

    def forward_vanilla(self, input):

        x,DCT_x = input

        x = self.backbone.conv1(x)
        x = self.backbone.bn1(x)
        x = self.backbone.relu(x)
        x1 = self.backbone.maxpool(x)

        x2 = self.backbone.layer1(x1)

        f1 = self.backbone.layer2(x2)
        f2 = self.backbone.layer3(f1)
        f3 = self.backbone.layer4(f2)
        # print('---->f1.shape',f1.shape) #torch.Size([32, 512, 28, 28])
        # print('---->f2.shape',f2.shape) #torch.Size([32, 1024, 14, 14])
        # print('---->f3.shape',f3.shape) #torch.Size([32, 2048, 7, 7])
        feats = f3

        # 这里的缩减channel是为什么？？？这里需要参考dct的论文
        x2 = self.conv_l2(x2)
        f1 = self.conv_l3(f1)
        f2 = self.conv_l4(f2)
        f3, f3_mid = self.conv_l5(f3, mid=True)
        f3 = self.hor(f3, f3_mid)
        # print('====>f1.shape',f1.shape) #torch.Size([32, 64, 28, 28]) => 512
        # print('====>f2.shape',f2.shape) # torch.Size([32, 64, 14, 14]) => 1024
        # print('====>f3.shape',f3.shape) # torch.Size([32, 64, 7, 7])      => 2048

        self.seg = self.seg.to(DCT_x.device)
        feat_y = DCT_x[:, 0:64, :, :] * (self.seg + norm(self.vector_y))
        feat_Cb = DCT_x[:, 64:128, :, :] * (self.seg + norm(self.vector_cb))
        feat_Cr = DCT_x[:, 128:192, :, :] * (self.seg + norm(self.vector_cr))
        origin_feat_DCT = torch.cat((torch.cat((feat_y, feat_Cb), 1), feat_Cr), 1)
        origin_feat_DCT = self.shuffle(origin_feat_DCT)
        # print('feat_y', feat_y.shape)  # torch.Size([32, 64, 28, 28])
        # print('>>>>origin_feat_DCT.shape', origin_feat_DCT.shape)  # torch.Size([32, 192, 28, 28])

        high = torch.cat([feat_y[:, 32:, :, :], feat_Cb[:, 32:, :, :], feat_Cr[:, 32:, :, :]], 1)
        low = torch.cat([feat_y[:, :32, :, :], feat_Cb[:, :32, :, :], feat_Cr[:, :32, :, :]], 1)
        # print('high1', high.shape) #t orch.Size([32, 96, 28, 28])

        b, n, h, w = high.shape
        high = torch.nn.functional.interpolate(high, size=(16, 16))
        low = torch.nn.functional.interpolate(low, size=(16, 16))
        # print('high2', high.shape) # torch.Size([32, 96, 16, 16])

        high = rearrange(high, 'b n h w -> b n (h w)')
        low = rearrange(low, 'b n h w -> b n (h w)')

        high = self.high_band(high)
        low = self.low_band(low)
        # print('high3', high.shape) # high3 torch.Size([32, 96, 256])

        y_h, b_h, r_h = torch.split(high, 32, 1)
        y_l, b_l, r_l = torch.split(low, 32, 1)
        # print('y_h.shape', y_h.shape)  torch.Size([32, 32, 256])
        # 这里又给平起来是为什么？？？？？dct
        feat_y = torch.cat([y_l, y_h], 1)
        feat_Cb = torch.cat([b_l, b_h], 1)
        feat_Cr = torch.cat([r_l, r_h], 1)
        feat_DCT = torch.cat((torch.cat((feat_y, feat_Cb), 1), feat_Cr), 1)
        # print('feat_y.shape', feat_y.shape) #  torch.Size([32, 64, 256])
        # print('>>>>feat_DCT.shape', feat_DCT.shape) # torch.Size([32, 192, 256])

        feat_DCT = self.band(feat_DCT)                      # Figur3 step2 left output
        feat_DCT = feat_DCT.transpose(1, 2)
        feat_DCT = self.spatial(feat_DCT)                   # Figur3 step2 right
        feat_DCT = feat_DCT.transpose(1, 2)
        feat_DCT = rearrange(feat_DCT, 'b n (h w) -> b n h w', h=16)
        feat_DCT = torch.nn.functional.interpolate(feat_DCT, size=(h, w))
        feat_DCT = origin_feat_DCT + feat_DCT               # rgb + freq
        # print('feat_DCT1.shape', feat_DCT.shape)    # torch.Size([32, 192, 256])
        # print('feat_DCT2.shape', feat_DCT.shape)    # torch.Size([32, 256, 192])
        # print('feat_DCT3.shape', feat_DCT.shape)    # torch.Size([32, 256, 192])
        # print('feat_DCT4.shape', feat_DCT.shape)    # torch.Size([32, 192, 256])
        # print('feat_DCT5.shape', feat_DCT.shape)    # torch.Size([32, 192, 16, 16])
        # print('feat_DCT6.shape', feat_DCT.shape)    # torch.Size([32, 192, 28, 28])
        # print('origin_feat_DCT.shape', origin_feat_DCT.shape) # torch.Size([32, 192, 28, 28])
        # print('feat_DCT7.shape', feat_DCT.shape) # torch.Size([32, 192, 28, 28])

        #using 1*1conv to change the numbers of the channel of DCT_x
        feat_DCT2 = self.con1_2(feat_DCT)
        feat_DCT3 = self.con1_3(feat_DCT)
        feat_DCT4 = self.con1_4(feat_DCT)
        feat_DCT5 = self.con1_5(feat_DCT)
        # print('feat_DCT2.shape',feat_DCT2.shape) # torch.Size([32, 64, 28, 28])
        # print('feat_DCT3.shape',feat_DCT3.shape) # torch.Size([32, 64, 28, 28])
        # print('feat_DCT4.shape',feat_DCT4.shape) # torch.Size([32, 64, 28, 28])
        # print('feat_DCT5.shape',feat_DCT5.shape) # torch.Size([32, 64, 28, 28])
        #
        # print('x2.size()',x2.size()) # torch.Size([32, 64, 56, 56])
        # print('f1.size()',f1.size()) # torch.Size([32, 64, 28, 28])
        # print('f2.size()',f2.size()) # torch.Size([32, 64, 14, 14])
        # print('f3.size()',f3.size()) # torch.Size([32, 64, 7, 7])
        # print('f3.size()',f3.size()[2:]) # torch.Size([7, 7])

        feat_DCT2 = torch.nn.functional.interpolate(feat_DCT2,size=x2.size()[2:],mode='bilinear',align_corners=True)
        feat_DCT3 = torch.nn.functional.interpolate(feat_DCT3,size=f1.size()[2:],mode='bilinear',align_corners=True)
        feat_DCT4 = torch.nn.functional.interpolate(feat_DCT4,size=f2.size()[2:],mode='bilinear',align_corners=True)
        feat_DCT5 = torch.nn.functional.interpolate(feat_DCT5,size=f3.size()[2:],mode='bilinear',align_corners=True)
        # print('feat_DCT2.shape',feat_DCT2.shape) # torch.Size([32, 64, 56, 56])
        # print('feat_DCT3.shape',feat_DCT3.shape) # torch.Size([32, 64, 28, 28])
        # print('feat_DCT4.shape',feat_DCT4.shape) # torch.Size([32, 64, 14, 14])
        # print('feat_DCT5.shape',feat_DCT5.shape) # torch.Size([32, 64, 7, 7])

        #feature fusion
        x2 = self.PAM2(x2, feat_DCT2)
        f1 = self.PAM3(f1, feat_DCT3)
        f2 = self.PAM4(f2, feat_DCT4)
        f3 = self.PAM5(f3, feat_DCT5)
        # print('x2.shape',x2.shape) # torch.Size([32, 64, 56, 56])
        # print('f1.shape',f1.shape) # torch.Size([32, 64, 28, 28])
        # print('f2.shape',f2.shape) # torch.Size([32, 64, 14, 14])
        # print('f3.shape',f3.shape) # torch.Size([32, 64, 7, 7])

        # feat2 = self.conv_r2(x2)
        feat3 = self.conv_r3(f1)
        feat4 = self.conv_r4(f2)
        feat5 = self.conv_r5(f3)
        # print('feat2.shape',feat2.shape) # torch.Size([32, 64, 56, 56])
        # print('feat3.shape',feat3.shape) # torch.Size([32, 64, 28, 28])
        # print('feat4.shape',feat4.shape) # torch.Size([32, 64, 14, 14])
        # print('feat5.shape',feat5.shape) # torch.Size([32, 64, 7, 7])
# ------------------------------这里我直接折叠了
#         #connect feat5 and feat4#
#         size4 = feat4.size()[2:]
#         feat5 = torch.nn.functional.interpolate(feat5, size=size4, mode='bilinear', align_corners=True)
#         feat4 = torch.cat((feat4, feat5), 1)
#         feat4 = self.conv_decoder1(feat4)
#         # print('feat4.shape', feat4.shape)  # torch.Size([32, 64, 14, 14])
#
#         # connect feat4 and feat3#
#         size3 = feat3.size()[2:]
#         feat4 = torch.nn.functional.interpolate(feat4, size=size3, mode='bilinear', align_corners=True)
#         feat3 = torch.cat((feat3, feat4), 1)
#         feat3 = self.conv_decoder2(feat3)
#         # print('feat3.shape', feat3.shape)  # torch.Size([32, 64, 28, 28])
#         # connect feat3 and feat2#
#         size2 = feat2.size()[2:]
#         feat3 = torch.nn.functional.interpolate(feat3, size=size2, mode='bilinear', align_corners=True)
#         feat2 = torch.cat((feat2, feat3), 1)
#         feat2 = self.conv_decoder3(feat2)
#         # print('feat2.shape', feat2.shape)  # torch.Size([32, 64, 56, 56])
#
#         #output
#         sizex = x.size()[2:]
#         output = self.conv_out(feat2)
#         output_1 = self.conv_out(feat3)
#         output_2 = self.conv_out(feat4)
#         output_3 = self.conv_out(feat5)
#         # output_1 = torch.nn.functional.interpolate(output_1, size=sizex, mode='bilinear', align_corners=True)
#         # output_2 = torch.nn.functional.interpolate(output_2, size=sizex, mode='bilinear', align_corners=True)
#         # output_3 = torch.nn.functional.interpolate(output_3, size=sizex, mode='bilinear', align_corners=True)
#         # print('output.shape', output.shape)     # torch.Size([32, 1, 56, 56])
#         # print('output_1.shape', output_1.shape) # torch.Size([32, 1, 56, 56])
#         # print('output_2.shape', output_2.shape) # torch.Size([32, 1, 28, 28])
#         # print('output_3.shape', output_3.shape) # torch.Size([32, 1, 14, 14])
#
#         # print('output_1.interpolate.shape', output_1.shape) # torch.Size([32, 1, 112, 112])
#         # print('output_2.interpolate.shape', output_2.shape) # torch.Size([32, 1, 112, 112])
#         # print('output_3.interpolate.shape', output_3.shape) # torch.Size([32, 1, 112, 112])
#

        # -------------------------------------------------------------------------------------------------
        f11 = self.backbone.conv_block1(feat3).view(-1, self.num_ftrs // 2)
        f22 = self.backbone.conv_block2(feat4).view(-1, self.num_ftrs // 2)
        f33 = self.backbone.conv_block3(feat5).view(-1, self.num_ftrs // 2)
        f33_b = self.backbone.b3(f33)
        output = self.backbone.fc(f33_b)
        f44 = torch.cat((f11, f22, f33), -1)
        f44_b = self.backbone.hashing_concat(f44)
        return self.alpha1, self.alpha2, f44_b, output, feats


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