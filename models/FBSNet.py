import torch
import torch.nn as nn
import torch.utils.checkpoint as checkpoint
from mpmath import sigmoid
from timm.models.layers import DropPath, to_2tuple, trunc_normal_
import numpy as np
import torch.nn.functional as F
import os
from swin_transformer import SwinTransformer
from mobilenetv2 import mobilenet_v2
from IIA_module import FL

def conv3x3(in_planes, out_planes, stride=1, has_bias=False):
    "3x3 convolution with padding"
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=has_bias)

def conv3x3_bn_relu(in_planes, out_planes, stride=1):
    return nn.Sequential(
            conv3x3(in_planes, out_planes, stride),
            nn.BatchNorm2d(out_planes),
            nn.ReLU(inplace=True),
            )

class convbnrelu(nn.Module):
    def __init__(self, in_channel, out_channel, k=3, s=1, p=1, g=1, d=1, bias=False, bn=True, relu=True):
        super(convbnrelu, self).__init__()
        conv = [nn.Conv2d(in_channel, out_channel, k, s, p, dilation=d, groups=g, bias=bias)]
        if bn:
            conv.append(nn.BatchNorm2d(out_channel))
        if relu:
            conv.append(nn.ReLU(inplace=True))
        self.conv = nn.Sequential(*conv)

    def forward(self, x):
        return self.conv(x)

class DSConv(nn.Module):
    def __init__(self, in_channel, out_channel, kernel_size=3, stride=1, padding=0,dilation=1, relu=True):
        super(DSConv, self).__init__()
        self.conv = nn.Sequential(
                convbnrelu(in_channel, in_channel, k=kernel_size, s=stride, p=padding, d=dilation, g=in_channel),
                convbnrelu(in_channel, out_channel, k=1, s=1, p=0, relu=relu)
                )

    def forward(self, x):
        return self.conv(x)

class ChannelGate_sub(nn.Module):
    """A mini-network that generates channel-wise gates conditioned on input tensor."""

    def __init__(self, in_channels,  reduction=16):
        super(ChannelGate_sub, self).__init__()

        num_gates = in_channels

        self.global_avgpool = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Conv2d(in_channels, in_channels//reduction, kernel_size=1, bias=True, padding=0)
        self.relu = nn.ReLU(inplace=True)
        self.fc2 = nn.Conv2d(in_channels//reduction, num_gates, kernel_size=1, bias=True, padding=0)

        self.gate_activation = nn.Sigmoid()

    def forward(self, x,y):
        f = x + y
        f = self.global_avgpool(f)
        f = self.fc1(f)
        f = self.relu(f)
        f = self.fc2(f)

        f = self.gate_activation(f)

        return f * x , (1-f) * y

class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super(ChannelAttention, self).__init__()

        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.fc1 = nn.Conv2d(in_planes, in_planes // 16, 1, bias=False)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Conv2d(in_planes // 16, in_planes, 1, bias=False)

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        max_out = self.fc2(self.relu1(self.fc1(self.max_pool(x))))
        out = max_out
        return self.sigmoid(out)


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()

        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        padding = 3 if kernel_size == 7 else 1

        self.conv1 = nn.Conv2d(1, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x=max_out
        x = self.conv1(x)
        return self.sigmoid(x)



class USAM(nn.Module):
    def __init__(self, kernel_size=3, padding=1, polish=True):
        super(USAM, self).__init__()

        kernel = torch.ones((kernel_size, kernel_size))
        kernel = kernel.unsqueeze(0).unsqueeze(0)
        self.weight = nn.Parameter(data=kernel, requires_grad=False)

        kernel2 = torch.ones((1, 1)) * (kernel_size * kernel_size)
        kernel2 = kernel2.unsqueeze(0).unsqueeze(0)
        self.weight2 = nn.Parameter(data=kernel2, requires_grad=False)

        self.polish = polish
        self.pad = padding
        self.relu = nn.ReLU()
        self.bn = nn.BatchNorm2d(1)

    def __call__(self, x):
        fmap = x.sum(1, keepdim=True)
        x1 = F.conv2d(fmap, self.weight, padding=self.pad)
        x2 = F.conv2d(fmap, self.weight2, padding=0)

        att = x2 - x1
        att = self.bn(att)
        att = self.relu(att)

        if self.polish:
            att[:, :, :, 0] = 0
            att[:, :, :, -1] = 0
            att[:, :, 0, :] = 0
            att[:, :, -1, :] = 0

        output = x + att * x

        return output

class IBN(nn.Module):
    def __init__(self, planes):
        super(IBN, self).__init__()
        half1 = int(planes / 2)
        self.half = half1
        half2 = planes - half1
        self.IN = nn.InstanceNorm2d(half1, affine=True)
        self.BN = nn.BatchNorm2d(half2)

    def forward(self, x):
        split = torch.split(x, self.half, 1)
        out1 = self.IN(split[0].contiguous())
        out2 = self.BN(split[1].contiguous())
        out = torch.cat((out1, out2), 1)
        return out



class FusionLayer(nn.Module):#两模态融合层
    def __init__(self, dim):
        super(FusionLayer, self).__init__()


        self.DS1 = DSConv(dim, dim, kernel_size=3, stride=1, padding=1)
        self.DS2 = DSConv(dim, dim, kernel_size=3, stride=1, padding=1)
        self.DS3 = DSConv(dim, dim, kernel_size=3, stride=1, padding=1)
        self.DS4 = DSConv(dim, dim, kernel_size=3, stride=1, padding=1)
        #
        self.pool1 = nn.AdaptiveMaxPool2d(1)
        self.pool2 = nn.AdaptiveMaxPool2d(1)
        self.pool3 = nn.AdaptiveMaxPool2d(1)
        self.pool4 = nn.AdaptiveMaxPool2d(1)
        #
        self.fc = nn.Conv2d(4, 4, 1, bias=False)

        self.fc1 = nn.Conv2d(dim, 1, 1, bias=False)
        self.fc2 = nn.Conv2d(dim, 1, 1, bias=False)
        self.fc3 = nn.Conv2d(dim, 1, 1, bias=False)
        self.fc4 = nn.Conv2d(dim, 1, 1, bias=False)
        #
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x1,x2,x3,x4):
        B, C, H, W = x1.size()
        xp1 = self.fc1(self.pool1(x1))#128 1 1
        xp2 = self.fc2(self.pool2(x2))
        xp3 = self.fc3(self.pool3(x3))
        xp4 = self.fc4(self.pool4(x4))
        #
        xp = torch.cat((xp1,xp2,xp3,xp4),dim=1)# 128*4
        xp = self.fc(xp)
        xp = self.softmax(xp)

        xp = torch.split(xp,1,dim=1)

        x1 = self.DS1(x1) * xp[0]
        x2 = self.DS2(x2) * xp[1]
        x3 = self.DS3(x3) * xp[2]
        x4 = self.DS4(x4) * xp[3]


        out = x1 + x2 + x3 + x4

        return out

class EhanceLayer(nn.Module):
    def __init__(self, dim,):
        super(EhanceLayer, self).__init__()

        self.q = DSConv(dim, dim//4 , kernel_size=3, stride=1, padding=1)
        self.k = DSConv(dim, dim//4, kernel_size=3, stride=1, padding=1)
        self.v = DSConv(dim, dim//4, kernel_size=3, stride=1, padding=1)

        self.out = DSConv(dim//4, dim , kernel_size=3, stride=1, padding=1)

        self.softmax = nn.Softmax(dim=1)
    def forward(self, x):
        B,C,H,W = x.size()


        q = self.q(x)
        k = self.k(x)
        v = self.v(x)

        qk = self.softmax(q * k)

        out = self.out(v * qk) + x

        return out

class BackProjectLayer(nn.Module):
    def __init__(self, dim,):
        super(BackProjectLayer, self).__init__()
        self.sa = SpatialAttention()
        self.global_avgpool = nn.AdaptiveAvgPool2d(1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x1,x2,x3,x4,y):
        fsa = self.sa(y)

        x1 = x1 * fsa + y + x1
        x2 = x2 * fsa + y + x2
        x3 = x3 * fsa + y + x3
        x4 = x4 * fsa + y + x4

        return x1,x2,x3,x4


class ProFusModel(nn.Module):#总程序
    def __init__(self, dim,num_steps):
        super(ProFusModel, self).__init__()

        self.fusion_layer1 = FusionLayer(dim)
        self.fusion_layer2 = FusionLayer(dim)
        self.fusion_layer3 = FusionLayer(dim)
        self.fusion_layer4 = FusionLayer(dim)

        self.back_project_layers = BackProjectLayer(dim)
        self.back_project_layers2 = BackProjectLayer(dim)
        self.back_project_layers3 = BackProjectLayer(dim)

        self.extend1 = EhanceLayer(dim)
        self.extend2 = EhanceLayer(dim)
        self.extend3 = EhanceLayer(dim)

        self.num_steps = num_steps

    def forward(self, x1, x2,x3,x4):
        fuse01 = self.fusion_layer1(x1,x2,x3,x4)
        fuse1 = self.extend1(fuse01)
        x12,x22,x32,x42 = self.back_project_layers(x1,x2,x3,x4,fuse1)

        fuse02 = self.fusion_layer2(x12,x22,x32,x42)
        fuse2 = self.extend2(fuse02)
        x13,x23,x33,x43 = self.back_project_layers2(x12,x22,x32,x42,fuse2)

        fuse03 = self.fusion_layer3(x13,x23,x33,x43)
        fuse3 = self.extend3(fuse03)
        x14, x24, x34, x44 = self.back_project_layers3(x13,x23,x33,x43, fuse3)

        fuse4 = self.fusion_layer4(x14, x24, x34, x44)

        return fuse4,fuse01,fuse02,fuse03

class Level(nn.Module):
    def __init__(self, dim,):
        super(Level, self).__init__()
        self.cov1 = nn.Sequential(nn.Conv2d(128, 128, 1, ))
        self.cov2 = nn.Sequential(nn.Conv2d(256, 128, 1, ), nn.UpsamplingBilinear2d(scale_factor=2))
        self.cov3 = nn.Sequential(nn.Conv2d(512, 128, 1, ), nn.UpsamplingBilinear2d(scale_factor=4))
        self.cov4 = nn.Sequential(nn.Conv2d(1024, 128 , 1, ), nn.UpsamplingBilinear2d(scale_factor=8))

        self.level = ProFusModel(dim,2)

    def forward(self, x1,x2,x3,x4):
        x1 = self.cov1(x1)
        x2 = self.cov2(x2)
        x3 = self.cov3(x3)
        x4 = self.cov4(x4)

        out,out1,out2,out3 = self.level(x1,x2,x3,x4)

        return out,out1,out2,out3

class ASP(nn.Module):
    def __init__(self, dim,):
        super(ASP, self).__init__()
        self.cov1 = convbnrelu(dim,dim//2,k=1,s=1,p=0)
        self.cov3 = nn.Sequential(nn.Conv2d(dim, dim//4, 3,1,1 ),nn.BatchNorm2d(dim//4),nn.ReLU(inplace=True))
        self.cov5 = nn.Sequential(nn.Conv2d(dim, dim//4, 5,1,2 ),nn.BatchNorm2d(dim//4),nn.ReLU(inplace=True))

        self.cov = nn.Sequential(nn.Conv2d(dim, dim, 1,1,0 ),nn.BatchNorm2d(dim),nn.ReLU(inplace=True))
    def forward(self, x):
        x1 = self.cov1(x)
        x3 = self.cov3(x)
        x5 = self.cov5(x)

        out = torch.cat([x1, x3, x5], dim=1)
        out = self.cov(out) + x
        return out

class Gamma(nn.Module):#总程序
    def __init__(self, dim):
        super(Gamma, self).__init__()
        self.rgb_pretrained = mobilenet_v2()
        self.depth_pretrained = mobilenet_v2()

        self.upsample1_l = nn.Sequential(nn.Conv2d(52, 36, 3, 1, 1, ), nn.BatchNorm2d(36), nn.GELU(),
                                         nn.UpsamplingBilinear2d(scale_factor=2, ))

        self.upsample2_l = nn.Sequential(nn.Conv2d(104, 36, 3, 1, 1, ), nn.BatchNorm2d(36), nn.GELU(),
                                         nn.UpsamplingBilinear2d(scale_factor=2, ))

        self.upsample3_l = nn.Sequential(nn.Conv2d(160, 80, 3, 1, 1, ), nn.BatchNorm2d(80), nn.GELU(),
                                         nn.UpsamplingBilinear2d(scale_factor=2, ))

        self.upsample4_l = nn.Sequential(nn.Conv2d(256, 128, 3, 1, 1, ), nn.BatchNorm2d(128), nn.GELU(),
                                         nn.UpsamplingBilinear2d(scale_factor=2, ))

        self.upsample5_l = nn.Sequential(nn.Conv2d(320, 160, 3, 1, 1, ), nn.BatchNorm2d(160), nn.GELU(),
                                         nn.UpsamplingBilinear2d(scale_factor=2, ))

        self.conv_l = nn.Conv2d(36, 1, 1)

        self.conv320 = DSConv(1, 1, kernel_size=3, stride=1, padding=1)
        self.conv256 = DSConv(1, 1, kernel_size=3, stride=1, padding=1)
        self.conv160 = DSConv(1, 1, kernel_size=3, stride=1, padding=1)
        self.conv104 = DSConv(1, 1, kernel_size=3, stride=1, padding=1)

        self.asp1 = ASP(24)
        self.asp2 = ASP(32)
        self.asp3 = ASP(96)
        self.asp4 = ASP(320)

        self.usam = USAM()

        self.sigmoid = nn.Sigmoid()
    def forward(self, v, g):
        A1, A2, A3, A4, A5 = self.rgb_pretrained(v)
        A1_d, A2_d, A3_d, A4_d, A5_d = self.depth_pretrained(g)

        f1 = A1 + A1_d
        f2 = A2 + A2_d
        f3 = A3 + A3_d
        f4 = A4 + A4_d
        f5 = A5 + A5_d

        L5 = self.upsample5_l(f5)
        L4 = torch.cat((f4, L5), dim=1)

        L4 = self.upsample4_l(L4)
        L3 = torch.cat((f3, L4), dim=1)

        L3 = self.upsample3_l(L3)
        L2 = torch.cat((f2, L3), dim=1)

        L2 = self.upsample2_l(L2)
        L1 = torch.cat((f1, L2), dim=1)

        L1 = self.upsample1_l(L1)

        out1 = self.conv_l(L1)

        with torch.no_grad():
            p5, _ = torch.max(self.usam(self.asp4(torch.abs(A5 - A5_d))), dim=1, keepdim=True)
            p4, _ = torch.max(self.usam(self.asp3(torch.abs(A4 - A4_d))), dim=1, keepdim=True)
            p3, _ = torch.max(self.usam(self.asp2(torch.abs(A3 - A3_d))), dim=1, keepdim=True)
            p2, _ = torch.max((self.usam(self.asp1(torch.abs(A2 - A2_d)))), dim=1, keepdim=True)

        p5 = self.sigmoid(self.conv320(p5))
        p4 = self.sigmoid(self.conv256(p4))
        p3 = self.sigmoid(self.conv160(p3))
        p2 = self.sigmoid(self.conv104(p2))

        return out1,p5,p4,p3,p2

class FBSNet(nn.Module):
    def __init__(self):
        super(FBSNet, self).__init__()

        self.rgb_swin = SwinTransformer(embed_dim=128, depths=[2,2,18,2], num_heads=[4,8,16,32])
        self.depth_swin = SwinTransformer(embed_dim=128, depths=[2,2,18,2], num_heads=[4,8,16,32])

        self.level = Level(128)
        self.gamma = Gamma(128)

        self.Ccf4 = ChannelGate_sub(128)
        self.Ccf3 = ChannelGate_sub(256)
        self.Ccf2 = ChannelGate_sub(512)
        self.Ccf1 = ChannelGate_sub(1024)

        self.Cf4 = FL(128, 2304)
        self.Cf3 = FL(256, 576)
        self.Cf2 = FL(512, 144)
        self.Cf1 = FL(1024, 36)

        self.up2 = nn.UpsamplingBilinear2d(scale_factor = 2)
        self.up4 = nn.UpsamplingBilinear2d(scale_factor = 4)

        self.conv128_32 = DSConv(128, 32, kernel_size=1, stride=1, padding=0)
        self.conv128_32_1 = DSConv(128, 32, kernel_size=1, stride=1, padding=0)
        self.conv128_32_2 = DSConv(128, 32, kernel_size=1, stride=1, padding=0)
        self.conv128_32_3 = DSConv(128, 32, kernel_size=1, stride=1, padding=0)


        self.conv32 = conv3x3(32, 1)
        self.conv32_1 = conv3x3(32, 1)
        self.conv32_2 = conv3x3(32, 1)
        self.conv32_3 = conv3x3(32, 1)


    def forward(self,x ,d,vg):
        x0 = x
        rgb_list,sp = self.rgb_swin(x)
        depth_list,sp = self.depth_swin(d)

        r4 = rgb_list[0]
        r3 = rgb_list[1]
        r2 = rgb_list[2]
        r1 = rgb_list[3]

        d4 = depth_list[0]
        d3 = depth_list[1]
        d2 = depth_list[2]
        d1 = depth_list[3]

        outg, p5,p4,p3,p2 = self.gamma(x0, vg)

        fr1,fd1 = self.Ccf1(r1 , d1)
        fr2,fd2 = self.Ccf2(r2 , d2)
        fr3,fd3 = self.Ccf3(r3 , d3)
        fr4,fd4 = self.Ccf4(r4 , d4)

        f1 = self.Cf1(fr1 * p5 + fr1 , fd1)
        f2 = self.Cf2(fr2 * p4 + fr2 , fd2)
        f3 = self.Cf3(fr3 * p3 + fr3 , fd3)
        f4 = self.Cf4(fr4 * p2 + fr4 , fd4)

        out, out1, out2,out3 = self.level(f4, f3, f2, f1)

        out = self.conv128_32(out )
        out = self.up4(out)
        sal_out1 = self.conv32(out)

        out1 = self.conv128_32_1(out1)
        out1 = self.up4(out1)
        out1 = self.conv32_1(out1)

        out2 = self.conv128_32_2(out2)
        out2 = self.up4(out2)
        out2 = self.conv32_2(out2)

        out3 = self.conv128_32_3(out3)
        out3 = self.up4(out3)
        out3 = self.conv32_3(out3)


        return sal_out1,out1,out2,out3,outg


    def load_pre(self, pre_model):
        self.rgb_swin.load_state_dict(torch.load(pre_model)['model'],strict=False)
        print(f"RGB SwinTransformer loading pre_model ${pre_model}")
        # self.depth_swin.load_state_dict(torch.load(pre_model)['model'], strict=False)
        # print(f"Depth SwinTransformer loading pre_model ${pre_model}")


if __name__ == '__main__':
    pre_path = '../Pre_train/swin_base_patch4_window7_224.pth'
    a = torch.randn(1,3,224,224)
    b = torch.randn(1,3,224,224)

    net = FBSNet()
    out = net(a,b)

    for i in out:
        print(i.shape)

