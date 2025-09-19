# -*- coding: utf-8 -*-
import copy
from typing import Optional, List
import torch
import torch.nn.functional as F
from torch import nn, Tensor
import numpy as np
import time
from transformer_block import get_sinusoid_encoding
from thop import profile, clever_format
from pytorch_wavelets import DWTForward
from pytorch_wavelets import DTCWTForward, DTCWTInverse
from pytorch_wavelets import DWTForward, DWTInverse


class BasicConv2d(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size=3, stride=1, padding=1, dilation=1):
        super(BasicConv2d, self).__init__()
        self.conv = nn.Conv2d(in_planes, out_planes,
                              kernel_size=kernel_size, stride=stride,
                              padding=padding, dilation=dilation, bias=False)
        self.bn = nn.BatchNorm2d(out_planes)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x


class EfficientAttention(nn.Module): # this is multiAttention
    def __init__(self, in_channels, key_channels, head_count, value_channels):
        super().__init__()
        self.in_channels = in_channels
        self.key_channels = key_channels
        self.head_count = head_count
        self.value_channels = value_channels

        self.keys = nn.Linear(self.in_channels, self.key_channels)
        self.queries = nn.Linear(self.in_channels, self.key_channels)
        self.values = nn.Linear(self.in_channels, self.value_channels)
        self.reprojection = nn.Linear(self.value_channels, self.in_channels)

    def forward(self, input_, x_pos_embed):
        B,N,C = input_.size()
        assert C == self.in_channels,"C {} != inchannels {}".format(C, self.in_channels)
        assert input_.shape[1:] == x_pos_embed.shape[1:], "x.shape {} != x_pos_embed.shape {}".format(input_.shape, x_pos_embed.shape)
        keys = self.keys(input_ + x_pos_embed).permute(0, 2, 1) #.reshape((n, self.key_channels, h * w))
        queries = self.queries(input_ + x_pos_embed).permute(0, 2, 1) #.reshape(n, self.key_channels, h * w)
        values = self.values(input_).permute(0, 2, 1)#.reshape((n, self.value_channels, h * w))
        head_key_channels = self.key_channels // self.head_count
        head_value_channels = self.value_channels // self.head_count

        attended_values = []
        for i in range(self.head_count):
            key = F.softmax(keys[
                            :,
                            i * head_key_channels: (i + 1) * head_key_channels,
                            :
                            ], dim=2)
            query = F.softmax(queries[
                              :,
                              i * head_key_channels: (i + 1) * head_key_channels,
                              :
                              ], dim=1)
            value = values[
                    :,
                    i * head_value_channels: (i + 1) * head_value_channels,
                    :
                    ]
            context = key @ value.transpose(1, 2)
            attended_value = context.transpose(1, 2) @ query

            attended_values.append(attended_value)

        aggregated_values = torch.cat(attended_values, dim=1)
        aggregated_values = aggregated_values.transpose(1,2)
        reprojected_value = self.reprojection(aggregated_values)

        return reprojected_value

class Multi_EfficientAttention(nn.Module): # this is multiAttention
    def __init__(self, x_channels, y_channels, key_channels, head_count, value_channels):
        super().__init__()
        self.x_channels = x_channels
        self.y_channels = y_channels
        self.key_channels = key_channels
        self.head_count = head_count
        self.value_channels = value_channels

        self.queries = nn.Linear(self.x_channels, self.key_channels)
        self.keys = nn.Linear(self.y_channels, self.key_channels)
        self.values = nn.Linear(self.y_channels, self.value_channels)
        self.reprojection = nn.Linear(self.value_channels, self.x_channels)


    def forward(self, x, y, x_pos_embed, y_pos_embed):
        Bx,Nx,Cx = x.size()
        assert Cx == self.x_channels,"Cx {} != inchannels {}".format(Cx, self.x_channels)
        assert x.shape[1:] == x_pos_embed.shape[1:], "x.shape {} != x_pos_embed.shape {}".format(x.shape, x_pos_embed.shape)
        By, Ny, Cy = y.size()
        assert Cy == self.y_channels, "Cy {} != inchannels {}".format(Cy, self.y_channels)
        assert y.shape[1:] == y_pos_embed.shape[1:], "y.shape {} != y_pos_embed.shape {}".format(y.shape, y_pos_embed.shape)

        queries = self.queries(x + x_pos_embed).permute(0, 2, 1) #.reshape(n, self.key_channels, h * w)
        keys = self.keys(y + y_pos_embed).permute(0, 2, 1)  # .reshape((n, self.key_channels, h * w))
        values = self.values(y).permute(0, 2, 1)#.reshape((n, self.value_channels, h * w))
        head_key_channels = self.key_channels // self.head_count
        head_value_channels = self.value_channels // self.head_count

        attended_values = []
        for i in range(self.head_count):
            key = F.softmax(keys[
                            :,
                            i * head_key_channels: (i + 1) * head_key_channels,
                            :
                            ], dim=2)
            query = F.softmax(queries[
                              :,
                              i * head_key_channels: (i + 1) * head_key_channels,
                              :
                              ], dim=1)
            value = values[
                    :,
                    i * head_value_channels: (i + 1) * head_value_channels,
                    :
                    ]

            context = key @ value.transpose(1, 2)

            attended_value = context.transpose(1, 2) @ query
            attended_values.append(attended_value)
        aggregated_values = torch.cat(attended_values, dim=1)
        aggregated_values = aggregated_values.transpose(1, 2)
        reprojected_value = self.reprojection(aggregated_values)

        return reprojected_value

class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.ReLU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x

class Block(nn.Module):
    def __init__(self,x_channels, nx, y_channels, ny):
        super(Block, self).__init__()
        self.x_channels = x_channels
        self.y_channels = y_channels
        self.nx = nx
        self.ny = ny
        self.x_pos_embed = nn.Parameter(data=get_sinusoid_encoding(n_position=nx, d_hid=x_channels), requires_grad=False)
        self.y_pos_embed = nn.Parameter(data=get_sinusoid_encoding(n_position=ny, d_hid=y_channels), requires_grad=False)
        self.x2_pos_embed = nn.Parameter(data=get_sinusoid_encoding(n_position=nx, d_hid=x_channels),requires_grad=False)
        self.y2_pos_embed = nn.Parameter(data=get_sinusoid_encoding(n_position=ny, d_hid=y_channels),
                                        requires_grad=False)
        self.x3_pos_embed = nn.Parameter(data=get_sinusoid_encoding(n_position=nx, d_hid=x_channels),
                                         requires_grad=False)
        self.y3_pos_embed = nn.Parameter(data=get_sinusoid_encoding(n_position=ny, d_hid=y_channels),
                                         requires_grad=False)
        self.q_pos_embed = nn.Parameter(data=get_sinusoid_encoding(n_position=nx, d_hid=x_channels),
                                         requires_grad=False)
        self.q1_pos_embed = nn.Parameter(data=get_sinusoid_encoding(n_position=ny, d_hid=y_channels),
                                         requires_grad=False)

        self.norm_layer = nn.LayerNorm(x_channels)
        # nn.LayerNorm

        self.self_attn = EfficientAttention(x_channels, x_channels, 4, x_channels)
        self.cross_attn = Multi_EfficientAttention(x_channels=x_channels, y_channels=y_channels, key_channels=x_channels, head_count=4, value_channels=x_channels)
        self.mlp = Mlp(in_features=x_channels, hidden_features=x_channels * 4,out_features= x_channels)
    def forward(self,x, y,q):
        x_atten = self.self_attn(x, self.x_pos_embed)
        Osa = self.norm_layer(x + x_atten)
        xy_attn = self.cross_attn(q, Osa, self.q_pos_embed, self.x2_pos_embed)


        Oca = self.norm_layer(xy_attn + Osa)
        Of = self.mlp(Oca)
        Oo = self.norm_layer(Of + Oca)
        #y
        y_atten = self.self_attn( y, self.y_pos_embed)
        ysa = self.norm_layer(y + y_atten)
        yx_attn = self.cross_attn(Oo, ysa, self.q1_pos_embed, self.y2_pos_embed)

        yca = self.norm_layer(yx_attn + ysa)
        yf = self.mlp(yca)
        yO = self.norm_layer(yf + yca)

        #out = Oo + yO

        return yO#out

class RandomMaskingGenerator:
  def __init__(self, mask_ratio):
    self.mask_ratio = mask_ratio

  def __call__(self, x):
    B, num_patches, embed_dim = x.shape  # 从输入张量x获取尺寸
    num_mask = int(self.mask_ratio * num_patches)

    # 生成随机遮罩数组
    mask = np.hstack([
      np.zeros(num_patches - num_mask),
      np.ones(num_mask),
    ])
    np.random.shuffle(mask)
    mask = torch.Tensor(mask).reshape(1, num_patches, 1).expand(B, -1, embed_dim)  # 转换为张量并重塑以匹配输入张量的高度和宽度
    mask = mask.to(x.device)

    # 应用遮罩
    return x * mask

# class BasicConv2d(nn.Module):
#     def __init__(self, in_planes, out_planes, kernel_size=3, stride=1, padding=1, dilation=1):
#         super(BasicConv2d, self).__init__()
#         self.conv = nn.Conv2d(in_planes, out_planes,
#                               kernel_size=kernel_size, stride=stride,
#                               padding=padding, dilation=dilation, bias=False)
#         self.bn = nn.BatchNorm2d(out_planes)
#         self.relu = nn.ReLU(inplace=True)
#
#     def forward(self, x):
#         x = self.conv(x)
#         x = self.bn(x)
#         x = self.relu(x)
#         return x

class TransBasicConv2d(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size=2, stride=2, padding=0, dilation=1,output_padding=0, bias=False):
        super(TransBasicConv2d, self).__init__()
        self.Deconv = nn.ConvTranspose2d(in_planes, out_planes,
                              kernel_size=kernel_size, stride=stride,
                              padding=padding,output_padding= output_padding, dilation=dilation, bias=bias)
        self.bn = nn.BatchNorm2d(out_planes)
        self.relu = nn.ReLU(inplace=True)
        self.inch = in_planes
    def forward(self, x):

        x = self.Deconv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x

# class FFT(nn.Module):
#     def __init__(self,inchannel,outchannel):
#         super().__init__()
#         self.DWT = DTCWTForward(J=3, biort='near_sym_b', qshift='qshift_b')
#         # self.DWT =DTCWTForward(J=3, include_scale=True)
#         self.IWT = DTCWTInverse(biort='near_sym_b', qshift='qshift_b')
#         self.conv1 = BasicConv2d(outchannel, outchannel)
#         self.conv2 = BasicConv2d(inchannel, outchannel)
#         self.conv3 = BasicConv2d(outchannel, outchannel)
#         self.change = TransBasicConv2d(outchannel, outchannel)
#
#     def forward(self, x, y):
#         y = self.conv2(y)
#         Xl, Xh = self.DWT(x)
#         Yl, Yh = self.DWT(y)
#         x_y = self.conv1(Xl) + self.conv1(Yl)
#
#         x_m = self.IWT((x_y, Xh))
#         y_m = self.IWT((x_y, Yh))
#
#         out = self.conv3(x_m + y_m) + x + y
#         return out


class IIA(nn.Module):
    def __init__(self, inchannels,num_blocks, x_channels, nx, y_channels, ny):
        super(IIA, self).__init__()
        assert x_channels == y_channels, "channel_X-{} should same as channel_Y-{}".format(x_channels, y_channels)
        self.num_blocks = num_blocks
        self.blocks = nn.ModuleList([
            Block(x_channels=x_channels, nx=nx, y_channels=y_channels, ny=ny)for i in range(self.num_blocks)
        ])
        self.norm =nn.LayerNorm(x_channels)
        self.conv = nn.Sequential(nn.Conv2d(inchannels, x_channels, 1), nn.BatchNorm2d(x_channels), nn.ReLU(True))
        self.convd = nn.Sequential(nn.Conv2d(inchannels, y_channels, 1), nn.BatchNorm2d(y_channels), nn.ReLU(True))

        #self.Cf = FFT(x_channels, x_channels)
        self.xmask = RandomMaskingGenerator(mask_ratio=0.5)
        self.ymask = RandomMaskingGenerator(mask_ratio=0.5)
    def forward(self,x,y):
        x0 = self.conv(x)#.flatten(2).permute(0, 2, 1)
        y0 = self.convd(y)#.flatten(2).permute(0, 2, 1)
        x = x0.flatten(2).permute(0, 2, 1)
        y = y0.flatten(2).permute(0, 2, 1)

        B, N, C = x.size()
        #print(x.shape)
        # q = self.Cf(x0,y0).flatten(2).permute(0, 2, 1)
        q = nn.Parameter(torch.zeros(B, N, C, device=x.device), requires_grad=True)#.permute(0, 2, 1)

        for block in self.blocks:
            x = block(x, y,q)
        x = self.norm(x)
        x = x.transpose(1, 2).reshape(B, C, int(np.sqrt(N)), int(np.sqrt(N)))
        return x

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

class WT(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(WT, self).__init__()
        self.wt = DWTForward(J=1, mode='zero', wave='haar')

        self.conv_bn_relu = nn.Sequential(
                                    nn.Conv2d(in_ch*3, in_ch, kernel_size=1, stride=1),
                                    nn.BatchNorm2d(in_ch),
                                    nn.ReLU(inplace=True),
                                    )
        self.outconv_bn_relu_L = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=1, stride=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )
        self.outconv_bn_relu_H = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=1, stride=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

        self.usam = USAM(kernel_size=3, padding=1, polish=False)

    def forward(self, x):
        yL, yH = self.wt(x)
        y_HL = self.usam(yH[0][:,:,0,::])
        y_LH = self.usam(yH[0][:,:,1,::])
        y_HH = self.usam(yH[0][:,:,2,::])

        yH = torch.cat([y_HL, y_LH, y_HH], dim=1)
        yH = self.conv_bn_relu(yH)

        yL = self.outconv_bn_relu_L(yL)
        yH = self.outconv_bn_relu_H(yH)

        return yL,yH

class FL1(nn.Module):#两模态融合层
    def __init__(self, dim):
        super(FL1, self).__init__()
        self.WTR = WT(in_ch=dim, out_ch=dim // 2)
        self.WTD = WT(in_ch=dim, out_ch=dim//2)
        self.conv1 = BasicConv2d(dim//2, dim//2)
        self.conv2 = BasicConv2d(dim//2, dim//2)
        self.conv3 = BasicConv2d(dim, dim)
        self.up2 = nn.UpsamplingBilinear2d(scale_factor=2)
    def forward(self, r, d):
        rL,rH = self.WTR(r)
        dL,dH = self.WTD(d)

        L = self.conv1(rL) +  self.conv2(dL)

        r1 = torch.cat([L,rH], dim=1)
        d1 = torch.cat([L, dH], dim=1)

        out = self.conv3(self.up2(r1 + d1)) + r + d
        return out

# class FL(nn.Module):#两模态融合层
#     def __init__(self, dim,nn):
#         super(FL, self).__init__()
#
#         # J为分解的层次数,wave表示使用的变换方法
#         self.xfm = DWTForward(J=1, mode='zero', wave='haar')  # Accepts all wave types available to PyWavelets
#         self.ifm = DWTInverse(mode='zero', wave='haar')
#
#         self.WTR = WT(in_ch=dim, out_ch=dim // 2)
#         self.WTD = WT(in_ch=dim, out_ch=dim // 2)
#
#         self.cf = IIA(dim//2 ,num_blocks=3,x_channels=dim//2, nx=nn, y_channels=dim//2, ny=nn)
#
#         #self.usam = USAM()
#
#         # self.WTR = WT(in_ch=dim, out_ch=dim // 2)
#         # self.WTD = WT(in_ch=dim, out_ch=dim//2)
#         # self.conv1 = BasicConv2d(dim, dim)
#         # self.conv2 = BasicConv2d(dim, dim)
#         # self.conv3 = BasicConv2d(dim, dim)
#         #self.up2 = nn.UpsamplingBilinear2d(scale_factor=2)
#         self.conv3 = BasicConv2d(dim, dim)
#         self.up2 = torch.nn.UpsamplingBilinear2d(scale_factor=2)
#     def forward(self, r, d):
#
#         rl, rh = self.WTR(r)
#         dl, dh = self.WTD(d)
#
#
#         L = self.cf(rl,dl)
#
#         r1 = torch.cat([L, rh], dim=1)
#         d1 = torch.cat([L, dh], dim=1)
#
#         out = self.conv3(self.up2(r1 + d1)) + r + d
#
#
#         return out#,L#,out#rh[0][:,:,0,::]

class FL(nn.Module):#两模态融合层
    def __init__(self, dim,nn):
        super(FL, self).__init__()

        # J为分解的层次数,wave表示使用的变换方法
        self.xfm = DWTForward(J=1, mode='zero', wave='haar')  # Accepts all wave types available to PyWavelets
        self.ifm = DWTInverse(mode='zero', wave='haar')

        self.DWT = DTCWTForward(J=3, biort='near_sym_b', qshift='qshift_b')
        self.IWT = DTCWTInverse(biort='near_sym_b', qshift='qshift_b')

        self.WTR = WT(in_ch=dim, out_ch=dim // 2)
        self.WTD = WT(in_ch=dim, out_ch=dim // 2)

        self.cf = IIA(dim ,num_blocks=3,x_channels=dim, nx=nn, y_channels=dim, ny=nn)

        #self.usam = USAM()

        # self.WTR = WT(in_ch=dim, out_ch=dim // 2)
        # self.WTD = WT(in_ch=dim, out_ch=dim//2)
        # self.conv1 = BasicConv2d(dim, dim)
        # self.conv2 = BasicConv2d(dim, dim)
        # self.conv3 = BasicConv2d(dim, dim)
        #self.up2 = nn.UpsamplingBilinear2d(scale_factor=2)
        self.conv3 = BasicConv2d(dim, dim)
        self.up2 = torch.nn.UpsamplingBilinear2d(scale_factor=2)
    def forward(self, r, d):
        # rl, rh = self.WTR(r)
        # dl, dh = self.WTD(d)
        rl, rh = self.xfm(r)
        dl, dh = self.xfm(d)
        #print(r.shape,rl.shape)
        L = self.cf(rl,dl)

        r1 = self.ifm((L, rh))#torch.cat([L, rh], dim=1)
        d1 = self.ifm((L, dh))#torch.cat([L, dh], dim=1)
        #print(r1.shape)
        out = self.conv3(r1 + d1) + r + d

        return out