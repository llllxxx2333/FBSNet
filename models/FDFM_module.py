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

        y_atten = self.self_attn( y, self.y_pos_embed)
        ysa = self.norm_layer(y + y_atten)
        yx_attn = self.cross_attn(Oo, ysa, self.q1_pos_embed, self.y2_pos_embed)

        yca = self.norm_layer(yx_attn + ysa)
        yf = self.mlp(yca)
        yO = self.norm_layer(yf + yca)

        return yO

class ATT(nn.Module):
    def __init__(self, inchannels,num_blocks, x_channels, nx, y_channels, ny):
        super(ATT, self).__init__()
        assert x_channels == y_channels, "channel_X-{} should same as channel_Y-{}".format(x_channels, y_channels)
        self.num_blocks = num_blocks
        self.blocks = nn.ModuleList([
            Block(x_channels=x_channels, nx=nx, y_channels=y_channels, ny=ny)for i in range(self.num_blocks)
        ])
        self.norm =nn.LayerNorm(x_channels)
        self.conv = nn.Sequential(nn.Conv2d(inchannels, x_channels, 1), nn.BatchNorm2d(x_channels), nn.ReLU(True))
        self.convd = nn.Sequential(nn.Conv2d(inchannels, y_channels, 1), nn.BatchNorm2d(y_channels), nn.ReLU(True))

    def forward(self,x,y):
        x0 = self.conv(x)
        y0 = self.convd(y)
        x = x0.flatten(2).permute(0, 2, 1)
        y = y0.flatten(2).permute(0, 2, 1)

        B, N, C = x.size()

        q = nn.Parameter(torch.zeros(B, N, C, device=x.device), requires_grad=True)

        for block in self.blocks:
            x = block(x, y,q)
        x = self.norm(x)
        x = x.transpose(1, 2).reshape(B, C, int(np.sqrt(N)), int(np.sqrt(N)))
        return x

class FDFM(nn.Module):#两模态融合层
    def __init__(self, dim,nn):
        super(FDFM, self).__init__()

        self.xfm = DWTForward(J=1, mode='zero', wave='haar')
        self.ifm = DWTInverse(mode='zero', wave='haar')

        self.cf = ATT(dim ,num_blocks=3,x_channels=dim, nx=nn, y_channels=dim, ny=nn)


        self.conv3 = BasicConv2d(dim, dim)
        self.up2 = torch.nn.UpsamplingBilinear2d(scale_factor=2)
    def forward(self, r, d):

        rl, rh = self.xfm(r)
        dl, dh = self.xfm(d)

        L = self.cf(rl,dl)

        r1 = self.ifm((L, rh))
        d1 = self.ifm((L, dh))

        out = self.conv3(r1 + d1) + r + d

        return out