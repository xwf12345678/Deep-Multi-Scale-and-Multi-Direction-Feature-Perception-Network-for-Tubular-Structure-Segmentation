# -*- coding: utf-8 -*-
import torch
from torch import nn, cat
from .LKSC import LKSC
import torch.nn.functional as F
from .ConvClass import  EncoderConv,DecoderConv, BasicConv, Scale, DSConv3x3, convbnrelu
class SpatiallyFocusedLSKA(nn.Module):
    def __init__(self, dim, large_k_size):
        super().__init__()
        # 使用较大的卷积核进行空间特征提取
        self.conv_spatial_h = nn.Conv2d(dim, dim, kernel_size=(1, large_k_size), padding=(0, large_k_size // 2),
                                        groups=dim)
        self.conv_spatial_v = nn.Conv2d(dim, dim, kernel_size=(large_k_size, 1), padding=(large_k_size // 2, 0),
                                        groups=dim)
        # 后续操作
        self.conv1 = nn.Conv2d(dim, dim, 1)  # 通道重标定

    def forward(self, x):
        attn = self.conv_spatial_h(x)
        attn = self.conv_spatial_v(attn)
        attn = self.conv1(attn)  # 可以是简单的通道重标定
        return x * attn


class ChannelFocusedLSKA(nn.Module):
    def __init__(self, dim, small_k_size):
        super().__init__()
        # 使用较小的卷积核强调局部空间特征
        self.conv_spatial_h = nn.Conv2d(dim, dim, kernel_size=(1, small_k_size), padding=(0, small_k_size // 2),
                                        groups=dim)
        self.conv_spatial_v = nn.Conv2d(dim, dim, kernel_size=(small_k_size, 1), padding=(small_k_size // 2, 0),
                                        groups=dim)
        # 通过更复杂的操作进行通道重标定，例如可以增加卷积层或者使用注意力机制
        self.conv1 = nn.Conv2d(dim, dim * 8, 1)  # 扩大通道，提供更多的参数进行学习
        self.conv2 = nn.Conv2d(dim * 8, dim, 1)  # 重新压缩通道，进行重标定

    def forward(self, x):
        attn = self.conv_spatial_h(x)
        attn = self.conv_spatial_v(attn)
        attn = self.conv1(attn)
        attn = self.conv2(attn)  # 更复杂的通道重标定
        return x * attn


class MSHA(nn.Module):
    def __init__(self, in_channels, out_channels,branch_number = 3):
        super(MSHA, self).__init__()
        self.branch_number = branch_number
        self.out_conv = EncoderConv(in_channels, out_channels)
        self.conv = DSConv3x3(in_channels, in_channels, stride=1)
        self.branches = nn.ModuleList([
            nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1, bias=False),
            nn.Conv2d(in_channels, in_channels, kernel_size=5, padding=2, bias=False),
            nn.Conv2d(in_channels, in_channels, kernel_size=7, padding=3, bias=False),
        ])

        ### ChannelGate
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.fc2 = nn.Conv2d(in_channels, (self.branch_number+1) * in_channels, 1, 1, 0, bias=False)
        self.fuse = convbnrelu(in_channels, in_channels, k=1, s=1, p=0, relu=False)
        ### SpatialGate
        self.SA = SpatiallyFocusedLSKA(in_channels, 17)
        self.CA = ChannelFocusedLSKA(in_channels, 7)
        self.planes = in_channels

    def forward(self, x):
        conv = self.conv(x)
        brs = [branch(conv) for branch in self.branches]
        brs.append(conv)

        gather = sum(brs)
        ### ChannelGate

        d = self.CA(self.gap(gather))
        d = self.fc2(d)
        d = torch.unsqueeze(d, dim=1).view(-1, (self.branch_number+1), self.planes, 1, 1)

        ### SpatialGate
        s = self.SA(gather).unsqueeze(1)

        ### Fuse two gates
        f = d * s
        f = F.softmax(f, dim=1)

        return self.out_conv(self.fuse(sum([brs[i] * f[:, i, ...] for i in range(self.branch_number + 1)])) + x)

class ChannelAttention(nn.Module):
    """Channel attention used in RCAN.
    Args:
        num_feat (int): Channel number of intermediate features.
        squeeze_factor (int): Channel squeeze factor. Default: 16.
    """

    def __init__(self, num_feat, squeeze_factor=8):
        super(ChannelAttention, self).__init__()
        self.attention = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(num_feat, num_feat // squeeze_factor, 1, padding=0),
            nn.GELU(),
            nn.Conv2d(num_feat // squeeze_factor, num_feat, 1, padding=0),
            nn.Sigmoid())

    def forward(self, x):
        y = self.attention(x)
        return x * y


class MSFRU(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(MSFRU, self).__init__()
        hidden_channel = int(in_channel / 4)
        self.conv1 = DSConv3x3(in_channel, hidden_channel, stride=1, dilation=1)
        self.conv2 = DSConv3x3(in_channel, hidden_channel, stride=1, dilation=2)
        self.conv3 = DSConv3x3(in_channel, hidden_channel, stride=1, dilation=4)
        self.conv4 = DSConv3x3(in_channel, hidden_channel, stride=1, dilation=8)
        self.out = EncoderConv(in_channel * 2, out_channel)
        self.CA = ChannelAttention(2 * in_channel)

    def forward(self, x):
        feat1 = self.conv1(x)
        feat2 = self.conv2(x)
        feat3 = self.conv3(x)
        feat4 = self.conv4(x)
        x1 = torch.cat([x, feat1, feat2, feat3, feat4], dim=1)
        x = self.out(self.CA(x1)) + x

        return x


class BFEM(nn.Module):

    def __init__(self, n_feat):
        super(BFEM, self).__init__()

        self.n_feat = n_feat

        # discrimintor

        self.encoder = MSFRU(n_feat, n_feat)

        self.down = nn.AvgPool2d(kernel_size=2)
        self.decoder_low = MSFRU(n_feat, n_feat)
        self.decoder_high = MSFRU(n_feat, n_feat)
        self.alise = MSFRU(n_feat, n_feat)
        self.alise2 = BasicConv(2 * n_feat, n_feat, 1, 1, 0)
        self.down = nn.AvgPool2d(kernel_size=2)

        self.scale_1 = Scale(0.5)
        self.scale_2 = Scale(0.5)
        self.sigmoid = nn.Sigmoid()
        self.compress1 = nn.Conv2d(256, 128, 1)
        self.compress2 = nn.Conv2d(128, 128, 1)

    def forward(self, x):  # (32,128,32,32)

        x2 = self.down(x)  # (32,64,32,32)
        high = x - F.interpolate(x2, size=x.size()[-2:], mode='bilinear', align_corners=True)  # (32,64,64,64)
        x2 = self.decoder_low(x2)  # (32,64,32,32))
        x3 = x2  # (32,64,32,32)

        high1 = self.decoder_high(high)  # (32,64,64,64)
        x4 = F.interpolate(x3, size=x.size()[-2:], mode='bilinear', align_corners=True)  # (32,64,64,64)

        x4 = self.scale_1(x4)
        high1 = self.scale_2(high1)
        concat1 = torch.cat([x4, high1], dim=1)  # (32,128,64,64)

        concat1 = self.compress1(concat1)

        y1 = self.decoder_high(concat1)  # (32,64,64,64)

        # new
        y3 = self.down(x2)
        mid = x2 - F.interpolate(y3, size=x2.size()[-2:], mode='bilinear', align_corners=True)
        y3 = self.decoder_low(y3)
        y3 = F.interpolate(y3, size=x2.size()[-2:], mode='bilinear', align_corners=True)
        y3 = self.scale_1(y3)
        # new

        y2 = self.decoder_low(x2)  # (32,64,32,32)
        # new
        y2 = self.scale_2(y2)
        y2 = self.alise2(torch.cat([y2, y3], dim=1))
        # new
        y2 = self.decoder_low(y2)

        y2 = F.interpolate(y2, size=x.size()[-2:], mode='bilinear', align_corners=True)  # (32,64,64,64)
        y2 = self.scale_2(y2)
        y1 = self.scale_1(y1)

        sig = self.compress2(x)
        x_sig = x * sig
        out = self.alise2(torch.cat([y2, y1], dim=1))
        out = self.alise(out) + x_sig
        return out

class M2PNet(nn.Module):

    def __init__(
            self,
            n_channels,
            n_classes,
            kernel_size,
            extend_scope,
            if_offset,
            device,
            number,
            dim,
    ):
        """
        Our DSCNet
        :param n_channels: input channel
        :param n_classes: output channel
        :param kernel_size: the size of kernel
        :param extend_scope: the range to expand (default 1 for this method)
        :param if_offset: whether deformation is required, if it is False, it is the standard convolution kernel
        :param device: set on gpu
        :param number: basic layer numbers
        :param dim:
        """
        super(M2PNet, self).__init__()
        self.device = device
        self.kernel_size = kernel_size
        self.extend_scope = extend_scope
        self.if_offset = if_offset
        self.relu = nn.ReLU(inplace=True)
        self.number = number
        """
        The three contributions proposed in our paper are relatively independent. 
        In order to facilitate everyone to use them separately, 
        we first open source the network part of DSCNet. 
        <dim> is a parameter used by multiple templates, 
        which we will open source in the future ...
        """
        self.dim = dim  # This version dim is set to 1 by default, referring to a group of x-axes and y-axes
        """
        Here is our framework. Since the target also has non-tubular structure regions, 
        our designed model also incorporates the standard convolution kernel, 
        for fairness, we also add this operation to compare with other methods (like: Deformable Convolution).
        """
        self.conv00 = EncoderConv(n_channels, self.number)
        self.conv0x = LKSC(
            n_channels,
            self.number,
            self.kernel_size,
            self.extend_scope,
            0,
            self.if_offset,
            self.device,
        )
        self.conv0y = LKSC(
            n_channels,
            self.number,
            self.kernel_size,
            self.extend_scope,
            1,
            self.if_offset,
            self.device,
        )
        self.conv1 = MSHA(3 * self.number, self.number)
        self.conv20 = EncoderConv(self.number, 2 * self.number)
        self.conv2x = LKSC(
            self.number,
            2 * self.number,
            self.kernel_size,
            self.extend_scope,
            0,
            self.if_offset,
            self.device,
        )
        self.conv2y = LKSC(
            self.number,
            2 * self.number,
            self.kernel_size,
            self.extend_scope,
            1,
            self.if_offset,
            self.device,
        )
        self.conv3 = MSHA(6 * self.number, 2 * self.number)
        self.conv40 = EncoderConv(2 * self.number, 4 * self.number)
        self.conv4x = LKSC(
            2 * self.number,
            4 * self.number,
            self.kernel_size,
            self.extend_scope,
            0,
            self.if_offset,
            self.device,
        )
        self.conv4y = LKSC(
            2 * self.number,
            4 * self.number,
            self.kernel_size,
            self.extend_scope,
            1,
            self.if_offset,
            self.device,
        )
        self.conv5 = MSHA(12 * self.number, 4 * self.number)
        self.conv60 = EncoderConv(4 * self.number, 8 * self.number)
        self.conv6x = LKSC(
            4 * self.number,
            8 * self.number,
            self.kernel_size,
            self.extend_scope,
            0,
            self.if_offset,
            self.device,
        )
        self.conv6y = LKSC(
            4 * self.number,
            8 * self.number,
            self.kernel_size,
            self.extend_scope,
            1,
            self.if_offset,
            self.device,
        )
        self.conv7 = MSHA(24 * self.number, 8 * self.number)
        self.Enhance = nn.Sequential(BFEM(8 * self.number), BFEM(8 * self.number), BFEM(8 * self.number),
                                 BFEM(8 * self.number))
        self.conv120 = DecoderConv(12 * self.number, 4 * self.number)
        self.conv12x = LKSC(
            12 * self.number,
            4 * self.number,
            self.kernel_size,
            self.extend_scope,
            0,
            self.if_offset,
            self.device,
        )
        self.conv12y = LKSC(
            12 * self.number,
            4 * self.number,
            self.kernel_size,
            self.extend_scope,
            1,
            self.if_offset,
            self.device,
        )
        self.conv13 = MSHA(12 * self.number, 4 * self.number)

        self.conv140 = DecoderConv(6 * self.number, 2 * self.number)
        self.conv14x = LKSC(
            6 * self.number,
            2 * self.number,
            self.kernel_size,
            self.extend_scope,
            0,
            self.if_offset,
            self.device,
        )
        self.conv14y = LKSC(
            6 * self.number,
            2 * self.number,
            self.kernel_size,
            self.extend_scope,
            1,
            self.if_offset,
            self.device,
        )
        self.conv15 = MSHA(6 * self.number, 2 * self.number)

        self.conv160 = DecoderConv(3 * self.number, self.number)
        self.conv16x = LKSC(
            3 * self.number,
            self.number,
            self.kernel_size,
            self.extend_scope,
            0,
            self.if_offset,
            self.device,
        )
        self.conv16y = LKSC(
            3 * self.number,
            self.number,
            self.kernel_size,
            self.extend_scope,
            1,
            self.if_offset,
            self.device,
        )

        self.MSFRU1 = MSFRU(self.number,self.number)
        self.MSFRU2 = MSFRU(2*self.number,2*self.number)
        self.MSFRU3 = MSFRU(4*self.number,4*self.number)
        self.conv17 = MSHA(3 * self.number, self.number)
        self.out_conv = nn.Conv2d(self.number, n_classes, 1)
        self.maxpooling = nn.MaxPool2d(2)
        self.up = nn.Upsample(scale_factor=2,
                              mode="bilinear",
                              align_corners=True)
        self.sigmoid = nn.Sigmoid()
        self.softmax = nn.Softmax(dim=1)
    def forward(self, x):
        x_00_0 = self.conv00(x)
        x_0x_0 = self.conv0x(x)
        x_0y_0 = self.conv0y(x)
        x_0_1 = self.MSFRU1(self.conv1(torch.cat([x_00_0, x_0x_0, x_0y_0], dim=1)))
        x = self.maxpooling(x_0_1)
        x_20_0 = self.conv20(x)
        x_2x_0 = self.conv2x(x)
        x_2y_0 = self.conv2y(x)
        x_1_1 = self.MSFRU2(self.conv3(torch.cat([x_20_0, x_2x_0, x_2y_0], dim=1)))
        # block2
        x = self.maxpooling(x_1_1)
        x_40_0 = self.conv40(x)
        x_4x_0 = self.conv4x(x)
        x_4y_0 = self.conv4y(x)
        x_2_1 = self.MSFRU3(self.conv5(torch.cat([x_40_0, x_4x_0, x_4y_0], dim=1)))
        # block
        x = self.maxpooling(x_2_1)
        x_60_0 = self.conv60(x)
        x_6x_0 = self.conv6x(x)
        x_6y_0 = self.conv6y(x)
        x_3_1 = self.conv7(torch.cat([x_60_0, x_6x_0, x_6y_0], dim=1))
        x_3_1 = self.Enhance(x_3_1)
        # block4
        x = self.up(x_3_1)
        x_120_2 = self.conv120(cat([x, x_2_1], dim=1))
        x_12x_2 = self.conv12x(cat([x, x_2_1], dim=1))
        x_12y_2 = self.conv12y(cat([x, x_2_1], dim=1))
        x_2_3 = self.MSFRU3(self.conv13(torch.cat([x_120_2, x_12x_2, x_12y_2], dim=1)))
        # block5
        x = self.up(x_2_3)
        x_140_2 = self.conv140(cat([x, x_1_1], dim=1))
        x_14x_2 = self.conv14x(cat([x, x_1_1], dim=1))
        x_14y_2 = self.conv14y(cat([x, x_1_1], dim=1))
        x_1_3 = self.MSFRU2(self.conv15(torch.cat([x_140_2, x_14x_2, x_14y_2], dim=1)))
        # block6
        x = self.up(x_1_3)
        x_160_2 = self.conv160(cat([x, x_0_1], dim=1))
        x_16x_2 = self.conv16x(cat([x, x_0_1], dim=1))
        x_16y_2 = self.conv16y(cat([x, x_0_1], dim=1))
        x_0_3 = self.MSFRU1(self.conv17(torch.cat([x_160_2, x_16x_2, x_16y_2], dim=1)))
        out = self.out_conv(x_0_3)
        out = self.sigmoid(out)
        return out