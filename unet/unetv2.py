""" Full assembly of the parts to form the complete network """
import math
from .unet_parts import *
from utils.Sobelxy import Sobelxy
from math import exp
from unet.fs_loss import Fusionloss
from .scSE import ChannelSpatialSELayer


class Fusion_ADD(torch.nn.Module):
    def forward(self, en_ir, en_vi):
        temp = en_ir + en_vi
        return temp

class Fusion_AVG(torch.nn.Module):
    def forward(self, en_ir, en_vi):
        temp = (en_ir + en_vi) / 2
        return temp

class Fusion_MAX(torch.nn.Module):
    def forward(self, en_ir, en_vi):
        temp = torch.max(en_ir, en_vi)
        return temp

class ChannelAttentionConv(nn.Module):
    """ 通道注意力机制——一维卷积版本
    """
    def __init__(self, in_channel, gamma = 2, b = 1):
        """ 初始化
            - channel: 输入特征图的通道数
            - gamma: 公式中的两个系数
            - b: 公式中的两个系数
        """
        super(ChannelAttentionConv, self).__init__()
        # 根据输入通道数自适应调整卷积核大小
        kernel_size = int(abs((math.log(in_channel, 2) + b) / gamma))
        # 如果卷积核大小是奇数
        kernel_size = kernel_size if kernel_size % 2 else kernel_size + 1
        # 池化
        self.avg_pooling = nn.AdaptiveAvgPool2d(1)
        self.max_pooling = nn.AdaptiveMaxPool2d(1)
        # 一维卷积
        self.conv = nn.Conv1d(1, 1, kernel_size = kernel_size,
                              padding = (kernel_size - 1) // 2, bias = False)
        self.sigmoid = nn.Sigmoid()
        # self.conv1 = nn.Conv2d(in_channel, in_channel//2, kernel_size=1,  bias=False)

    def forward(self, X):
        """ 前向传播
        """
        # 全局池化 [b,c,h,w]==>[b,c,1,1]
        avg_x = self.avg_pooling(X)
        max_x = self.max_pooling(X)
        # [b,c,1,1]==>[b,1,c] =1D卷积=> [b,1,c]==>[b,c,1,1]
        avg_out = self.conv(avg_x.squeeze(-1).transpose(-1, -2)).transpose(-1, -2).unsqueeze(-1)
        max_out = self.conv(max_x.squeeze(-1).transpose(-1, -2)).transpose(-1, -2).unsqueeze(-1)
        # 权值归一化
        v = self.sigmoid(avg_out + max_out)
        # 输入特征图和通道权重相乘 [b,c,h,w]
        return  X * v

class Down(nn.Module):
    """Downscaling with maxpool then double conv"""
    def __init__(self, in_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.AvgPool2d(2), #原本max
            nn.Conv2d(in_channels, in_channels * 2, kernel_size=1)
        )
        # self.conv = nn.Conv2d(in_channels, in_channels * 2, kernel_size=1)

        # self.conv = nn.Conv2d(in_channels=in_channels, out_channels=in_channels * 2, kernel_size=3, stride=2, padding=1)

    def forward(self, x):

        return self.maxpool_conv(x)


class Fusion_Encoder(nn.Module):
    def __init__(self):
        super(Fusion_Encoder, self).__init__()
        dim= [64, 128, 256, 512, 1024]
        heads = [8, 8, 8]

        self.down1 = Down(dim[0])
        self.down2 = Down(dim[1])
        self.down3 = Down(dim[2])
        self.down4 = Down(dim[3])

        self.ca1 = ChannelAttentionConv(dim[1])
        self.ca2 = ChannelAttentionConv(dim[2])
        # self.ca1 = ChannelSpatialSELayer(dim[1])
        # self.ca2 = ChannelSpatialSELayer(dim[2])

        # self.ca3 = ChannelAttentionConv(dim[3])
        # self.ca4 = ChannelAttentionConv(dim[4])
        self.ca3 = ChannelSpatialSELayer(dim[3])
        self.ca4 = ChannelSpatialSELayer(dim[4])

    def Asy_fusion1(self, fea_vi, fea_ir):# 小的放前面，大的放后面
        fea_vi = self.down1(fea_vi)
        add_fea = fea_ir + fea_vi
        res = self.ca1(add_fea)
        return res
    def Asy_fusion2(self, fea_vi, fea_ir):#小的放前面，大的放后面
        fea_vi = self.down2(fea_vi)
        add_fea = fea_ir + fea_vi
        res = self.ca2(add_fea)
        return res
    def Asy_fusion3(self, fea_vi, fea_ir):#小的放前面，大的放后面
        fea_vi = self.down3(fea_vi)

        add_fea = fea_ir + fea_vi
        # add_fea = torch.cat([fea_ir,fea_vi], dim=1)
        res = self.ca3(add_fea)
        return res
    def Asy_fusion4(self, fea_vi, fea_ir):#小的放前面，大的放后面
        fea_vi = self.down4(fea_vi)
        add_fea = fea_ir + fea_vi
        res = self.ca4(add_fea)
        return res

    def forward(self, Encoder_ir, Encoder_vi):
        f0_0 = self.Asy_fusion1(Encoder_vi[0], Encoder_ir[1])  # -可见0(64)- 红外1(128)
        f1_0 = self.Asy_fusion2(Encoder_vi[1], Encoder_ir[2])  # -可见1(128)- 红外2(256)
        f2_0 = self.Asy_fusion3(Encoder_vi[2], Encoder_ir[3])  #可见2(256)- 红外3(512)
        f3_0 = self.Asy_fusion4(Encoder_vi[3], Encoder_ir[4])  #-可见3(512)- 红外3(1024)

        return [f0_0, f1_0, f2_0, f3_0]

class Upsample(nn.Module):
    def __init__(self, n_feat):
        super(Upsample, self).__init__()
        self.body = nn.Sequential(nn.Conv2d(n_feat, n_feat * 2, kernel_size=3, stride=1, padding=1, bias=False), nn.PixelShuffle(2))

    def forward(self, x):
        return self.body(x)


class Up(nn.Module):
    """Upscaling then double conv"""
    def __init__(self, in_channels, out_channels, bilinear=True):
        super().__init__()
        # if bilinear, use the normal convolutions to reduce the number of channels
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels, in_channels // 2)
            # self.conv =  Restormer_CNN_block(in_channels, out_channels)
        else:
            # self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
            self.up = Upsample(in_channels)
            self.conv = DoubleConv(in_channels, out_channels)
            # self.conv = Restormer_CNN_block(in_channels, out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        # input is CHW
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        # if you have padding issues, see
        # https://github.com/HaiyongJiang/U-Net-Pytorch-Unstructured-Buggy/commit/0e854509c2cea854e247a9c615f175f76fbb2e3a
        # https://github.com/xiaopeng-liao/Pytorch-UNet/commit/8ebac70e633bac59fc22bb5195e513d5832fb3bd
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)

class Decoder(nn.Module):
    def __init__(self, bilinear=False):
        super(Decoder, self).__init__()
        self.bilinear = bilinear
        factor = 2 if self.bilinear else 1
        self.up1 = (Up(1024, 512 // factor, bilinear))
        self.up2 = (Up(512, 256 // factor, bilinear))
        self.up3 = (Up(256, 128 // factor, bilinear))
        self.up4 = (Upsample(128))
        self.outc = (OutConv(64, 3))

    def forward(self, fusion_fea):
        x = self.up1(fusion_fea[3], fusion_fea[2])
        x = self.up2(x, fusion_fea[1])
        x = self.up3(x, fusion_fea[0])
        x = self.up4(x)
        logits = self.outc(x)

        return logits

class Unet(nn.Module):
    def __init__(self, training=True):
        super(Unet, self).__init__()
        self.fusion_encoder = Fusion_Encoder()
        self.decoder = Decoder()
        self.fusionloss = Fusionloss()
        self.training = training

    def forward_loss(self, fused_img, ir, vis):
        loss_ssim, loss_in, loss_grad = self.fusionloss(vis, ir, fused_img)
        loss = 10 * loss_grad + 0.5 * loss_ssim + loss_in
        loss = loss.mean()
        return loss

    def forward(self, Encoder_ir, Encoder_vi, ir, vis):
        fusion_fea = self.fusion_encoder(Encoder_ir, Encoder_vi)
        logits = self.decoder(fusion_fea)
        if self.training:
            loss = self.forward_loss(logits, ir, vis)
        else:
            loss = 0
        # return logits, loss, fusion_fea
        return logits, loss