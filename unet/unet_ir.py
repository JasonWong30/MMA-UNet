""" Full assembly of the parts to form the complete network """

from .unet_parts2 import *
from utils.Sobelxy import Sobelxy
from math import exp

class SELayer(nn.Module):
    def __init__(self, channel, reduction=16):
        super(SELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )
    def forward(self, x, vis_fea):
        b, c, _, _ = vis_fea.size()
        y = self.avg_pool(vis_fea).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)

#红外光的UNet
class UNet(nn.Module):
    def __init__(self, in_channels, out_channels, bilinear=False):
        super(UNet, self).__init__()
        self.in_channels = in_channels
        self.bilinear = bilinear

        self.inc = (DoubleConv(in_channels, 64))
        self.down1 = (Down(64, 128))
        self.down2 = (Down(128, 256))
        self.down3 = (Down(256, 512))
        factor = 2 if bilinear else 1
        self.down4 = (Down(512, 1024 // factor))
        self.up1 = (Up(1024, 512 // factor, bilinear))
        self.up2 = (Up(512, 256 // factor, bilinear))
        self.up3 = (Up(256, 128 // factor, bilinear))
        self.up4 = (Up(128, 64, bilinear))
        self.outc = (OutConv(64, out_channels))

        self.attention1 = SELayer(64)
        self.attention2 = SELayer(128)
        self.attention3 = SELayer(256)
        self.attention4 = SELayer(512)
        self.attention5 = SELayer(1024)


    def forward(self, x, fea_vis):

        fea_vis_x1, fea_vis_x2, fea_vis_x3, fea_vis_x4, fea_vis_x5 = fea_vis
        x1 = self.inc(x)
        x1 = self.attention1(x1, fea_vis_x1)
        x2 = self.down1(x1)
        x2 = self.attention2(x2, fea_vis_x2)
        x3 = self.down2(x2)
        x3 = self.attention3(x3, fea_vis_x3)
        x4 = self.down3(x3)
        x4 = self.attention4(x4, fea_vis_x4)
        x5 = self.down4(x4)
        x5 = self.attention5(x5, fea_vis_x5)

        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        logits = self.outc(x)

        return logits, [x1, x2, x3, x4, x5]
