import math

import torch
import torch.nn as nn
import torch.nn.functional as F


def autopad(k, p=None, d=1):
    if d > 1:
        k = d * (k - 1) + 1 if isinstance(k, int) else [d * (x - 1) + 1 for x in k]
    if p is None:
        p = k // 2 if isinstance(k, int) else [x // 2 for x in k]
    return p


class Conv(nn.Module):
    default_act = nn.GELU()

    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, d=1, act=True):
        super().__init__()
        self.conv = nn.Conv2d(c1, c2, k, s, autopad(k, p, d), groups=g, dilation=d, bias=False)
        self.bn = nn.BatchNorm2d(c2, eps=0.001, momentum=0.03, affine=True, track_running_stats=True)
        self.act = self.default_act if act is True else act if isinstance(act, nn.Module) else nn.Identity()

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))


class DWConv(Conv):
    def __init__(self, c1, c2, k=1, s=1, d=1, act=True):
        super().__init__(c1, c2, k, s, g=math.gcd(c1, c2), d=d, act=act)


class CMRF(nn.Module):
    def __init__(self, c1, c2, n=8, shortcut=True, e=0.5):
        super().__init__()
        self.n = n
        self.c = int(c2 * e / self.n)
        self.add = shortcut and c1 == c2

        self.pwconv1 = Conv(c1, c2 // self.n, 1, 1)
        self.pwconv2 = Conv(c2 // 2, c2, 1, 1)
        self.blocks = nn.ModuleList(DWConv(self.c, self.c, k=3, act=False) for _ in range(n - 1))

    def forward(self, x):
        residual = x
        x = self.pwconv1(x)

        parts = [x[:, 0::2, :, :], x[:, 1::2, :, :]]
        parts.extend(block(parts[-1]) for block in self.blocks)
        parts[0] = parts[0] + parts[1]
        parts.pop(1)

        y = torch.cat(parts, dim=1)
        y = self.pwconv2(y)
        return residual + y if self.add else y


class UNetEncoder(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.cmrf = CMRF(in_channels, out_channels)
        self.downsample = nn.MaxPool2d(kernel_size=2, stride=2)

    def forward(self, x):
        x = self.cmrf(x)
        return self.downsample(x), x


class UNetDecoder(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.cmrf = CMRF(in_channels, out_channels)

    def forward(self, x, skip_connection):
        x = F.interpolate(x, scale_factor=2, mode="bicubic", align_corners=False)
        x = torch.cat([x, skip_connection], dim=1)
        return self.cmrf(x)


class TinyUNet(nn.Module):
    def __init__(self, in_channels=3, num_classes=2):
        super().__init__()
        in_filters = [192, 384, 768, 1024]
        out_filters = [64, 128, 256, 512]

        self.encoder1 = UNetEncoder(in_channels, 64)
        self.encoder2 = UNetEncoder(64, 128)
        self.encoder3 = UNetEncoder(128, 256)
        self.encoder4 = UNetEncoder(256, 512)

        self.decoder4 = UNetDecoder(in_filters[3], out_filters[3])
        self.decoder3 = UNetDecoder(in_filters[2], out_filters[2])
        self.decoder2 = UNetDecoder(in_filters[1], out_filters[1])
        self.decoder1 = UNetDecoder(in_filters[0], out_filters[0])
        self.final_conv = nn.Conv2d(out_filters[0], num_classes, kernel_size=1)

    def forward(self, x):
        x, skip1 = self.encoder1(x)
        x, skip2 = self.encoder2(x)
        x, skip3 = self.encoder3(x)
        x, skip4 = self.encoder4(x)

        x = self.decoder4(x, skip4)
        x = self.decoder3(x, skip3)
        x = self.decoder2(x, skip2)
        x = self.decoder1(x, skip1)
        return self.final_conv(x)


def tiny_unet(n_classes, input_height=416, input_width=608):
    del input_height, input_width
    return TinyUNet(in_channels=3, num_classes=n_classes)


def resnet50_unet(n_classes, input_height=416, input_width=608):
    # Keep the old factory name so existing imports keep working.
    return tiny_unet(n_classes=n_classes, input_height=input_height, input_width=input_width)
