"""
ResUNet++ 风格分割网络（带 SE、ASPP、注意力解码）。
与 train.py 中的 CE + Dice 配套：输出 logits 形状 (N, n_classes, H, W)。
"""
from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn


class SqueezeExcitation(nn.Module):
    def __init__(self, channel: int, r: int = 8):
        super().__init__()
        hidden = max(1, channel // r)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.net = nn.Sequential(
            nn.Linear(channel, hidden, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, channel, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        b, c, _, _ = inputs.shape
        x = self.pool(inputs).view(b, c)
        x = self.net(x).view(b, c, 1, 1)
        return inputs * x


class StemBlock(nn.Module):
    def __init__(self, in_c: int, out_c: int, stride: int):
        super().__init__()
        self.c1 = nn.Sequential(
            nn.Conv2d(in_c, out_c, kernel_size=3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(out_c),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_c, out_c, kernel_size=3, padding=1, bias=False),
        )
        self.c2 = nn.Sequential(
            nn.Conv2d(in_c, out_c, kernel_size=1, stride=stride, padding=0, bias=False),
            nn.BatchNorm2d(out_c),
        )
        self.attn = SqueezeExcitation(out_c)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        x = self.c1(inputs)
        s = self.c2(inputs)
        return self.attn(x + s)


class ResNetBlock(nn.Module):
    def __init__(self, in_c: int, out_c: int, stride: int):
        super().__init__()
        self.c1 = nn.Sequential(
            nn.BatchNorm2d(in_c),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_c, out_c, kernel_size=3, padding=1, stride=stride, bias=False),
            nn.BatchNorm2d(out_c),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_c, out_c, kernel_size=3, padding=1, bias=False),
        )
        self.c2 = nn.Sequential(
            nn.Conv2d(in_c, out_c, kernel_size=1, stride=stride, padding=0, bias=False),
            nn.BatchNorm2d(out_c),
        )
        self.attn = SqueezeExcitation(out_c)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        x = self.c1(inputs)
        s = self.c2(inputs)
        return self.attn(x + s)


class ASPP(nn.Module):
    def __init__(self, in_c: int, out_c: int, rate=(1, 6, 12, 18)):
        super().__init__()
        self.c1 = nn.Sequential(
            nn.Conv2d(in_c, out_c, kernel_size=3, dilation=rate[0], padding=rate[0], bias=False),
            nn.BatchNorm2d(out_c),
        )
        self.c2 = nn.Sequential(
            nn.Conv2d(in_c, out_c, kernel_size=3, dilation=rate[1], padding=rate[1], bias=False),
            nn.BatchNorm2d(out_c),
        )
        self.c3 = nn.Sequential(
            nn.Conv2d(in_c, out_c, kernel_size=3, dilation=rate[2], padding=rate[2], bias=False),
            nn.BatchNorm2d(out_c),
        )
        self.c4 = nn.Sequential(
            nn.Conv2d(in_c, out_c, kernel_size=3, dilation=rate[3], padding=rate[3], bias=False),
            nn.BatchNorm2d(out_c),
        )
        self.c5 = nn.Conv2d(out_c, out_c, kernel_size=1, padding=0, bias=False)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        x1 = self.c1(inputs)
        x2 = self.c2(inputs)
        x3 = self.c3(inputs)
        x4 = self.c4(inputs)
        x = x1 + x2 + x3 + x4
        return self.c5(x)


class AttentionBlock(nn.Module):
    def __init__(self, in_c):
        super().__init__()
        out_c = in_c[1]
        self.g_conv = nn.Sequential(
            nn.BatchNorm2d(in_c[0]),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_c[0], out_c, kernel_size=3, padding=1, bias=False),
            nn.MaxPool2d((2, 2)),
        )
        self.x_conv = nn.Sequential(
            nn.BatchNorm2d(in_c[1]),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_c[1], out_c, kernel_size=3, padding=1, bias=False),
        )
        self.gc_conv = nn.Sequential(
            nn.BatchNorm2d(in_c[1]),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_c, out_c, kernel_size=3, padding=1, bias=False),
        )

    def forward(self, g: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        g_pool = self.g_conv(g)
        x_conv = self.x_conv(x)
        gc_sum = g_pool + x_conv
        gc_conv = self.gc_conv(gc_sum)
        return gc_conv * x


class DecoderBlock(nn.Module):
    def __init__(self, in_c, out_c: int):
        super().__init__()
        self.a1 = AttentionBlock(in_c)
        self.up = nn.Upsample(scale_factor=2, mode="nearest")
        self.r1 = ResNetBlock(in_c[0] + in_c[1], out_c, stride=1)

    def forward(self, g: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        d = self.a1(g, x)
        d = self.up(d)
        d = torch.cat([d, g], dim=1)
        return self.r1(d)


class ResUNetPlusPlus(nn.Module):
    def __init__(self, n_classes: int = 2):
        super().__init__()
        self.n_classes = int(n_classes)

        self.c1 = StemBlock(3, 16, stride=1)
        self.c2 = ResNetBlock(16, 32, stride=2)
        self.c3 = ResNetBlock(32, 64, stride=2)
        self.c4 = ResNetBlock(64, 128, stride=2)

        self.b1 = ASPP(128, 256)

        self.d1 = DecoderBlock([64, 256], 128)
        self.d2 = DecoderBlock([32, 128], 64)
        self.d3 = DecoderBlock([16, 64], 32)

        self.aspp_out = ASPP(32, 16)
        self.output = nn.Conv2d(16, self.n_classes, kernel_size=1, padding=0)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        c1 = self.c1(inputs)
        c2 = self.c2(c1)
        c3 = self.c3(c2)
        c4 = self.c4(c3)

        b1 = self.b1(c4)

        d1 = self.d1(c3, b1)
        d2 = self.d2(c2, d1)
        d3 = self.d3(c1, d2)

        out = self.aspp_out(d3)
        return self.output(out)


def resunet_plusplus(
    n_classes: int = 2,
    input_height: Optional[int] = None,
    input_width: Optional[int] = None,
):
    """与 tiny_unet / resnet50_unet 相同的工厂函数签名（高宽仅保留接口兼容）。"""
    del input_height, input_width
    return ResUNetPlusPlus(n_classes=n_classes)


if __name__ == "__main__":
    m = ResUNetPlusPlus(n_classes=2)
    x = torch.randn(1, 3, 256, 256)
    y = m(x)
    print("out:", y.shape)
    try:
        from ptflops import get_model_complexity_info

        flops, params = get_model_complexity_info(
            m, (3, 256, 256), as_strings=True, print_per_layer_stat=False
        )
        print("Flops:", flops, "Params:", params)
    except ImportError:
        print("(可选) pip install ptflops 可打印 FLOPs")
