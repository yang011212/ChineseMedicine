import math

import torch
import torch.nn as nn
import torch.nn.functional as F

IMAGE_ORDERING = 'channels_first'
MERGE_AXIS = 1

def _conv_bn_relu(in_ch, out_ch, k=3, p=1):
    return nn.Sequential(
        nn.Conv2d(in_ch, out_ch, kernel_size=k, padding=p, bias=False),
        nn.BatchNorm2d(out_ch),
        nn.ReLU(inplace=True),
    )


class UpBlock(nn.Module):
    def __init__(self, in_ch, skip_ch, out_ch):
        super().__init__()
        self.conv1 = _conv_bn_relu(in_ch + skip_ch, out_ch)
        self.conv2 = _conv_bn_relu(out_ch, out_ch)

    def forward(self, x, skip):
        x = F.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False)
        x = torch.cat([x, skip], dim=MERGE_AXIS)
        x = self.conv2(self.conv1(x))
        return x


class ResUNetDecoder(nn.Module):
    def __init__(self, n_classes):
        super().__init__()
        self.center = _conv_bn_relu(2048, 1024)
        self.up4 = UpBlock(1024, 1024, 512)
        self.up3 = UpBlock(512, 512, 256)
        self.up2 = UpBlock(256, 256, 128)
        self.up1 = UpBlock(128, 64, 64)
        self.classifier = nn.Conv2d(64, n_classes, kernel_size=1)

    def forward(self, encoder_output):
        # encoder_output: (img_input, [f1,f2,f3,f4,f5])
        _, feats = encoder_output
        f1, f2, f3, f4, f5 = feats
        x = self.center(f5)
        x = self.up4(x, f4)
        x = self.up3(x, f3)
        x = self.up2(x, f2)
        x = self.up1(x, f1)
        x = self.classifier(x)
        return x
    
from .resnet50 import get_resnet50_encoder
from .model_utils import get_segmentation_model


def resnet50_unet(n_classes, input_height=416, input_width=608):
    encoder = get_resnet50_encoder(input_height=input_height, input_width=input_width)
    decoder = ResUNetDecoder(n_classes=n_classes)
    model = get_segmentation_model(encoder, decoder, n_classes, input_height, input_width)
    return model


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


class TinyUNetEncoder(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.cmrf = CMRF(in_channels, out_channels)
        self.downsample = nn.MaxPool2d(kernel_size=2, stride=2)

    def forward(self, x):
        x = self.cmrf(x)
        return self.downsample(x), x


class TinyUNetDecoder(nn.Module):
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
        self.n_classes = int(num_classes)
        in_filters = [192, 384, 768, 1024]
        out_filters = [64, 128, 256, 512]

        self.encoder1 = TinyUNetEncoder(in_channels, 64)
        self.encoder2 = TinyUNetEncoder(64, 128)
        self.encoder3 = TinyUNetEncoder(128, 256)
        self.encoder4 = TinyUNetEncoder(256, 512)

        self._build_decoders(in_filters, out_filters, num_classes)

    def _build_decoders(self, in_filters, out_filters, num_classes):
        self.decoder4 = TinyUNetDecoder(in_filters[3], out_filters[3])
        self.decoder3 = TinyUNetDecoder(in_filters[2], out_filters[2])
        self.decoder2 = TinyUNetDecoder(in_filters[1], out_filters[1])
        self.decoder1 = TinyUNetDecoder(in_filters[0], out_filters[0])
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


class TinyUNetPretrainedEncoder(nn.Module):
    """
    ImageNet 预训练 ResNet50 多尺度特征，经 1x1 对齐到原 TinyUNet 解码器所需的 skip / bottleneck 形状：
    bottleneck: 512 @ H/16；skip4:512@H/8；skip3:256@H/4；skip2:128@H/2；skip1:64@H

    （与旧版 ResNet18 骨干不兼容：曾用 tinyunet_pt 训练的权重需重新训练。）
    """

    def __init__(self, pretrained: bool = True):
        super().__init__()
        try:
            from torchvision.models import resnet50, ResNet50_Weights

            w = ResNet50_Weights.IMAGENET1K_V1 if pretrained else None
            backbone = resnet50(weights=w)
        except Exception:
            from torchvision.models import resnet50

            backbone = resnet50(pretrained=pretrained)

        self.conv1 = backbone.conv1
        self.bn1 = backbone.bn1
        self.relu = backbone.relu
        self.maxpool = backbone.maxpool
        self.layer1 = backbone.layer1
        self.layer2 = backbone.layer2
        self.layer3 = backbone.layer3

        # stem 仍为 64@H/2；layer1/2/3 为 256 / 512 / 1024 通道（ResNet50 bottleneck）
        self.proj_skip1 = nn.Conv2d(64, 64, kernel_size=1, bias=False)
        self.proj_skip2 = nn.Conv2d(64, 128, kernel_size=1, bias=False)
        self.proj_skip3 = nn.Conv2d(256, 256, kernel_size=1, bias=False)
        self.proj_skip4 = nn.Conv2d(512, 512, kernel_size=1, bias=False)
        self.proj_bot = nn.Conv2d(1024, 512, kernel_size=1, bias=False)

        nn.init.kaiming_normal_(self.proj_skip1.weight, mode="fan_out", nonlinearity="relu")
        nn.init.kaiming_normal_(self.proj_skip2.weight, mode="fan_out", nonlinearity="relu")
        nn.init.kaiming_normal_(self.proj_skip3.weight, mode="fan_out", nonlinearity="relu")
        nn.init.kaiming_normal_(self.proj_skip4.weight, mode="fan_out", nonlinearity="relu")
        nn.init.kaiming_normal_(self.proj_bot.weight, mode="fan_out", nonlinearity="relu")

    def forward(self, x: torch.Tensor):
        H, W = x.shape[2], x.shape[3]

        x0 = self.relu(self.bn1(self.conv1(x)))
        skip1 = self.proj_skip1(F.interpolate(x0, size=(H, W), mode="bilinear", align_corners=False))

        x = self.maxpool(x0)
        l1 = self.layer1(x)
        l2 = self.layer2(l1)
        l3 = self.layer3(l2)

        skip2 = self.proj_skip2(x0)
        skip3 = self.proj_skip3(l1)
        skip4 = self.proj_skip4(l2)
        bottleneck = self.proj_bot(l3)
        return bottleneck, skip1, skip2, skip3, skip4


class TinyUNetPretrained(TinyUNet):
    """TinyUNet 解码器 + ImageNet 预训练 ResNet50 编码器（投影层随机初始化）。"""

    def __init__(self, in_channels=3, num_classes=2, pretrained: bool = True):
        nn.Module.__init__(self)
        self.n_classes = int(num_classes)
        if in_channels != 3:
            raise ValueError("TinyUNetPretrained 当前仅支持 in_channels=3（与 ResNet 预训练一致）")

        in_filters = [192, 384, 768, 1024]
        out_filters = [64, 128, 256, 512]
        self.encoder = TinyUNetPretrainedEncoder(pretrained=pretrained)
        self._build_decoders(in_filters, out_filters, num_classes)

    def forward(self, x):
        x, skip1, skip2, skip3, skip4 = self.encoder(x)
        x = self.decoder4(x, skip4)
        x = self.decoder3(x, skip3)
        x = self.decoder2(x, skip2)
        x = self.decoder1(x, skip1)
        return self.final_conv(x)


def tiny_unet(n_classes, input_height=416, input_width=608, pretrained: bool = False):
    del input_height, input_width
    if pretrained:
        return TinyUNetPretrained(in_channels=3, num_classes=n_classes, pretrained=True)
    return TinyUNet(in_channels=3, num_classes=n_classes)