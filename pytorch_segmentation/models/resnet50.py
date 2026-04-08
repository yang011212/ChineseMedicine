import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models

IMAGE_ORDERING = 'channels_first'

def one_side_pad(x, padding=1):
    # 先进行零填充
    x = F.pad(x, (padding, padding, padding, padding), mode='constant', value=0)
    # 移除右侧和下侧的填充
    x = x[:, :, :-padding, :-padding]
    return x

class IdentityBlock(nn.Module):

    def __init__(self, in_channels, filters, kernel_size=3, stage=1, block='a'):
        super(IdentityBlock, self).__init__()
        filters1, filters2, filters3 = filters

        # 第一部分：Conv2D(1x1)+BN+ReLU
        self.conv1 = nn.Conv2d(in_channels, filters1, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(filters1)

        # 第二部分：Conv2D(3x3)+BN+ReLU
        self.conv2 = nn.Conv2d(filters1, filters2, kernel_size=kernel_size, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(filters2)

        # 第三部分：Conv2D(1x1)+BN
        self.conv3 = nn.Conv2d(filters2, filters3, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(filters3)

        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        identity = x

        # 第一部分
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        # 第二部分
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        # 第三部分
        out = self.conv3(out)
        out = self.bn3(out)

        # Add + ReLU
        out += identity
        out = self.relu(out)

        return out
    
class ConvBlock(nn.Module):

    def __init__(self, in_channels, filters, kernel_size=3, stride=2, stage=1, block='a'):
        super(ConvBlock, self).__init__()
        filters1, filters2, filters3 = filters

        # 主分支
        self.conv1 = nn.Conv2d(in_channels, filters1, kernel_size=1, stride=stride, bias=False)
        self.bn1 = nn.BatchNorm2d(filters1)

        self.conv2 = nn.Conv2d(filters1, filters2, kernel_size=kernel_size, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(filters2)

        self.conv3 = nn.Conv2d(filters2, filters3, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(filters3)

        # 快捷连接
        self.shortcut = nn.Sequential(
            nn.Conv2d(in_channels, filters3, kernel_size=1, stride=stride, bias=False),
            nn.BatchNorm2d(filters3)
        )

        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        identity = self.shortcut(x)

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        # Add + ReLU
        out += identity
        out = self.relu(out)

        return out
    
class ResNet50Encoder(nn.Module):

    def __init__(self, input_height=416, input_width=608):
        super(ResNet50Encoder, self).__init__()
        self.input_height = input_height
        self.input_width = input_width

        # 加载预训练的 ResNet50
        try:
            # 新版本 torchvision 使用 weights 参数
            resnet = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
        except:
            # 兼容旧版本
            resnet = models.resnet50(pretrained=True)

        # 提取各阶段的特征层
        self.conv1 = resnet.conv1
        self.bn1 = resnet.bn1
        self.relu = resnet.relu
        self.maxpool = resnet.maxpool

        self.layer1 = resnet.layer1  # 256 channels
        self.layer2 = resnet.layer2  # 512 channels
        self.layer3 = resnet.layer3  # 1024 channels
        self.layer4 = resnet.layer4  # 2048 channels

    def forward(self, x):
        # 创建输入层
        img_input = x

        # 第一层特征
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        f1 = x  # 64 channels

        x = self.maxpool(x)

        # 各层特征提取
        f2 = self.layer1(x)   # 256 channels
        f3 = self.layer2(f2)  # 512 channels
        f4 = self.layer3(f3)  # 1024 channels
        f5 = self.layer4(f4)  # 2048 channels

        return img_input, [f1, f2, f3, f4, f5]


def get_resnet50_encoder(input_height=416, input_width=608):
    encoder = ResNet50Encoder(input_height, input_width)
    return encoder