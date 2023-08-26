import torch
import torch.nn as nn


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_channels, out_channels, stride=1):
        super(BasicBlock, self).__init__()

        # 保留
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels * self.expansion:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels * self.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels * self.expansion)
            )

        # 残差
        self.residual = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, stride=stride, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),

            nn.Conv2d(out_channels, out_channels * self.expansion, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels * self.expansion)
        )

    def forward(self, x):
        return nn.ReLU(inplace=True)(
            self.residual(x) + self.shortcut(x)
        )


class BottleNeck(nn.Module):
    expansion = 4

    def __init__(self, in_channels, out_channels, stride=1):
        super(BottleNeck, self).__init__()

        # 保留
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels * self.expansion:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels * self.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels * self.expansion)
            )

        # 残差
        self.residual = nn.Sequential(

            nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),

            nn.Conv2d(out_channels, out_channels, stride=stride, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),

            nn.Conv2d(out_channels, out_channels * self.expansion, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels * self.expansion)
        )

    def forward(self, x):
        return nn.ReLU(inplace=True)(
            self.residual(x) + self.shortcut(x)
        )


class ResNet(nn.Module):
    def __init__(self, block, block_num, class_num=100):
        self.in_channels = 64
        super(ResNet, self).__init__()

        self.pool1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=7, padding=3, stride=2, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            # nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        )

        self.pool2 = self._make_layer(block, 64, block_num[0], 1)
        self.pool3 = self._make_layer(block, 128, block_num[1], 2)
        self.pool4 = self._make_layer(block, 256, block_num[2], 2)
        self.pool5 = self._make_layer(block, 512, block_num[3], 2)
        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(block.expansion * 512, class_num)

    def _make_layer(self, block, out_channels, block_num, stride):
        strides = [stride] + [1] * (block_num - 1)
        inner_layers = []
        for stride in strides:
            inner_layers.append(block(self.in_channels, out_channels, stride))
            self.in_channels = out_channels * block.expansion

        return nn.Sequential(*inner_layers)

    def forward(self, x):
        x = self.pool1(x)
        x = self.pool2(x)
        x = self.pool3(x)
        x = self.pool4(x)
        x = self.pool5(x)
        x = self.avg_pool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x


def resnet101():
    return ResNet(BottleNeck, [3, 4, 23, 3])


def resnet50():
    return ResNet(BottleNeck, [3, 4, 6, 3])


def resnet34():
    return ResNet(BasicBlock, [3, 4, 6, 3])


def resnet18():
    return ResNet(BasicBlock, [2, 2, 2, 2])
