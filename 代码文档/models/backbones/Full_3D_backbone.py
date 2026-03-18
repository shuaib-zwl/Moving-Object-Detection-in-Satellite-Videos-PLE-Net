import torch
import torch.nn as nn
from mmcv.runner import BaseModule
from mmdet.models.builder import BACKBONES
import math
import os
import torch
import numpy as np
import matplotlib.pyplot as plt

class BasicConv3d(nn.Module):
    """Basic 3D Convolution Module."""
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding=0):
        super(BasicConv3d, self).__init__()
        self.conv = nn.Conv3d(
            in_channels, out_channels,
            kernel_size=kernel_size, stride=stride, padding=padding, bias=False
        )
        self.bn = nn.BatchNorm3d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x


@BACKBONES.register_module()
class Backbone3D(BaseModule):
    """Custom 3D Backbone based on the provided script."""
    def __init__(self,
                 in_channels=3,
                 channels=[16, 32, 64],
                 init_cfg=None):
        super(Backbone3D, self).__init__(init_cfg)
        # 3D卷积核的格式是[depth, H, W]
        # Branch 1
        self.branch1 = nn.Sequential(
            # BasicConv3d(in_channels, channels[0], kernel_size=(1, 1, 5), stride=(1, 1, 1), padding=(0, 0, 2)),
            BasicConv3d(in_channels, channels[0], kernel_size=(1, 1, 3), stride=(1, 1, 1), padding=(0, 0, 1)),
            # BasicConv3d(channels[0], channels[0], kernel_size=(1, 5, 1), stride=(1, 1, 1), padding=(0, 2, 0)),
            BasicConv3d(channels[0], channels[0], kernel_size=(1, 3, 1), stride=(1, 1, 1), padding=(0, 1, 0)),
            BasicConv3d(channels[0], channels[0], kernel_size=(3, 1, 1), stride=(1, 1, 1), padding=(1, 0, 0)),
        )
        # Branch 2
        self.branch2 = nn.Sequential(
            BasicConv3d(channels[0], channels[1], kernel_size=(1, 1, 3), stride=(1, 1, 1), padding=(0, 0, 1)),
            BasicConv3d(channels[1], channels[1], kernel_size=(1, 3, 1), stride=(1, 1, 1), padding=(0, 1, 0)),
            BasicConv3d(channels[1], channels[1], kernel_size=(3, 1, 1), stride=(1, 1, 1), padding=(1, 0, 0)),
        )
        # Branch 3
        self.branch3 = nn.Sequential(
            BasicConv3d(channels[1], channels[2], kernel_size=(1, 1, 3), stride=(1, 1, 1), padding=(0, 0, 1)),
            BasicConv3d(channels[2], channels[2], kernel_size=(1, 3, 1), stride=(1, 1, 1), padding=(0, 1, 0)),
            BasicConv3d(channels[2], channels[2], kernel_size=(3, 1, 1), stride=(1, 1, 1), padding=(1, 0, 0)),
        )
        self.maxpool = nn.MaxPool3d(kernel_size=(1, 2, 2))
        self.maxpool1 = nn.MaxPool3d(kernel_size=(3, 1, 1))

    def forward(self, x):
        layers = []
        x = self.branch1(x)
        layers.append(self.maxpool1(x).squeeze(2))  # First layer's feature map
        x = self.maxpool(x)
        x = self.branch2(x)
        layers.append(self.maxpool1(x).squeeze(2))  # Second layer's feature map
        x = self.maxpool(x)
        x = self.branch3(x)
        layers.append(self.maxpool1(x).squeeze(2))  # Third layer's feature map
        return layers  # Return multi-scale feature maps

