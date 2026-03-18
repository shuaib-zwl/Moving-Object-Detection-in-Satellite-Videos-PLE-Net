import torch
import torch.nn as nn
from mmcv.runner import BaseModule
from mmdet.models.builder import BACKBONES


class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        # 防止除零错误，加个 max(1, ...)
        hidden_planes = max(1, in_planes // ratio)
        self.fc1 = nn.Conv2d(in_planes, hidden_planes, 1, bias=False)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Conv2d(hidden_planes, in_planes, 1, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc2(self.relu1(self.fc1(self.avg_pool(x))))
        max_out = self.fc2(self.relu1(self.fc1(self.max_pool(x))))
        out = avg_out + max_out
        return self.sigmoid(out)


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        padding = 3 if kernel_size == 7 else 1
        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv1(x)
        return self.sigmoid(x)


# -----------------------------------------------------------------
# 2. Res_CBAM_block (支持自定义 kernel_size)
# -----------------------------------------------------------------
class Res_CBAM_block(BaseModule):
    def __init__(self, in_channels, out_channels, stride=1, kernel_size=3, init_cfg=None):
        super(Res_CBAM_block, self).__init__(init_cfg)

        # 自动计算 padding，保证特征图尺寸不变 (k=3->p=1, k=5->p=2)
        padding = kernel_size // 2

        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)

        if stride != 1 or out_channels != in_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride),
                nn.BatchNorm2d(out_channels))
        else:
            self.shortcut = None

        self.ca = ChannelAttention(out_channels)
        self.sa = SpatialAttention()

    def forward(self, x):
        residual = x
        if self.shortcut is not None:
            residual = self.shortcut(x)
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.ca(out) * out
        out = self.sa(out) * out
        out += residual
        out = self.relu(out)
        return out


@BACKBONES.register_module()
class DNANet_FrameStack(BaseModule):
    def __init__(self,
                 num_frames=5,  # <--- 新增参数：帧数
                 input_channels=3,  # 单帧通道数
                 num_blocks=[2, 2, 2, 2],
                 nb_filter=[16, 32, 64, 128, 256],
                 max_downsample=8,  # 建议设为8，适应小目标
                 deep_supervision=False,
                 init_cfg=None):
        super(DNANet_FrameStack, self).__init__(init_cfg)

        assert max_downsample in [8, 16], "max_downsample must be 8 or 16"
        self.max_downsample = max_downsample
        self.deep_supervision = deep_supervision
        self.nb_filter = nb_filter

        # --- 方法一核心：计算堆叠后的通道数 ---
        self.stacked_channels = input_channels * num_frames  # 3 * 5 = 15

        # 基础组件
        self.pool = nn.MaxPool2d(2, 2)
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.down = nn.Upsample(scale_factor=0.5, mode='bilinear', align_corners=True)

        block = Res_CBAM_block

        # ---------------- Stage 0 (Stride 1) ----------------
        # <--- 核心修改：输入通道为 15，kernel_size 为 5 --->
        self.conv0_0 = self._make_layer_custom_kernel(block, self.stacked_channels, nb_filter[0], kernel_size=5)

        self.conv0_1 = self._make_layer(block, nb_filter[0] + nb_filter[1], nb_filter[0])
        self.conv0_2 = self._make_layer(block, nb_filter[0] * 2 + nb_filter[1], nb_filter[0])
        self.conv0_3 = self._make_layer(block, nb_filter[0] * 3 + nb_filter[1], nb_filter[0])
        self.conv0_4 = self._make_layer(block, nb_filter[0] * 4 + nb_filter[1], nb_filter[0])

        self.conv0_1_1x1 = nn.Conv2d(nb_filter[1], nb_filter[0], kernel_size=1)
        self.conv0_2_1x1 = nn.Conv2d(nb_filter[2], nb_filter[0], kernel_size=1)
        self.conv0_3_1x1 = nn.Conv2d(nb_filter[3], nb_filter[0], kernel_size=1)

        # ---------------- Stage 1 (Stride 2) ----------------
        self.conv1_0 = self._make_layer(block, nb_filter[0], nb_filter[1], num_blocks[0])
        self.conv1_1 = self._make_layer(block, nb_filter[1] + nb_filter[2] + nb_filter[0], nb_filter[1], num_blocks[0])
        self.conv1_2 = self._make_layer(block, nb_filter[1] * 2 + nb_filter[2] + nb_filter[0], nb_filter[1],
                                        num_blocks[0])
        self.conv1_3 = self._make_layer(block, nb_filter[1] * 3 + nb_filter[2] + nb_filter[0], nb_filter[1],
                                        num_blocks[0])

        # ---------------- Stage 2 (Stride 4) ----------------
        self.conv2_0 = self._make_layer(block, nb_filter[1], nb_filter[2], num_blocks[1])
        self.conv2_1 = self._make_layer(block, nb_filter[2] + nb_filter[3] + nb_filter[1], nb_filter[2], num_blocks[1])
        self.conv2_2 = self._make_layer(block, nb_filter[2] * 2 + nb_filter[3] + nb_filter[1], nb_filter[2],
                                        num_blocks[1])

        # ---------------- Stage 3 (Stride 8) ----------------
        self.conv3_0 = self._make_layer(block, nb_filter[2], nb_filter[3], num_blocks[2])

        # 根据 max_downsample 构建后续层
        if self.max_downsample == 16:
            self.conv4_0 = self._make_layer(block, nb_filter[3], nb_filter[4], num_blocks[3])
            self.conv3_1 = self._make_layer(block, nb_filter[3] + nb_filter[4] + nb_filter[2], nb_filter[3],
                                            num_blocks[2])

            self.conv0_4_final = self._make_layer(block, nb_filter[0] * 5, nb_filter[0])
            self.conv0_4_1x1 = nn.Conv2d(nb_filter[4], nb_filter[0], kernel_size=1)

            self.up_4 = nn.Upsample(scale_factor=4, mode='bilinear', align_corners=True)
            self.up_8 = nn.Upsample(scale_factor=8, mode='bilinear', align_corners=True)
            self.up_16 = nn.Upsample(scale_factor=16, mode='bilinear', align_corners=True)
        else:
            self.conv4_0 = None
            self.conv3_1 = self._make_layer(block, nb_filter[3] + nb_filter[2], nb_filter[3], num_blocks[2])

            self.conv0_4_final = self._make_layer(block, nb_filter[0] * 4, nb_filter[0])
            self.conv0_4_1x1 = None

            self.up_4 = nn.Upsample(scale_factor=4, mode='bilinear', align_corners=True)
            self.up_8 = nn.Upsample(scale_factor=8, mode='bilinear', align_corners=True)

    # 辅助构建层函数
    def _make_layer(self, block, input_channels, output_channels, num_blocks=1):
        layers = []
        layers.append(block(input_channels, output_channels))
        for i in range(num_blocks - 1):
            layers.append(block(output_channels, output_channels))
        return nn.Sequential(*layers)

    # <--- 新增辅助函数：用于构建大卷积核层 --->
    def _make_layer_custom_kernel(self, block, input_channels, output_channels, kernel_size):
        return block(input_channels, output_channels, kernel_size=kernel_size)

    def forward(self, input):
        # input shape: [B, 3, 5, H, W] 或 [B, 15, H, W]
        # 如果是 5D 张量，先展平,得到RRRRR GGGGG BBBBB
        if input.dim() == 5:
            B, C, T, H, W = input.shape
            input = input.view(B, C * T, H, W)

        # --- DNA Net 前向传播逻辑 (标准) ---
        # Column 0
        x0_0 = self.conv0_0(input)  # 使用 5x5 卷积处理 15 通道

        x1_0 = self.conv1_0(self.pool(x0_0))
        x0_1 = self.conv0_1(torch.cat([x0_0, self.up(x1_0)], 1))

        x2_0 = self.conv2_0(self.pool(x1_0))
        x1_1 = self.conv1_1(torch.cat([x1_0, self.up(x2_0), self.down(x0_1)], 1))
        x0_2 = self.conv0_2(torch.cat([x0_0, x0_1, self.up(x1_1)], 1))

        x3_0 = self.conv3_0(self.pool(x2_0))
        x2_1 = self.conv2_1(torch.cat([x2_0, self.up(x3_0), self.down(x1_1)], 1))
        x1_2 = self.conv1_2(torch.cat([x1_0, x1_1, self.up(x2_1), self.down(x0_2)], 1))
        x0_3 = self.conv0_3(torch.cat([x0_0, x0_1, x0_2, self.up(x1_2)], 1))

        # Handling Depth (16 vs 8)
        if self.max_downsample == 16:
            x4_0 = self.conv4_0(self.pool(x3_0))
            x3_1 = self.conv3_1(torch.cat([x3_0, self.up(x4_0), self.down(x2_1)], 1))
            x2_2 = self.conv2_2(torch.cat([x2_0, x2_1, self.up(x3_1), self.down(x1_2)], 1))
            x1_3 = self.conv1_3(torch.cat([x1_0, x1_1, x1_2, self.up(x2_2), self.down(x0_3)], 1))
            x0_4 = self.conv0_4(torch.cat([x0_0, x0_1, x0_2, x0_3, self.up(x1_3)], 1))

            Final_x0_4 = self.conv0_4_final(torch.cat([
                self.up_16(self.conv0_4_1x1(x4_0)),
                self.up_8(self.conv0_3_1x1(x3_1)),
                self.up_4(self.conv0_2_1x1(x2_2)),
                self.up(self.conv0_1_1x1(x1_3)),
                x0_4], 1))
        else:
            x3_1 = self.conv3_1(torch.cat([x3_0, self.down(x2_1)], 1))
            x2_2 = self.conv2_2(torch.cat([x2_0, x2_1, self.up(x3_1), self.down(x1_2)], 1))
            x1_3 = self.conv1_3(torch.cat([x1_0, x1_1, x1_2, self.up(x2_2), self.down(x0_3)], 1))
            x0_4 = self.conv0_4(torch.cat([x0_0, x0_1, x0_2, x0_3, self.up(x1_3)], 1))

            Final_x0_4 = self.conv0_4_final(torch.cat([
                self.up_8(self.conv0_3_1x1(x3_1)),
                self.up_4(self.conv0_2_1x1(x2_2)),
                self.up(self.conv0_1_1x1(x1_3)),
                x0_4], 1))

        # MMDetection Backbone 要求返回 tuple
        if self.deep_supervision:
            return (Final_x0_4,)
        else:
            return (Final_x0_4,)