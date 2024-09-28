import torch
import torch.nn as nn
import torch.nn.functional as F
from mamba_ssm.modules.mamba_simple import Mamba  # 引入Mamba模块


class MambaSSM(nn.Module):
    def __init__(self):
        super(MambaSSM, self).__init__()
        # 主路径
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, padding=1)

        # 增加多尺度特征提取
        self.conv1_small = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, padding=1, stride=2)
        self.conv2_small = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1, stride=2)

        # 用于将 x2_small 的通道数对齐为 256
        self.match_channels = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=1)

        self.ssm = Mamba(
            d_model=256,
            d_state=64,
            d_conv=4,
            expand=2,
        ).to("cuda")

        # 解卷积部分
        self.deconv1 = nn.ConvTranspose2d(in_channels=256, out_channels=128, kernel_size=3, padding=1)
        self.deconv2 = nn.ConvTranspose2d(in_channels=128, out_channels=64, kernel_size=3, padding=1)
        self.deconv3 = nn.ConvTranspose2d(in_channels=64, out_channels=3, kernel_size=3, padding=1)

    def forward(self, x):
        # 主路径
        x1 = F.relu(self.conv1(x))
        x2 = F.relu(self.conv2(x1))
        x3 = F.relu(self.conv3(x2))

        # 多尺度分支
        x1_small = F.relu(self.conv1_small(x))
        x2_small = F.relu(self.conv2_small(x1_small))

        # 将 x2_small 通道数对齐为 256
        x2_small_matched = self.match_channels(x2_small)

        # SSM 处理
        b, c, h, w = x3.shape
        x3_flat = x3.view(b, c, -1).permute(0, 2, 1)
        x3_ssm = self.ssm(x3_flat)
        x3_ssm = x3_ssm.permute(0, 2, 1).view(b, c, h, w)

        # 特征融合（主路径 + 多尺度）
        x3_fused = x3_ssm + F.interpolate(x2_small_matched, size=(h, w), mode='bilinear', align_corners=False)

        # 解卷积
        x4 = F.relu(self.deconv1(x3_fused))
        x5 = F.relu(self.deconv2(x4 + x2))
        x6 = self.deconv3(x5 + x1)

        return torch.sigmoid(x6)
