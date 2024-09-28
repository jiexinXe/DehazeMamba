import torch
import torch.nn as nn
import torch.nn.functional as F
from mamba_ssm.modules.mamba_simple import Mamba  # 引入Mamba模块


class MambaSSM(nn.Module):
    def __init__(self):
        super(MambaSSM, self).__init__()

        # 使用不同的卷积核大小来实现多尺度卷积
        self.conv1_1 = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, padding=1)
        self.conv1_2 = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=5, padding=2)
        self.conv1_3 = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=7, padding=3)

        # 第二层卷积：减少输出通道数以节省显存
        self.conv2 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1)  # 修改通道数为64

        # 第三层卷积
        self.conv3 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, padding=1)

        # Mamba模块
        self.ssm = Mamba(
            d_model=256,  # 模型维度 d_model
            d_state=64,  # SSM 状态扩展因子
            d_conv=4,  # 局部卷积宽度
            expand=2,  # 扩展因子
        ).to("cuda")

        # 反卷积层
        self.deconv1 = nn.ConvTranspose2d(in_channels=256, out_channels=128, kernel_size=3, padding=1)
        self.deconv2 = nn.ConvTranspose2d(in_channels=128, out_channels=64, kernel_size=3, padding=1)
        self.deconv3 = nn.ConvTranspose2d(in_channels=64, out_channels=3, kernel_size=3, padding=1)

        # 特征调制模块（类似于特征融合）
        self.feature_modulation = nn.Sequential(
            nn.Conv2d(256, 256, kernel_size=1),
            nn.Sigmoid()
        )

        # 新增：1x1卷积，用于减少x1的通道数从96到64
        self.conv1x1_reduce = nn.Conv2d(in_channels=96, out_channels=64, kernel_size=1)

    def forward(self, x):
        # 多尺度卷积的输出
        x1_1 = F.relu(self.conv1_1(x))
        x1_2 = F.relu(self.conv1_2(x))
        x1_3 = F.relu(self.conv1_3(x))

        # 将多尺度的输出拼接
        x1 = torch.cat([x1_1, x1_2, x1_3], dim=1)  # 96通道

        # 添加一个1x1卷积来将通道数从96减少到64
        x1_reduced = F.relu(self.conv1x1_reduce(x1))

        # 第二层卷积
        x2 = F.relu(self.conv2(x1_reduced))  # 修改后的 x1_reduced 匹配 conv2

        # 第三层卷积
        x3 = F.relu(self.conv3(x2))

        # 调整维度以便Mamba模块处理
        b, c, h, w = x3.shape
        x3_flat = x3.view(b, c, -1).permute(0, 2, 1)
        x3_ssm = self.ssm(x3_flat)
        x3_ssm = x3_ssm.permute(0, 2, 1).view(b, c, h, w)

        # 特征调制（融合全局与局部特征）
        modulation = self.feature_modulation(x3_ssm)
        x3_ssm = x3_ssm * modulation

        # 反卷积部分
        x4 = F.relu(self.deconv1(x3_ssm))
        x5 = F.relu(self.deconv2(x4 + x2))  # 融合跳跃连接
        x6 = self.deconv3(x5 + x1_reduced)  # 使用x1_reduced替换x1，以匹配通道数

        return torch.sigmoid(x6)  # 限制输出到[0, 1]
