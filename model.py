import torch
import torch.nn as nn
import torch.nn.functional as F
from mamba_ssm.modules.mamba_simple import Mamba  # 引入Mamba模块


class MambaSSM(nn.Module):
    def __init__(self):
        super(MambaSSM, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, padding=1)
        self.ssm = Mamba(
            d_model=256,  # Model dimension d_model
            d_state=64,  # SSM state expansion factor
            d_conv=4,  # Local convolution width
            expand=2,  # Block expansion factor
        ).to("cuda")
        self.deconv1 = nn.ConvTranspose2d(in_channels=256, out_channels=128, kernel_size=3, padding=1)
        self.deconv2 = nn.ConvTranspose2d(in_channels=128, out_channels=64, kernel_size=3, padding=1)
        self.deconv3 = nn.ConvTranspose2d(in_channels=64, out_channels=3, kernel_size=3, padding=1)

    def forward(self, x):
        x1 = F.relu(self.conv1(x))
        x2 = F.relu(self.conv2(x1))
        x3 = F.relu(self.conv3(x2))

        # SSM层调整维度
        b, c, h, w = x3.shape
        x3_flat = x3.view(b, c, -1).permute(0, 2, 1)
        x3_ssm = self.ssm(x3_flat)
        x3_ssm = x3_ssm.permute(0, 2, 1).view(b, c, h, w)

        x4 = F.relu(self.deconv1(x3_ssm))
        x5 = F.relu(self.deconv2(x4 + x2))
        x6 = self.deconv3(x5 + x1)
        return F.relu(x6)
