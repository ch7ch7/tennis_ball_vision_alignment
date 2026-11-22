# model.py
import torch
import torch.nn as nn
from torchvision.models import efficientnet_b3

class MultiBranchEfficientNet(nn.Module):
    def __init__(self):
        super().__init__()

        # 分支 A (細節分支)
        self.base_A = efficientnet_b3(weights=None)
        self._convert_to_grayscale(self.base_A)

        # 分支 B (輪廓分支)
        self.base_B = efficientnet_b3(weights=None)
        self._convert_to_grayscale(self.base_B)

        fusion_dim = 1536 * 2 # efficientnet_B0：1280 * 2   efficientnet_B3：1536 * 2

        def make_head(out_dim):
            return nn.Sequential(
                nn.Flatten(),
                nn.Linear(fusion_dim, 256),
                nn.BatchNorm1d(256),
                nn.ReLU(),
                nn.Dropout(0.3),
                nn.Linear(256, out_dim)
            )

        # init_angle 預測 (sin, cos)
        self.init_angle_head = make_head(2)
        # angle 預測 (sin, cos)
        self.angle_head = make_head(2)
        # roll 預測
        self.roll_head = make_head(1)

    def _convert_to_grayscale(self, base):
        conv1 = base.features[0][0]
        new_conv = nn.Conv2d(
            in_channels=1,
            out_channels=conv1.out_channels,
            kernel_size=conv1.kernel_size,
            stride=conv1.stride,
            padding=conv1.padding,
            bias=conv1.bias is not None
        )
        with torch.no_grad():
            new_conv.weight[:] = conv1.weight[:, :1, :, :]
        base.features[0][0] = new_conv

    def forward(self, x, x_blur):
        f_A = self.base_A.features(x)
        f_A = self.base_A.avgpool(f_A)

        f_B = self.base_B.features(x_blur)
        f_B = self.base_B.avgpool(f_B)

        f = torch.cat([f_A, f_B], dim=1)

        init_out = self.init_angle_head(f)
        angle_out = self.angle_head(f)
        roll_out = self.roll_head(f)

        return torch.cat([init_out, angle_out, roll_out], dim=1)