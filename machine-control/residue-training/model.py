# model.py
import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = ["UNet", "DiceLoss", "FocalLoss"]

class ConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch, dropout=0.3):
        super().__init__()
        layers = [
            nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        ]
        if dropout > 0:
            layers.append(nn.Dropout2d(p=dropout))
        self.conv = nn.Sequential(*layers)
    def forward(self, x):
        return self.conv(x)

class UpBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_ch, out_ch, 2, 2)
        self.conv = ConvBlock(in_ch, out_ch)
    def forward(self, x, skip):
        x = self.up(x)
        if x.size()[2:] != skip.size()[2:]:
            dy = skip.size(2) - x.size(2)
            dx = skip.size(3) - x.size(3)
            x = F.pad(x, [dx//2, dx-dx//2, dy//2, dy-dy//2])
        x = torch.cat([skip, x], dim=1)
        return self.conv(x)

class UNet(nn.Module):
    def __init__(self, in_ch=3, out_ch=1, base=32, dropout=0.3):
        super().__init__()
        self.enc1 = ConvBlock(in_ch, base, dropout)
        self.enc2 = ConvBlock(base, base*2, dropout)
        self.enc3 = ConvBlock(base*2, base*4, dropout)
        self.enc4 = ConvBlock(base*4, base*8, dropout)
        self.pool = nn.MaxPool2d(2)
        self.center = ConvBlock(base*8, base*16, dropout)
        self.up4 = UpBlock(base*16, base*8)
        self.up3 = UpBlock(base*8, base*4)
        self.up2 = UpBlock(base*4, base*2)
        self.up1 = UpBlock(base*2, base)
        self.final = nn.Conv2d(base, out_ch, 1)
    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        e4 = self.enc4(self.pool(e3))
        c = self.center(self.pool(e4))
        d4 = self.up4(c, e4)
        d3 = self.up3(d4, e3)
        d2 = self.up2(d3, e2)
        d1 = self.up1(d2, e1)
        return self.final(d1)

class DiceLoss(nn.Module):
    def __init__(self, eps=1e-6):
        super().__init__()
        self.eps = eps
    def forward(self, logits, targets):
        probs = torch.sigmoid(logits)
        probs = probs.view(probs.size(0), -1)
        targets = targets.view(targets.size(0), -1)
        inter = (probs * targets).sum(1)
        denom = probs.sum(1) + targets.sum(1)
        loss = 1 - ((2 * inter + self.eps) / (denom + self.eps))
        return loss.mean()

# <<< 新增 FocalLoss 類別 >>>
class FocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2.0, reduction='mean'):
        """
        Focal Loss 的 PyTorch 實現。
        :param alpha: 用於平衡正負樣本的權重，數值越高，越關注正樣本。
        :param gamma: 聚焦參數，用於降低簡單樣本的權重。
        :param reduction: 'mean', 'sum', 'none'
        """
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, logits, targets):
        # 使用 BCEWithLogitsLoss 來獲得 log_pt，數值上更穩定
        bce_loss = F.binary_cross_entropy_with_logits(logits, targets, reduction='none')
        
        # 得到預測機率 pt
        pt = torch.exp(-bce_loss)
        
        # 計算 focal loss
        focal_loss = (1 - pt)**self.gamma * bce_loss
        
        # 如果設定了 alpha，則加入權重
        if self.alpha is not None:
            alpha_t = targets * self.alpha + (1 - targets) * (1 - self.alpha)
            focal_loss = alpha_t * focal_loss
            
        # 根據 reduction 參數返回結果
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss