"""model.py - Defines the neural network architecture (U-Net)."""
import torch
import torch.nn as nn
import torch.nn.functional as F


class DoubleConv(nn.Module):
    def __init__(self, c_in, c_out, dropout=0.0):
        super().__init__()
        layers = [
            nn.Conv2d(c_in, c_out, 3, padding=1, bias=False), nn.BatchNorm2d(c_out), nn.ReLU(inplace=True),
            nn.Conv2d(c_out, c_out, 3, padding=1, bias=False), nn.BatchNorm2d(c_out), nn.ReLU(inplace=True),
        ]
        if dropout > 0:
            layers.append(nn.Dropout2d(dropout))
        self.block = nn.Sequential(*layers)

    def forward(self, x):
        return self.block(x)


class Up(nn.Module):
    def __init__(self, c_in, c_out, dropout=0.0):
        super().__init__()
        self.up = nn.ConvTranspose2d(c_in, c_out, 2, stride=2)
        self.conv = DoubleConv(c_out * 2, c_out, dropout)

    def forward(self, x, skip):
        x = self.up(x)
        if x.shape[-2:] != skip.shape[-2:]:
            x = F.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False)
        return self.conv(torch.cat([skip, x], dim=1))


class UNet(nn.Module):
    """Classic 4-level U-Net. Output = raw logits (num_classes channels; 1 for binary)."""

    def __init__(self, in_channels=3, num_classes=1, base_channels=32, dropout=0.1):
        super().__init__()
        b = base_channels
        self.enc1 = DoubleConv(in_channels, b)
        self.enc2 = DoubleConv(b, b * 2)
        self.enc3 = DoubleConv(b * 2, b * 4)
        self.enc4 = DoubleConv(b * 4, b * 8)
        self.bottleneck = DoubleConv(b * 8, b * 16, dropout)
        self.pool = nn.MaxPool2d(2)
        self.up4 = Up(b * 16, b * 8, dropout)
        self.up3 = Up(b * 8, b * 4, dropout)
        self.up2 = Up(b * 4, b * 2)
        self.up1 = Up(b * 2, b)
        self.head = nn.Conv2d(b, num_classes, 1)

    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        e4 = self.enc4(self.pool(e3))
        bn = self.bottleneck(self.pool(e4))
        d4 = self.up4(bn, e4)
        d3 = self.up3(d4, e3)
        d2 = self.up2(d3, e2)
        d1 = self.up1(d2, e1)
        return self.head(d1)

    def freeze_encoder(self):
        for m in (self.enc1, self.enc2, self.enc3, self.enc4, self.bottleneck):
            for p in m.parameters():
                p.requires_grad = False
