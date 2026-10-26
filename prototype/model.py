"""U-Net with a ResNet-34 encoder. Same architecture as Prototype.ipynb section 7, so the saved
state dicts (`<variant>_best.pt`) load without changes. Dropout is inactive at inference."""
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision


class UpBlock(nn.Module):
    def __init__(self, in_ch, skip_ch, out_ch):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch + skip_ch, out_ch, 3, padding=1, bias=False), nn.BatchNorm2d(out_ch), nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False), nn.BatchNorm2d(out_ch), nn.ReLU(inplace=True))

    def forward(self, x, skip=None):
        x = F.interpolate(x, scale_factor=2, mode="bilinear", align_corners=False)
        if skip is not None:
            x = torch.cat([x, skip], 1)
        return self.conv(x)


class UNetResNet34(nn.Module):
    def __init__(self, in_ch, n_classes):
        super().__init__()
        enc = torchvision.models.resnet34(weights=None)  # weights come from the uploaded checkpoint
        self.stem = nn.Sequential(nn.Conv2d(in_ch, 64, 7, stride=2, padding=3, bias=False), enc.bn1, enc.relu)
        self.pool = enc.maxpool
        self.enc1, self.enc2, self.enc3, self.enc4 = enc.layer1, enc.layer2, enc.layer3, enc.layer4
        self.dec4 = UpBlock(512, 256, 256)
        self.dec3 = UpBlock(256, 128, 128)
        self.dec2 = UpBlock(128, 64, 64)
        self.dec1 = UpBlock(64, 64, 64)
        self.dec0 = UpBlock(64, 0, 32)
        self.dropout = nn.Dropout2d(0.0)
        self.head = nn.Conv2d(32, n_classes, 1)

    def forward(self, x):
        s0 = self.stem(x)
        s1 = self.enc1(self.pool(s0))
        s2 = self.enc2(s1)
        s3 = self.enc3(s2)
        s4 = self.enc4(s3)
        d = self.dec4(s4, s3)
        d = self.dec3(d, s2)
        d = self.dec2(d, s1)
        d = self.dec1(d, s0)
        d = self.dec0(d)
        return self.head(self.dropout(d))
