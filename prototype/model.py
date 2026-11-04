"""U-Net with an exchangeable encoder, identical to Notebooks/3rd_iteration_*.ipynb section 6.

With encoder="resnet34" (the default, and what model cards without an "encoder" entry mean) the layers and
their names are those of the UNetResNet34 of the 1st and 2nd iteration, so all saved state dicts load unchanged.
The weights come from the checkpoint, so nothing is downloaded (pretrained=False). Dropout is inactive at inference.
"""
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


def adapt_first_conv(conv, in_ch):
    """The ImageNet first layer for in_ch inputs: mean of the RGB filters, scaled by 3 / in_ch (as before)."""
    new = nn.Conv2d(in_ch, conv.out_channels, conv.kernel_size, conv.stride, conv.padding, bias=conv.bias is not None)
    with torch.no_grad():
        new.weight.copy_(conv.weight.mean(1, keepdim=True).repeat(1, in_ch, 1, 1) * 3 / in_ch)
        if conv.bias is not None:
            new.bias.copy_(conv.bias)
    return new


# Channels of the 5 encoder outputs s0 (1/2 resolution) ... s4 (1/32)
ENCODER_CHANNELS = {
    "resnet34": (64, 64, 128, 256, 512),
    "resnet50": (64, 256, 512, 1024, 2048),
    "vgg16_bn": (128, 256, 512, 512, 512),
    "densenet121": (64, 256, 512, 1024, 1024),
    "efficientnet_b0": (16, 24, 40, 112, 320),
}


def build_encoder(name, in_ch, pretrained):
    """torchvision classifier -> (stem, pool, [enc1, enc2, enc3, enc4]). stem.0 is always the first convolution."""
    net = getattr(torchvision.models, name)(weights="DEFAULT" if pretrained else None)
    if name.startswith("resnet"):
        stem = nn.Sequential(adapt_first_conv(net.conv1, in_ch), net.bn1, net.relu)
        return stem, net.maxpool, [net.layer1, net.layer2, net.layer3, net.layer4]
    if name == "vgg16_bn":
        f = net.features   # 5 blocks of 3x3 conv + BN + ReLU, max pool after each (indices 6, 13, 23, 33, 43)
        stem = nn.Sequential(adapt_first_conv(f[0], in_ch), *f[1:13])
        return stem, nn.Identity(), [f[13:23], f[23:33], f[33:43], f[43:44]]
    if name == "densenet121":
        f = net.features
        stem = nn.Sequential(adapt_first_conv(f.conv0, in_ch), f.norm0, f.relu0)
        return stem, f.pool0, [f.denseblock1, nn.Sequential(f.transition1, f.denseblock2),
                               nn.Sequential(f.transition2, f.denseblock3),
                               nn.Sequential(f.transition3, f.denseblock4, f.norm5, nn.ReLU(inplace=True))]
    if name == "efficientnet_b0":
        f = net.features   # f[8], the 1x1 convolution to 1280 channels for the classifier, is not used
        stem = nn.Sequential(adapt_first_conv(f[0][0], in_ch), f[0][1], f[0][2], f[1])
        return stem, nn.Identity(), [f[2], f[3], nn.Sequential(f[4], f[5]), nn.Sequential(f[6], f[7])]
    raise ValueError(f"unknown encoder {name}; choose one of {list(ENCODER_CHANNELS)}")


class UNet(nn.Module):
    """U-Net with an exchangeable encoder. With resnet34 the layers and their names are identical to the
    UNetResNet34 of the 2nd iteration, so its weights load unchanged."""

    def __init__(self, in_ch, n_classes, encoder="resnet34", pretrained=False):
        super().__init__()
        ch = ENCODER_CHANNELS[encoder]
        self.stem, self.pool, (self.enc1, self.enc2, self.enc3, self.enc4) = build_encoder(encoder, in_ch, pretrained)
        self.dec4 = UpBlock(ch[4], ch[3], 256)
        self.dec3 = UpBlock(256, ch[2], 128)
        self.dec2 = UpBlock(128, ch[1], 64)
        self.dec1 = UpBlock(64, ch[0], 64)
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
