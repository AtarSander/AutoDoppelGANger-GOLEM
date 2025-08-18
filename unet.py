from unet_def import ConvBlock, DownSample, UpSample
import torch
import torch.nn as nn

class UNet(nn.Module):
    def __init__(self, in_channels=3, out_channels=3, features=[32, 64, 64]):
        super().__init__()

        # Encoder
        self.down1 = DownSample(in_channels, features[0])
        self.down2 = DownSample(features[0], features[1])
        self.down3 = DownSample(features[1], features[2])

        # Bottleneck
        self.bottleneck = ConvBlock(features[2], features[2])

        # Decoder
        self.up3 = UpSample(features[2], features[1])
        self.up2 = UpSample(features[1], features[0])
        self.up1 = UpSample(features[0], features[0])

        # Output
        self.final_conv = nn.Conv2d(features[0], out_channels, kernel_size=1)

    def forward(self, x):
        # Encoder
        x1, x = self.down1(x)
        x2, x = self.down2(x)
        x3, x = self.down3(x)

        # Bottleneck
        x = self.bottleneck(x)

        # Decoder
        x = self.up3(x, x3)
        x = self.up2(x, x2)
        x = self.up1(x, x1)

        # Output
        return self.final_conv(x)
