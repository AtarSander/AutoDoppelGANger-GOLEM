from unet_def import ConvBlock, DownSample, UpSample
import torch
import torch.nn as nn

class UNet(nn.Module):
    def __init__(self, in_channels=3, out_channels=3, features=[64, 128, 256, 512]):
        super().__init__()

        # Encoder
        self.down1 = DownSample(in_channels, features[0])
        self.down2 = DownSample(features[0], features[1])
        self.down3 = DownSample(features[1], features[2])
        self.down4 = DownSample(features[2], features[3])

        # Bottleneck
        self.bottleneck = ConvBlock(features[3], features[3] * 2)

        # Decoder
        self.up4 = UpSample(features[3] * 2, features[3])
        self.up3 = UpSample(features[3], features[2])
        self.up2 = UpSample(features[2], features[1])
        self.up1 = UpSample(features[1], features[0])

        # Output
        self.final_conv = nn.Conv2d(features[0], out_channels, kernel_size=1)

    def forward(self, x):
        # Encoder
        x1, x = self.down1(x)
        x2, x = self.down2(x)
        x3, x = self.down3(x)
        x4, x = self.down4(x)

        # Bottleneck
        x = self.bottleneck(x)

        # Decoder
        x = self.up4(x, x4)
        x = self.up3(x, x3)
        x = self.up2(x, x2)
        x = self.up1(x, x1)

        # Output
        return self.final_conv(x)
