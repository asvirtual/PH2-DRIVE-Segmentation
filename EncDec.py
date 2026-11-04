from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class EncDec(nn.Module):
 
    def __init__(
        self,
        in_channels: int = 3,
        out_channels: int = 1,
        base_channels: int = 64,
    ):
        super().__init__()
        features = base_channels
        # Encoder: convolution + ReLU + 2x2 max pooling.
        self.enc_conv0 = nn.Conv2d(in_channels, features, 3, padding=1)
        self.pool0 = nn.MaxPool2d(2, 2)

        self.enc_conv1 = nn.Conv2d(features, features, 3, padding=1)
        self.pool1 = nn.MaxPool2d(2, 2)

        self.enc_conv2 = nn.Conv2d(features, features, 3, padding=1)
        self.pool2 = nn.MaxPool2d(2, 2)

        self.enc_conv3 = nn.Conv2d(features, features, 3, padding=1)
        self.pool3 = nn.MaxPool2d(2, 2)

        # Bottleneck.
        self.bottleneck_conv = nn.Conv2d(features, features, 3, padding=1)

        # Decoder: 2x bilinear upsampling + convolution + ReLU.
        self.upsample0 = nn.Upsample(
            scale_factor=2, mode="bilinear", align_corners=False
        )
        self.dec_conv0 = nn.Conv2d(features, features, 3, padding=1)

        self.upsample1 = nn.Upsample(
            scale_factor=2, mode="bilinear", align_corners=False
        )
        self.dec_conv1 = nn.Conv2d(features, features, 3, padding=1)

        self.upsample2 = nn.Upsample(
            scale_factor=2, mode="bilinear", align_corners=False
        )
        self.dec_conv2 = nn.Conv2d(features, features, 3, padding=1)

        self.upsample3 = nn.Upsample(
            scale_factor=2, mode="bilinear", align_corners=False
        )
        self.dec_conv3 = nn.Conv2d(features, out_channels, 3, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.shape[-2] % 16 or x.shape[-1] % 16:
            raise ValueError(
                "Input height and width must be divisible by 16; "
                f"got {tuple(x.shape[-2:])}."
            )

        # Encoder.
        x = self.pool0(F.relu(self.enc_conv0(x)))
        x = self.pool1(F.relu(self.enc_conv1(x)))
        x = self.pool2(F.relu(self.enc_conv2(x)))
        x = self.pool3(F.relu(self.enc_conv3(x)))

        # Bottleneck.
        x = F.relu(self.bottleneck_conv(x))

        # Decoder.
        x = F.relu(self.dec_conv0(self.upsample0(x)))
        x = F.relu(self.dec_conv1(self.upsample1(x)))
        x = F.relu(self.dec_conv2(self.upsample2(x)))
        x = self.dec_conv3(self.upsample3(x))  # Keep raw logits; no sigmoid.

        return x


def count_parameters(model: nn.Module) -> int:
    """Count trainable parameters."""
    return sum(parameter.numel() for parameter in model.parameters()
               if parameter.requires_grad)


if __name__ == "__main__":
    model = EncDec()
    for size in (128, 256, 512):
        sample = torch.randn(2, 3, size, size)
        with torch.no_grad():
            output = model(sample)
        print(f"{size}x{size}: {tuple(sample.shape)} -> {tuple(output.shape)}")
    print(f"Trainable parameters: {count_parameters(model):,}")
