"""U-Net for binary segmentation of PH2 skin lesions and DRIVE retinal vessels.

This is the U-Net of Ronneberger et al. (2015) with two small changes:
  * the 3x3 convolutions use padding=1, so they keep H and W and the skip
    connections can be concatenated without cropping;
  * every convolution is followed by BatchNorm.

The network pools `depth` times, so H and W must be divisible by 2**depth.
With depth=4 this holds for both 256x256 (PH2) and 512x512 (DRIVE).
The output is one channel of logits; sigmoid(logits) > 0.5 (i.e. logits > 0)
gives the predicted mask.

Run `python unet.py` to print the feature-map shape of every layer.
"""

from __future__ import annotations

import torch
from torch import nn


def conv_block(in_channels: int, out_channels: int) -> nn.Sequential:
    """(3x3 conv -> BatchNorm -> ReLU) twice. The spatial size is unchanged."""
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace=True),
        nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace=True),
    )


class UNet(nn.Module):
    def __init__(
        self,
        in_channels: int = 3,
        out_channels: int = 1,
        base_channels: int = 32,
        depth: int = 4,
    ):
        super().__init__()
        self.depth = depth
        # Channels per level, e.g. [32, 64, 128, 256, 512]; the last is the bottleneck.
        channels = [base_channels * 2**level for level in range(depth + 1)]

        # Encoder: conv block, then 2x2 max-pool halves H and W.
        self.encoders = nn.ModuleList()
        previous = in_channels
        for level in range(depth):
            self.encoders.append(conv_block(previous, channels[level]))
            previous = channels[level]
        self.pool = nn.MaxPool2d(kernel_size=2)
        self.bottleneck = conv_block(channels[depth - 1], channels[depth])

        # Decoder, from the deepest level up: a 2x2 transposed conv doubles H and W
        # and halves the channels, then the skip connection is concatenated.
        self.ups = nn.ModuleList()
        self.decoders = nn.ModuleList()
        for level in reversed(range(depth)):
            self.ups.append(
                nn.ConvTranspose2d(
                    channels[level + 1], channels[level], kernel_size=2, stride=2
                )
            )
            self.decoders.append(conv_block(2 * channels[level], channels[level]))

        # 1x1 conv to one logit per pixel.
        self.head = nn.Conv2d(channels[0], out_channels, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        factor = 2**self.depth
        if x.shape[-2] % factor or x.shape[-1] % factor:
            raise ValueError(
                f"Input size {tuple(x.shape[-2:])} must be divisible by {factor}."
            )

        skips = []
        for encoder in self.encoders:
            x = encoder(x)
            skips.append(x)
            x = self.pool(x)

        x = self.bottleneck(x)

        for up, decoder, skip in zip(self.ups, self.decoders, reversed(skips)):
            x = up(x)
            x = decoder(torch.cat([skip, x], dim=1))

        return self.head(x)

    def named_layers(self):
        """Yield (name, module) in forward order, e.g. enc1 ... dec1, head."""
        for level, encoder in enumerate(self.encoders, start=1):
            yield f"enc{level}", encoder
        yield "bottleneck", self.bottleneck
        for index, (up, decoder) in enumerate(zip(self.ups, self.decoders)):
            level = self.depth - index
            yield f"up{level}", up
            yield f"dec{level}", decoder
        yield "head", self.head


@torch.no_grad()
def layer_shapes(model: UNet, image_size: int):
    """
    Pass a dummy image through the model and return a list of
    (layer name, input [C, H, W], output [C, H, W]) in forward order.
    """
    rows = []

    def record(name):
        def hook(module, inputs, output):
            rows.append((name, tuple(inputs[0].shape[1:]), tuple(output.shape[1:])))

        return hook

    hooks = [
        module.register_forward_hook(record(name))
        for name, module in model.named_layers()
    ]
    was_training = model.training
    model.eval()
    try:
        first_conv = model.encoders[0][0]
        device = first_conv.weight.device
        model(torch.zeros(1, first_conv.in_channels, image_size, image_size, device=device))
    finally:
        for hook in hooks:
            hook.remove()
        model.train(was_training)
    return rows


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


def format_shape(shape) -> str:
    return " x ".join(str(s) for s in shape)


def print_shape_table(model: UNet, image_size: int):
    print(f"\nInput 3 x {image_size} x {image_size}")
    print(f"{'layer':<12}{'input [C x H x W]':<22}output [C x H x W]")
    for name, in_shape, out_shape in layer_shapes(model, image_size):
        print(f"{name:<12}{format_shape(in_shape):<22}{format_shape(out_shape)}")


if __name__ == "__main__":
    unet = UNet()
    print(f"U-Net with {count_parameters(unet) / 1e6:.2f}M parameters")
    for size in (256, 512):
        print_shape_table(unet, size)
