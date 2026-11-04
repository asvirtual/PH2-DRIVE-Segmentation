"""Quick checks of the simple Encoder-Decoder, metrics and losses.

Run before long training jobs:

    python check_encdec.py --data-root /dtu/datasets1/02516

Checks:
1. Encoder, bottleneck and decoder feature-map shapes for 256 and 512 inputs.
2. Metrics match a hand-computed example and ignore pixels outside a valid mask.
3. One real PH2 batch and one real DRIVE batch pass through EncDec and every loss.
4. EncDec can overfit a single PH2 batch (the loss drops).
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import torch
import torch.nn as nn

from dataloaders import create_drive_loaders, create_ph2_loaders
from encdec import EncDec
from engine import unpack_batch
from losses import LOSS_NAMES, build_loss
from metrics import segmentation_metrics


def expected_shapes(
    image_size: int,
    base_channels: int = 64,
    in_channels: int = 3,
    out_channels: int = 1,
):
    """Expected output [C,H,W] for each named layer in EncDec."""
    rows = []
    h = image_size
    c = in_channels

    # Encoder: convolution keeps resolution; pooling halves it.
    for index in range(4):
        rows.append((f"enc_conv{index}", (base_channels, h, h)))
        h //= 2
        rows.append((f"pool{index}", (base_channels, h, h)))
        c = base_channels

    # Bottleneck.
    rows.append(("bottleneck_conv", (base_channels, h, h)))

    # Decoder: upsampling doubles resolution; convolution keeps it.
    for index in range(4):
        h *= 2
        rows.append((f"upsample{index}", (base_channels, h, h)))
        channels = out_channels if index == 3 else base_channels
        rows.append((f"dec_conv{index}", (channels, h, h)))

    return rows


def measured_shapes(model: EncDec, image_size: int, device: torch.device):
    """Capture the output shape of each Conv2d, MaxPool2d and Upsample layer."""
    captured = {}
    handles = []

    layer_names = (
        "enc_conv0", "pool0",
        "enc_conv1", "pool1",
        "enc_conv2", "pool2",
        "enc_conv3", "pool3",
        "bottleneck_conv",
        "upsample0", "dec_conv0",
        "upsample1", "dec_conv1",
        "upsample2", "dec_conv2",
        "upsample3", "dec_conv3",
    )

    for name in layer_names:
        layer = getattr(model, name)

        def hook(_module, _inputs, output, layer_name=name):
            captured[layer_name] = tuple(output.shape[1:])

        handles.append(layer.register_forward_hook(hook))

    try:
        model.eval()
        with torch.no_grad():
            model(torch.randn(1, 3, image_size, image_size, device=device))
    finally:
        for handle in handles:
            handle.remove()

    return [(name, captured[name]) for name in layer_names]


def check_shapes(device: torch.device, base_channels: int = 64):
    model = EncDec(base_channels=base_channels).to(device)

    for size in (256, 512):
        measured = measured_shapes(model, size, device)
        expected = expected_shapes(size, base_channels=base_channels)
        assert measured == expected, (
            f"size {size}:\nMeasured: {measured}\nExpected: {expected}"
        )

        measured_dict = dict(measured)
        bottleneck = measured_dict["bottleneck_conv"]
        output = measured_dict["dec_conv3"]
        print(
            f"  {size}x{size}: all {len(measured)} layers match "
            f"(bottleneck {bottleneck}, output {output})"
        )


def check_metrics():
    # 8x8 image; target is a 3x3 square, prediction is the same square
    # shifted one pixel right: TP=6, FP=3, FN=3, TN=52.
    target = torch.zeros(1, 1, 8, 8)
    target[..., 2:5, 2:5] = 1
    pred = torch.zeros_like(target)
    pred[..., 2:5, 3:6] = 1

    expected = {
        "dice": 12 / 18,
        "iou": 6 / 12,
        "accuracy": 58 / 64,
        "sensitivity": 6 / 9,
        "specificity": 52 / 55,
    }

    for name, value in segmentation_metrics(pred, target).items():
        assert math.isclose(value.item(), expected[name], rel_tol=1e-6), name

    # Errors outside the valid mask must not count.
    valid = torch.zeros_like(target)
    valid[..., :, :6] = 1
    noisy = target.clone()
    noisy[..., :, 7] = 1
    with_mask = segmentation_metrics(noisy, target, valid)
    without_mask = segmentation_metrics(noisy, target)

    assert all(math.isclose(v.item(), 1.0) for v in with_mask.values())
    assert without_mask["dice"].item() < 1
    print("  hand-computed example and valid-mask handling OK")


def check_real_batches(
    data_root: Path,
    device: torch.device,
    base_channels: int = 64,
):
    model = EncDec(base_channels=base_channels).to(device)

    for name, folder, create_loaders in (
        ("PH2", "PH2_Dataset_images", create_ph2_loaders),
        ("DRIVE", "DRIVE", create_drive_loaders),
    ):
        (train_loader, _, _), _ = create_loaders(data_root / folder)
        images, targets, valid = unpack_batch(next(iter(train_loader)), device)

        logits = model(images)
        assert logits.shape == targets.shape, (logits.shape, targets.shape)

        losses = {}
        for loss_name in LOSS_NAMES:
            loss_fn = build_loss(loss_name, pos_weight=5.0).to(device)
            losses[loss_name] = loss_fn(logits, targets, valid).item()
            assert math.isfinite(losses[loss_name]), loss_name

        mask_info = "none" if valid is None else f"FOV {tuple(valid.shape)}"
        print(
            f"  {name}: images {tuple(images.shape)} -> "
            f"logits {tuple(logits.shape)}, valid mask: {mask_info}, losses: "
            + ", ".join(f"{key}={value:.3f}" for key, value in losses.items())
        )


def check_overfit(
    data_root: Path,
    device: torch.device,
    base_channels: int = 64,
    steps: int = 60,
):
    torch.manual_seed(0)
    model = EncDec(base_channels=base_channels).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = build_loss("bce").to(device)

    (train_loader, _, _), _ = create_ph2_loaders(data_root / "PH2_Dataset_images")
    images, targets, valid = unpack_batch(next(iter(train_loader)), device)

    model.train()
    first_loss = None
    for step in range(steps):
        logits = model(images)
        loss = loss_fn(logits, targets, valid)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if step == 0:
            first_loss = loss.item()

    with torch.no_grad():
        final_logits = model(images)
        final_loss = loss_fn(final_logits, targets, valid).item()
        dice = segmentation_metrics(final_logits > 0, targets, valid)["dice"].mean().item()

    print(
        f"  {steps} steps on one PH2 batch: "
        f"loss {first_loss:.3f} -> {final_loss:.3f}, Dice {dice:.3f}"
    )
    assert final_loss < 0.5 * first_loss, "loss did not drop sufficiently on a single batch"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-root",
        type=Path,
        default=Path("/dtu/datasets1/02516"),
    )
    parser.add_argument(
        "--base-channels",
        type=int,
        default=64,
        help="Hidden feature channels in EncDec (default: 64).",
    )
    parser.add_argument(
        "--overfit-steps",
        type=int,
        default=60,
        help="Number of optimization steps in the single-batch overfit check.",
    )
    args = parser.parse_args()

    if args.base_channels < 1:
        parser.error("--base-channels must be a positive integer")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {device}")
    print(f"EncDec base_channels: {args.base_channels}")

    print("1. EncDec layer shapes")
    check_shapes(device, args.base_channels)

    print("2. Metrics")
    check_metrics()

    print("3. Real batches through EncDec and losses")
    check_real_batches(args.data_root, device, args.base_channels)

    print("4. Overfit a single PH2 batch")
    check_overfit(args.data_root, device, args.base_channels, args.overfit_steps)

    print("\nALL CHECKS PASSED")


if __name__ == "__main__":
    main()
