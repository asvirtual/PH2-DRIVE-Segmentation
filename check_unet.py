"""Quick checks of the U-Net, metrics and losses, to run before long training jobs.

    python check_unet.py --data-root ../data

1. The feature-map shape of every U-Net layer matches the sketch (256 and 512 inputs).
2. The metrics match a hand-computed example and ignore pixels outside the valid mask.
3. One real PH2 batch and one real DRIVE batch pass through the model and every loss.
4. The model can overfit a single PH2 batch (the loss drops and Dice rises).
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import torch

from dataloaders import create_drive_loaders, create_ph2_loaders
from engine import unpack_batch
from losses import LOSS_NAMES, build_loss
from metrics import segmentation_metrics
from unet import UNet, layer_shapes


def sketch_shapes(image_size, base=32, depth=4, in_channels=3, out_channels=1):
    """The U-Net sketch from the report: (layer, input [C,H,W], output [C,H,W])."""
    ch = [base * 2**level for level in range(depth + 1)]
    hw = [image_size // 2**level for level in range(depth + 1)]
    rows = []
    previous = in_channels
    for level in range(depth):  # input of enc2..enc4 is the pooled previous level
        rows.append((f"enc{level + 1}", (previous, hw[level], hw[level]), (ch[level], hw[level], hw[level])))
        previous = ch[level]
    rows.append(("bottleneck", (ch[depth - 1], hw[depth], hw[depth]), (ch[depth], hw[depth], hw[depth])))
    for level in reversed(range(depth)):
        rows.append((f"up{level + 1}", (ch[level + 1], hw[level + 1], hw[level + 1]), (ch[level], hw[level], hw[level])))
        rows.append((f"dec{level + 1}", (2 * ch[level], hw[level], hw[level]), (ch[level], hw[level], hw[level])))
    rows.append(("head", (ch[0], hw[0], hw[0]), (out_channels, hw[0], hw[0])))
    return rows


def check_shapes(device):
    model = UNet().to(device)
    for size in (256, 512):
        measured = layer_shapes(model, size)
        expected = sketch_shapes(size)
        assert measured == expected, f"size {size}:\n{measured}\n!=\n{expected}"
        bottleneck = dict((name, out) for name, _, out in measured)["bottleneck"]
        print(f"  {size}x{size}: all {len(measured)} layers match the sketch (bottleneck {bottleneck})")


def check_metrics():
    # 8x8 image; target is a 3x3 square, prediction is the same square shifted
    # one pixel right: TP=6, FP=3, FN=3, TN=52.
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

    # Errors outside the valid (FOV) mask must not count.
    valid = torch.zeros_like(target)
    valid[..., :, :6] = 1
    noisy = target.clone()
    noisy[..., :, 7] = 1  # false positives only outside the valid mask
    with_mask = segmentation_metrics(noisy, target, valid)
    without_mask = segmentation_metrics(noisy, target)
    assert all(math.isclose(v.item(), 1.0) for v in with_mask.values())
    assert without_mask["dice"].item() < 1
    print("  hand-computed example and valid-mask handling OK")


def check_real_batches(data_root, device):
    model = UNet().to(device)
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
            f"  {name}: images {tuple(images.shape)} -> logits {tuple(logits.shape)}, "
            f"valid mask: {mask_info}, losses: "
            + ", ".join(f"{k}={v:.3f}" for k, v in losses.items())
        )


def check_overfit(data_root, device, steps=60):
    torch.manual_seed(0)
    model = UNet().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = build_loss("bce")
    (train_loader, _, _), _ = create_ph2_loaders(data_root / "PH2_Dataset_images")
    images, targets, _ = unpack_batch(next(iter(train_loader)), device)

    model.train()
    for step in range(steps):
        loss = loss_fn(model(images), targets)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if step == 0:
            first_loss = loss.item()

    with torch.no_grad():
        dice = segmentation_metrics(model(images) > 0, targets)["dice"].mean().item()
    print(f"  {steps} steps on one PH2 batch: loss {first_loss:.3f} -> {loss.item():.3f}, train-mode Dice {dice:.3f}")
    assert loss.item() < 0.5 * first_loss, "loss did not drop on a single batch"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=Path("/dtu/datasets1/02516"))
    args = parser.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {device}")

    print("1. U-Net layer shapes")
    check_shapes(device)
    print("2. Metrics")
    check_metrics()
    print("3. Real batches through the model and losses")
    check_real_batches(args.data_root, device)
    print("4. Overfit a single batch")
    check_overfit(args.data_root, device)
    print("\nALL CHECKS PASSED")


if __name__ == "__main__":
    main()
