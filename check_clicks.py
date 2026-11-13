"""
Sanity checks for Part 2 (point supervision on PH2).

Run from the repository root:
    python check_clicks.py

Checks that
  1. the split is exactly the Part 1 split (same 140/30/30 images);
  2. positive clicks lie only on lesion pixels and negative clicks only on real
     background pixels (never on the padding), before and after augmentation;
  3. the clicks are fixed: building the dataset twice gives the same clicks;
  4. the point loss only depends on the clicked pixels (zero gradient elsewhere);
and saves clicks_examples.png (image, full mask, clicks) for a few images.
"""

from pathlib import Path

import numpy as np
import torch
from matplotlib.figure import Figure

from dataloaders import create_ph2_loaders
from losses import PointBCELoss
from point_supervision import create_ph2_click_loaders

PH2_ROOT = Path("/dtu/datasets1/02516/PH2_Dataset_images")
N_POS, N_NEG = 5, 5


def ids(dataset):
    return [s[0] for s in dataset.samples]


def main():
    torch.manual_seed(0)
    (train_loader, _, _), (clicks, full_train, val, test) = create_ph2_click_loaders(
        PH2_ROOT, N_POS, N_NEG
    )

    # 1. Same split as Part 1
    _, (p1_train, p1_val, p1_test) = create_ph2_loaders(PH2_ROOT)
    assert ids(clicks) == ids(full_train) == ids(p1_train), "train split differs from Part 1"
    assert ids(val) == ids(p1_val) and ids(test) == ids(p1_test), "val/test split differs from Part 1"
    print(f"Split identical to Part 1: {len(clicks)}/{len(val)}/{len(test)}  OK")

    placed = np.array(clicks.n_clicks)
    print(f"Clicks placed per image: pos {placed[:, 0].min()}-{placed[:, 0].max()}, "
          f"neg {placed[:, 1].min()}-{placed[:, 1].max()} (asked {N_POS}/{N_NEG})")

    # 2a. With augmentation: every click survives the flips/rotation
    #     (each 3x3 click keeps at least one pixel).
    clicks.augment = True
    for i in range(len(clicks)):
        _, target, click_mask = clicks[i]
        pos = (target * click_mask).sum().item()
        neg = ((1 - target) * click_mask).sum().item()
        assert pos > 0 or placed[i, 0] == 0, f"image {i}: positive clicks lost by augmentation"
        assert neg > 0 or placed[i, 1] == 0, f"image {i}: negative clicks lost by augmentation"
        assert (target * (1 - click_mask)).sum() == 0, f"image {i}: target outside click mask"

    # 2b. Without augmentation: compare with the full mask and the valid area.
    full_train.augment = False
    clicks.augment = False
    for i in range(len(clicks)):
        _, target, click_mask = clicks[i]
        _, mask, valid = full_train[i]
        assert torch.all(target <= mask), f"image {i}: positive click outside the lesion"
        assert torch.all((click_mask - target) <= (1 - mask) * valid), (
            f"image {i}: negative click on the lesion or on the padding"
        )
    clicks.augment = True
    print("Positive clicks only on lesion, negative only on real background,"
          " clicks survive flips/rotations: OK")

    # 3. Clicks are fixed (same seed -> same clicks)
    _, (clicks2, _, _, _) = create_ph2_click_loaders(PH2_ROOT, N_POS, N_NEG)
    same = all(
        np.array_equal(np.asarray(a), np.asarray(b))
        for a, b in zip(clicks.click_masks, clicks2.click_masks)
    )
    assert same, "clicks change between runs"
    print("Clicks are reproducible: OK")

    # 4. The loss only looks at clicked pixels
    images, target, click_mask = next(iter(train_loader))
    logits = torch.randn_like(target, requires_grad=True)
    PointBCELoss()(logits, target, click_mask).backward()
    outside = logits.grad[click_mask == 0].abs().max().item()
    inside = logits.grad[click_mask == 1].abs().max().item()
    assert outside == 0 and inside > 0, "loss uses pixels outside the clicks"
    print(f"Loss gradient is zero outside the clicks and non-zero on them: OK "
          f"({int(click_mask.sum())} clicked pixels in the batch)")

    # Figure
    clicks.augment = False
    n = 4
    fig = Figure(figsize=(9, 3 * n))
    axes = fig.subplots(n, 3, squeeze=False)
    for row in range(n):
        image, target, click_mask = clicks[row]
        _, mask, _ = full_train[row]
        overlay = image.permute(1, 2, 0).numpy().copy()
        overlay[(click_mask[0] * target[0]).bool().numpy()] = (0, 1, 0)        # positive: green
        overlay[(click_mask[0] * (1 - target[0])).bool().numpy()] = (1, 0, 0)  # negative: red
        for ax, panel, title in zip(
            axes[row],
            (image.permute(1, 2, 0), mask[0], overlay),
            ("image", "full mask (not used)", f"clicks ({N_POS} pos / {N_NEG} neg)"),
        ):
            ax.imshow(panel, cmap="gray" if panel.ndim == 2 else None)
            ax.set_title(title)
            ax.axis("off")
    fig.tight_layout()
    fig.savefig("clicks_examples.png", dpi=100)
    print("Saved clicks_examples.png")
    print("\nALL CHECKS PASSED")


if __name__ == "__main__":
    main()