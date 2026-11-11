"""
Part 2: point-level (click) supervision on PH2.

What this file provides
-----------------------
1. Click sampling: `sample_clicks()` turns a full lesion mask into a few
   positive clicks (inside the lesion) and negative clicks (outside it).
   Strategies are registered in CLICK_STRATEGIES, so new ones (task 1, Susee)
   can be added without touching the rest of the code.
2. A PH2 training dataset with clicks instead of full masks
   (`PH2ClickDataset`). It returns (image, click_target, click_mask):
       click_target  1 on positive clicks, 0 elsewhere
       click_mask    1 on every clicked pixel (positive or negative), 0 elsewhere
   so the training loop and the losses work unchanged: the loss is only
   computed where click_mask == 1 (it is passed as `valid`).
3. `create_ph2_click_loaders()`: same 140/30/30 split, preprocessing and
   augmentation as Part 1 (create_ph2_loaders). Only the TRAINING set uses
   clicks; validation and test keep the full masks, so the metrics are directly
   comparable with the fully supervised U-Net.

Design choices (and why)
------------------------
- Clicks are sampled ONCE per image, with a fixed seed (click_seed + image ID),
  like a real annotator who clicks once. They do not change between epochs.
- Clicks are sampled on the resized 256x256 mask (after fit_to_square), so no
  click is lost by downscaling. Negative clicks are only sampled on real image
  pixels (not on the black padding).
- Each click is drawn as a small square (radius 1 -> 3x3 pixels), clipped to its
  own class (positive squares only cover lesion pixels, negative squares only
  background pixels), so no pixel gets a wrong label. A single pixel could be
  lost by the nearest-neighbour rotation of the augmentation; a 3x3 square
  survives it. The only clicks that can disappear in an epoch are those that
  the rotation moves out of the image (near the corners), like the pixels
  under them.
- The click maps go through the same flips/rotation as the image
  (paired_augment with is_mask=True), so clicks stay on the right pixels.
"""

from __future__ import annotations

from pathlib import Path
from typing import Callable, Dict, Optional, Tuple, Union

import numpy as np
from PIL import Image
from torch.utils.data import Dataset

from dataloaders import (
    check_unet_size,
    create_ph2_loaders,
    fit_to_square,
    full_valid_mask,
    image_to_tensor,
    make_data_loaders,
    mask_to_tensor,
    paired_augment,
)

PathLike = Union[str, Path]


# --------------------------------------------------------------------------
# Click sampling strategies
# --------------------------------------------------------------------------
# A strategy receives boolean arrays `lesion` and `background` (both [H, W],
# background already restricted to valid pixels), the number of clicks and a
# numpy Generator, and returns two arrays of (row, col) click positions.

def _choose(candidates: np.ndarray, n: int, rng: np.random.Generator) -> np.ndarray:
    """Pick n distinct (row, col) positions from a boolean candidate map."""
    coords = np.argwhere(candidates)
    if n <= 0 or len(coords) == 0:
        return np.zeros((0, 2), dtype=int)
    n = min(n, len(coords))
    return coords[rng.choice(len(coords), size=n, replace=False)]


def sample_random(lesion, background, n_pos, n_neg, rng):
    """Uniform random clicks inside the lesion and in the background."""
    return _choose(lesion, n_pos, rng), _choose(background, n_neg, rng)


# Add new strategies here (task 1), e.g. "away_from_border" or "negatives_near_lesion".
CLICK_STRATEGIES: Dict[str, Callable] = {
    "random": sample_random,
}


def draw_clicks(
    shape: Tuple[int, int],
    coords: np.ndarray,
    allowed: np.ndarray,
    radius: int = 1,
) -> np.ndarray:
    """Boolean map with a (2*radius+1)^2 square at each click, clipped to `allowed`."""
    out = np.zeros(shape, dtype=bool)
    h, w = shape
    for r, c in coords:
        out[max(r - radius, 0):min(r + radius + 1, h), max(c - radius, 0):min(c + radius + 1, w)] = True
    return out & allowed


def sample_clicks(
    mask: np.ndarray,
    valid: np.ndarray,
    n_pos: int,
    n_neg: int,
    strategy: str = "random",
    radius: int = 1,
    rng: Optional[np.random.Generator] = None,
):
    """
    Simulate an annotator's clicks from a full binary mask.

    mask, valid: boolean arrays [H, W] (lesion mask and real-image area).
    Returns (click_target, click_mask, pos_coords, neg_coords), where the first
    two are boolean maps [H, W] and the coords are arrays of (row, col).
    """
    if strategy not in CLICK_STRATEGIES:
        raise ValueError(f"Unknown click strategy {strategy!r}; choose from {list(CLICK_STRATEGIES)}")
    rng = np.random.default_rng() if rng is None else rng

    lesion = mask & valid
    background = ~mask & valid
    pos, neg = CLICK_STRATEGIES[strategy](lesion, background, n_pos, n_neg, rng)

    pos_map = draw_clicks(mask.shape, pos, lesion, radius)
    neg_map = draw_clicks(mask.shape, neg, background, radius)
    return pos_map, pos_map | neg_map, pos, neg


# --------------------------------------------------------------------------
# Dataset
# --------------------------------------------------------------------------

def _image_seed(click_seed: int, sample_id: str) -> int:
    """Per-image seed, e.g. IMD042 -> click_seed * 1000 + 42 (independent of split order)."""
    digits = "".join(ch for ch in sample_id if ch.isdigit())
    return click_seed * 1000 + int(digits or 0)


class PH2ClickDataset(Dataset):
    """
    PH2 training images with simulated clicks instead of full masks.

    `samples` is the list of (sample_id, image_path, mask_path) of the training
    split, as in PH2Dataset.samples. Returns (image, click_target, click_mask),
    all tensors [C, H, W] (image C=3, the click maps C=1).
    """

    def __init__(
        self,
        samples,
        n_pos: int,
        n_neg: int,
        strategy: str = "random",
        click_radius: int = 1,
        click_seed: int = 0,
        image_size: int = 256,
        resize_mode: str = "pad",
        augment: bool = True,
        max_rotation: float = 15.0,
    ):
        self.samples = list(samples)
        self.n_pos, self.n_neg = n_pos, n_neg
        self.strategy = strategy
        self.click_radius = click_radius
        self.click_seed = click_seed
        self.image_size = image_size
        self.resize_mode = resize_mode
        self.augment = augment
        self.max_rotation = max_rotation

        # Sample the clicks once, here, so they are fixed for the whole training.
        self.click_targets, self.click_masks = [], []
        self.n_clicks = []  # (positive, negative) clicks actually placed per image
        for sample_id, image_path, mask_path in self.samples:
            image = Image.open(image_path).convert("RGB")
            mask = fit_to_square(Image.open(mask_path).convert("L"), image_size, resize_mode, is_mask=True)
            valid = fit_to_square(full_valid_mask(image), image_size, resize_mode, is_mask=True)
            rng = np.random.default_rng(_image_seed(click_seed, sample_id))
            pos_map, click_map, pos, neg = sample_clicks(
                np.asarray(mask) > 127, np.asarray(valid) > 127,
                n_pos, n_neg, strategy, click_radius, rng,
            )
            self.click_targets.append(Image.fromarray(pos_map.astype(np.uint8) * 255))
            self.click_masks.append(Image.fromarray(click_map.astype(np.uint8) * 255))
            self.n_clicks.append((len(pos), len(neg)))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        _, image_path, _ = self.samples[index]
        image = fit_to_square(Image.open(image_path).convert("RGB"), self.image_size, self.resize_mode)
        click_target = self.click_targets[index]
        click_mask = self.click_masks[index]

        if self.augment:
            image, click_target, click_mask = paired_augment(
                [image, click_target, click_mask], [False, True, True], self.max_rotation
            )

        return image_to_tensor(image), mask_to_tensor(click_target), mask_to_tensor(click_mask)


# --------------------------------------------------------------------------
# Loader factory
# --------------------------------------------------------------------------

def create_ph2_click_loaders(
    root: PathLike,
    n_pos: int,
    n_neg: int,
    strategy: str = "random",
    click_radius: int = 1,
    click_seed: int = 0,
    batch_size: int = 8,
    image_size: int = 256,
    resize_mode: str = "pad",
    max_rotation: float = 15.0,
    split_seed: int = 42,
    train_seed: int = 42,
    num_workers: int = 0,
):
    """
    PH2 loaders for point supervision.

    Same split (140/30/30, split_seed), preprocessing and augmentation as
    create_ph2_loaders. The training loader returns clicks; validation and test
    return full masks. Also returns the full-mask training dataset, to evaluate
    the training split against the real masks (not against the clicks).

    Returns ((train_loader, val_loader, test_loader),
             (click_train_dataset, full_train_dataset, val_dataset, test_dataset))
    """
    check_unet_size(image_size)
    _, (full_train, val_dataset, test_dataset) = create_ph2_loaders(
        root, batch_size=batch_size, image_size=image_size, resize_mode=resize_mode,
        max_rotation=max_rotation, split_seed=split_seed, train_seed=train_seed,
        num_workers=num_workers,
    )
    click_train = PH2ClickDataset(
        full_train.samples, n_pos, n_neg, strategy, click_radius, click_seed,
        image_size, resize_mode, augment=True, max_rotation=max_rotation,
    )
    loaders = make_data_loaders(
        click_train, val_dataset, test_dataset, batch_size, num_workers, train_seed
    )
    return loaders, (click_train, full_train, val_dataset, test_dataset)