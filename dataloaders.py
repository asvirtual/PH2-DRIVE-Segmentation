"""Shared utilities for the PH2 and DRIVE segmentation datasets."""

from __future__ import annotations

import random
from pathlib import Path
from typing import Optional, Sequence, Tuple

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset, random_split


def pil_resampling():
    """Return Pillow's resampling namespace across Pillow versions."""
    return getattr(Image, "Resampling", Image)


def image_to_tensor(image: Image.Image) -> torch.Tensor:
    """Convert a PIL RGB image to float tensor [3, H, W] in [0, 1]."""
    array = np.asarray(image.convert("RGB"), dtype=np.float32) / 255.0
    return torch.from_numpy(array).permute(2, 0, 1).contiguous()


def mask_to_tensor(mask: Image.Image) -> torch.Tensor:
    """Convert a PIL mask to binary float tensor [1, H, W]."""
    array = np.asarray(mask.convert("L"))
    array = (array > 0).astype(np.float32)
    return torch.from_numpy(array).unsqueeze(0)


def resize_square(
    image: Image.Image, size: int, is_mask: bool = False
) -> Image.Image:
    """Resize to size x size. Use nearest-neighbour for masks."""
    resampling = pil_resampling()
    method = resampling.NEAREST if is_mask else resampling.BILINEAR
    return image.resize((size, size), method)


def resize_and_pad(
    image: Image.Image, size: int, is_mask: bool = False
) -> Image.Image:
    """
    Preserve aspect ratio, resize to fit inside size x size, then pad centrally.
    Images receive black padding; masks receive background (zero) padding.
    """
    resampling = pil_resampling()
    method = resampling.NEAREST if is_mask else resampling.BILINEAR

    width, height = image.size
    scale = min(size / width, size / height)
    new_width = max(1, round(width * scale))
    new_height = max(1, round(height * scale))
    resized = image.resize((new_width, new_height), method)

    if is_mask:
        canvas = Image.new("L", (size, size), 0)
        resized = resized.convert("L")
    else:
        canvas = Image.new("RGB", (size, size), (0, 0, 0))
        resized = resized.convert("RGB")

    left = (size - new_width) // 2
    top = (size - new_height) // 2
    canvas.paste(resized, (left, top))
    return canvas


def paired_random_flips(
    image: torch.Tensor,
    mask: torch.Tensor,
    extra_mask: Optional[torch.Tensor] = None,
):
    """Apply identical random horizontal/vertical flips to paired tensors."""
    if torch.rand(()) < 0.5:
        image = torch.flip(image, dims=(-1,))
        mask = torch.flip(mask, dims=(-1,))
        if extra_mask is not None:
            extra_mask = torch.flip(extra_mask, dims=(-1,))

    if torch.rand(()) < 0.5:
        image = torch.flip(image, dims=(-2,))
        mask = torch.flip(mask, dims=(-2,))
        if extra_mask is not None:
            extra_mask = torch.flip(extra_mask, dims=(-2,))

    return image, mask, extra_mask


def make_data_loaders(
    train_dataset: Dataset,
    val_dataset: Dataset,
    test_dataset: Dataset,
    batch_size: int,
    num_workers: int = 0,
):
    """Create train/validation/test DataLoaders."""
    pin_memory = torch.cuda.is_available()

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
    return train_loader, val_loader, test_loader


def split_indices(
    dataset: Dataset,
    lengths: Sequence[int],
    seed: int = 42,
):
    """Return deterministic Subsets using a fixed split seed."""
    if sum(lengths) != len(dataset):
        raise ValueError(
            f"Split lengths {list(lengths)} sum to {sum(lengths)}, "
            f"but dataset contains {len(dataset)} samples."
        )
    generator = torch.Generator().manual_seed(seed)
    return random_split(dataset, list(lengths), generator=generator)


def seed_worker(worker_id: int):
    """Optionally seed NumPy/Python RNGs inside DataLoader workers."""
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def create_ph2_loaders(
    root: str | Path,
    batch_size: int = 8,
    image_size: int = 256,
    split_seed: int = 42,
    num_workers: int = 0,
):
    """Create PH2 loaders using the agreed 140/30/30 split."""
    from PH2 import PH2Dataset

    full_dataset = PH2Dataset(root, image_size=image_size, augment=False)
    if len(full_dataset) != 200:
        raise ValueError(
            f"Expected 200 PH2 cases for a 140/30/30 split; found {len(full_dataset)}."
        )

    subsets = split_indices(full_dataset, [140, 30, 30], seed=split_seed)
    train_indices, val_indices, test_indices = [
        subset.indices for subset in subsets
    ]

    # Separate dataset instances ensure augmentation is enabled only for training.
    train_dataset = PH2Dataset(
        root, image_size=image_size, augment=True, indices=train_indices
    )
    val_dataset = PH2Dataset(
        root, image_size=image_size, augment=False, indices=val_indices
    )
    test_dataset = PH2Dataset(
        root, image_size=image_size, augment=False, indices=test_indices
    )

    loaders = make_data_loaders(
        train_dataset, val_dataset, test_dataset, batch_size, num_workers
    )
    return loaders, (train_dataset, val_dataset, test_dataset)


def create_drive_loaders(
    root: str | Path,
    batch_size: int = 4,
    image_size: int = 512,
    split_seed: int = 42,
    num_workers: int = 0,
):
    """Create DRIVE loaders from its 20 labelled training cases (12/4/4)."""
    from DRIVE import DRIVEDataset

    full_dataset = DRIVEDataset(root, image_size=image_size, augment=False)
    if len(full_dataset) != 20:
        raise ValueError(
            f"Expected 20 labelled DRIVE training cases for a 12/4/4 split; "
            f"found {len(full_dataset)}."
        )

    subsets = split_indices(full_dataset, [12, 4, 4], seed=split_seed)
    train_indices, val_indices, test_indices = [
        subset.indices for subset in subsets
    ]

    train_records = [full_dataset.records[i] for i in train_indices]
    val_records = [full_dataset.records[i] for i in val_indices]
    test_records = [full_dataset.records[i] for i in test_indices]

    train_dataset = DRIVEDataset(
        root, image_size=image_size, augment=True, records=train_records
    )
    val_dataset = DRIVEDataset(
        root, image_size=image_size, augment=False, records=val_records
    )
    test_dataset = DRIVEDataset(
        root, image_size=image_size, augment=False, records=test_records
    )

    loaders = make_data_loaders(
        train_dataset, val_dataset, test_dataset, batch_size, num_workers
    )
    return loaders, (train_dataset, val_dataset, test_dataset)


def create_drive_official_test_loader(
    root: str | Path,
    batch_size: int = 4,
    image_size: int = 512,
    num_workers: int = 0,
):
    """Create inference loader for official DRIVE test images (no vessel labels)."""
    from DRIVE import DRIVEOfficialTestDataset

    dataset = DRIVEOfficialTestDataset(root, image_size=image_size)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )
    return loader, dataset
