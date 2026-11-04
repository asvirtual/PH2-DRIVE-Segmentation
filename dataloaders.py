"""
Data loading for the PH2 and DRIVE segmentation datasets (single file:
shared utilities, PH2Dataset, DRIVEDataset and the loader factories).

Main features:
  1. Training seed: set_seed() + seed_worker() are wired into the DataLoaders,
     so augmentation, batch order and weight initialisation are reproducible.
  2. Every dataset returns (image, target, valid). `valid` marks the pixels that
     should count in the loss and the metrics (DRIVE: FOV; PH2: real image area,
     i.e. not padding). masked_mean() applies it to a per-pixel loss.
  3. Images keep their aspect ratio: downscale to fit image_size and pad with
     black (DRIVE 565x584 -> 512x512, PH2 765x572 -> 256x256). We do not use
     DRIVE at full resolution, to keep GPU memory use low (see project notes).
  4. Small random rotations in addition to flips, applied identically to the
     image and all its masks (training split only).
  5. foreground_stats() computes the class balance (for the report and pos_weight).
"""

from __future__ import annotations

import random
import re
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset, random_split

PathLike = Union[str, Path]

RESIZE_MODES = ("pad", "square")


# --------------------------------------------------------------------------
# Reproducibility
# --------------------------------------------------------------------------

def set_seed(seed: int = 42):
    """Fix all random number generators (training seed).

    Call this at the very beginning of every training run. It controls the
    weight initialisation, the batch order and the augmentation.
    The data split has its own seed (split_seed) and does NOT depend on this one.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def seed_worker(worker_id: int):
    """Seed NumPy/Python RNGs inside DataLoader workers (used via worker_init_fn)."""
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


# --------------------------------------------------------------------------
# Conversions
# --------------------------------------------------------------------------

def pil_resampling():
    """Return Pillow's resampling namespace across Pillow versions."""
    return getattr(Image, "Resampling", Image)


def pil_transpose():
    """Return Pillow's transpose namespace across Pillow versions."""
    return getattr(Image, "Transpose", Image)


def image_to_tensor(image: Image.Image) -> torch.Tensor:
    """Convert a PIL RGB image to float tensor [3, H, W] in [0, 1]."""
    array = np.asarray(image.convert("RGB"), dtype=np.float32) / 255.0
    return torch.from_numpy(array).permute(2, 0, 1).contiguous()


def mask_to_tensor(mask: Image.Image) -> torch.Tensor:
    """Convert a PIL mask to binary float tensor [1, H, W]."""
    array = np.asarray(mask.convert("L"))
    array = (array > 127).astype(np.float32)
    return torch.from_numpy(array).unsqueeze(0)


def full_valid_mask(image: Image.Image) -> Image.Image:
    """White mask with the size of the original image (every real pixel is valid)."""
    return Image.new("L", image.size, 255)


# --------------------------------------------------------------------------
# Resizing
# --------------------------------------------------------------------------

def fit_to_square(
    image: Image.Image, size: int, mode: str = "pad", is_mask: bool = False
) -> Image.Image:
    """
    Turn an image into size x size.

    mode="pad":    keep aspect ratio, downscale to fit, pad with black (default)
    mode="square": stretch to size x size (changes the aspect ratio)
    Masks always use nearest-neighbour so they stay binary.
    """
    if mode not in RESIZE_MODES:
        raise ValueError(f"resize mode must be one of {RESIZE_MODES}, got {mode!r}")

    resampling = pil_resampling()
    method = resampling.NEAREST if is_mask else resampling.BILINEAR
    image = image.convert("L") if is_mask else image.convert("RGB")

    if mode == "square":
        return image.resize((size, size), method)

    # mode == "pad": keep aspect ratio, then paste in the centre of a black canvas
    width, height = image.size
    scale = min(size / width, size / height)
    new_size = (max(1, round(width * scale)), max(1, round(height * scale)))
    image = image.resize(new_size, method)

    canvas = Image.new(image.mode, (size, size), 0)
    canvas.paste(image, ((size - new_size[0]) // 2, (size - new_size[1]) // 2))
    return canvas


def check_unet_size(size: int, levels: int = 4):
    """A U-Net with `levels` poolings needs sizes divisible by 2**levels."""
    if size % (2 ** levels) != 0:
        warnings.warn(
            f"image_size={size} is not divisible by {2 ** levels}; "
            f"a {levels}-level U-Net will have shape mismatches."
        )


# --------------------------------------------------------------------------
# Augmentation
# --------------------------------------------------------------------------

def paired_augment(
    images: List[Image.Image],
    is_mask: List[bool],
    max_rotation: float = 15.0,
) -> List[Image.Image]:
    """
    Apply the SAME random flips and rotation to an image and all its masks.

    The random numbers come from torch, so they are controlled by set_seed()
    (and by the DataLoader generator when num_workers > 0).
    Masks are rotated with nearest-neighbour (stay binary); the corners that
    rotate in are black, i.e. background in the masks and invalid in `valid`.
    """
    transpose = pil_transpose()
    resampling = pil_resampling()

    if torch.rand(()) < 0.5:
        images = [im.transpose(transpose.FLIP_LEFT_RIGHT) for im in images]
    if torch.rand(()) < 0.5:
        images = [im.transpose(transpose.FLIP_TOP_BOTTOM) for im in images]

    if max_rotation > 0:
        angle = float((torch.rand(()) * 2 - 1) * max_rotation)
        rotated = []
        for im, mask in zip(images, is_mask):
            if mask:
                rotated.append(im.rotate(angle, resample=resampling.NEAREST, fillcolor=0))
            else:
                rotated.append(im.rotate(angle, resample=resampling.BILINEAR, fillcolor=(0, 0, 0)))
        images = rotated

    return images


# --------------------------------------------------------------------------
# Loss / metric helpers
# --------------------------------------------------------------------------

def masked_mean(per_pixel: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
    """
    Average a per-pixel loss only over valid pixels.

    Example (BCE only inside the DRIVE FOV):
        loss_map = F.binary_cross_entropy_with_logits(logits, target, reduction="none")
        loss = masked_mean(loss_map, valid)
    """
    return (per_pixel * valid).sum() / valid.sum().clamp(min=1.0)


def foreground_stats(dataset: Dataset):
    """
    Fraction of foreground pixels (lesion / vessel) inside the valid area,
    and the matching pos_weight = background / foreground for weighted BCE.
    Use a dataset WITHOUT augmentation (e.g. the train split with augment=False).
    """
    foreground = 0.0
    valid_total = 0.0
    for index in range(len(dataset)):
        _, target, valid = dataset[index]
        foreground += (target * valid).sum().item()
        valid_total += valid.sum().item()
    fraction = foreground / max(valid_total, 1.0)
    pos_weight = (valid_total - foreground) / max(foreground, 1.0)
    return fraction, pos_weight


# --------------------------------------------------------------------------
# PH2 dataset (skin lesions)
# --------------------------------------------------------------------------

class PH2Dataset(Dataset):
    """
    PH2 dermoscopic images and binary lesion masks.

    Expected sample layout:
      IMDxxx/
        IMDxxx_Dermoscopic_Image/IMDxxx.bmp
        IMDxxx_lesion/IMDxxx_lesion.bmp

    ROI files are deliberately not used as segmentation targets.

    Returns (image, lesion_mask, valid):
      image        [3, H, W] float in [0, 1]
      lesion_mask  [1, H, W] binary (target)
      valid        [1, H, W] binary, 1 on real image pixels, 0 on padding /
                   corners rotated in by augmentation (use it to mask loss and metrics)
    """

    def __init__(
        self,
        root: Union[str, Path],
        image_size: int = 256,
        resize_mode: str = "pad",
        augment: bool = False,
        max_rotation: float = 15.0,
        indices: Optional[Sequence[int]] = None,
    ):
        self.root = Path(root)
        self.image_size = image_size
        self.resize_mode = resize_mode
        self.augment = augment
        self.max_rotation = max_rotation

        if not self.root.is_dir():
            raise FileNotFoundError(f"PH2 root directory not found: {self.root}")

        samples = []
        for sample_dir in sorted(self.root.glob("IMD*")):
            if not sample_dir.is_dir():
                continue

            sample_id = sample_dir.name
            image_path = sample_dir / f"{sample_id}_Dermoscopic_Image" / f"{sample_id}.bmp"
            mask_path = sample_dir / f"{sample_id}_lesion" / f"{sample_id}_lesion.bmp"

            if not image_path.is_file() or not mask_path.is_file():
                raise FileNotFoundError(
                    f"Incomplete PH2 sample {sample_id}: "
                    f"image exists={image_path.is_file()}, "
                    f"lesion mask exists={mask_path.is_file()}"
                )

            samples.append((sample_id, image_path, mask_path))

        if not samples:
            raise RuntimeError(f"No PH2 samples found in {self.root}")

        self.samples = samples
        if indices is not None:
            self.samples = [self.samples[i] for i in indices]

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        sample_id, image_path, mask_path = self.samples[index]

        image = Image.open(image_path).convert("RGB")
        mask = Image.open(mask_path).convert("L")
        valid = full_valid_mask(image)

        size, mode = self.image_size, self.resize_mode
        image = fit_to_square(image, size, mode, is_mask=False)
        mask = fit_to_square(mask, size, mode, is_mask=True)
        valid = fit_to_square(valid, size, mode, is_mask=True)

        if self.augment:
            image, mask, valid = paired_augment(
                [image, mask, valid], [False, True, True], self.max_rotation
            )

        return image_to_tensor(image), mask_to_tensor(mask), mask_to_tensor(valid)


# --------------------------------------------------------------------------
# DRIVE dataset (retinal vessels)
# --------------------------------------------------------------------------

SUPPORTED_EXTENSIONS = {".bmp", ".png", ".jpg", ".jpeg", ".tif", ".tiff", ".gif"}


def _index_by_image_number(folder: Path, description: str) -> Dict[str, Path]:
    """Index files by leading numeric ID, e.g. 21_training.tif -> '21'."""
    if not folder.is_dir():
        raise FileNotFoundError(f"{description} directory not found: {folder}")

    indexed = {}
    for path in sorted(folder.iterdir()):
        if not path.is_file() or path.suffix.lower() not in SUPPORTED_EXTENSIONS:
            continue

        match = re.match(r"^(\d+)", path.stem)
        if match is None:
            continue

        image_id = str(int(match.group(1)))
        if image_id in indexed:
            raise RuntimeError(
                f"Duplicate image ID {image_id} in {description}: "
                f"{indexed[image_id].name} and {path.name}"
            )
        indexed[image_id] = path

    return indexed


class DRIVEDataset(Dataset):
    """
    Labelled DRIVE training cases.

    Returns (image, vessel_mask, fov_mask). The vessel target is read from
    training/1st_manual; training/mask is the field-of-view (FOV) mask.
    The FOV mask plays the role of `valid`: use it to restrict the loss and the
    metrics to the retina (it is 0 on the black border and on the padding).

    resize_mode="pad" (default) keeps the aspect ratio: 565x584 is downscaled to
    fit image_size x image_size and padded with black.
    """

    def __init__(
        self,
        root: Union[str, Path],
        image_size: int = 512,
        resize_mode: str = "pad",
        augment: bool = False,
        max_rotation: float = 15.0,
        records: Optional[Sequence[Tuple[str, Path, Path, Path]]] = None,
    ):
        self.root = Path(root)
        self.image_size = image_size
        self.resize_mode = resize_mode
        self.augment = augment
        self.max_rotation = max_rotation

        if records is None:
            training = self.root / "training"
            image_files = _index_by_image_number(training / "images", "DRIVE training images")
            vessel_files = _index_by_image_number(training / "1st_manual", "DRIVE vessel annotations")
            fov_files = _index_by_image_number(training / "mask", "DRIVE training FOV masks")

            image_ids = set(image_files)
            if image_ids != set(vessel_files) or image_ids != set(fov_files):
                raise RuntimeError(
                    "DRIVE training image/annotation/FOV IDs do not match. "
                    f"Images={sorted(image_ids)}, "
                    f"vessel={sorted(vessel_files)}, FOV={sorted(fov_files)}"
                )

            ids = sorted(image_ids, key=int)
            records = [
                (image_id, image_files[image_id], vessel_files[image_id], fov_files[image_id])
                for image_id in ids
            ]

        self.records = list(records)
        if not self.records:
            raise RuntimeError("No labelled DRIVE training cases found.")

    def __len__(self):
        return len(self.records)

    def __getitem__(self, index):
        image_id, image_path, vessel_path, fov_path = self.records[index]

        image = Image.open(image_path).convert("RGB")
        vessel = Image.open(vessel_path).convert("L")
        fov = Image.open(fov_path).convert("L")

        size, mode = self.image_size, self.resize_mode
        image = fit_to_square(image, size, mode, is_mask=False)
        vessel = fit_to_square(vessel, size, mode, is_mask=True)
        fov = fit_to_square(fov, size, mode, is_mask=True)

        if self.augment:
            image, vessel, fov = paired_augment(
                [image, vessel, fov], [False, True, True], self.max_rotation
            )

        return image_to_tensor(image), mask_to_tensor(vessel), mask_to_tensor(fov)


class DRIVEOfficialTestDataset(Dataset):
    """
    Official DRIVE test images with FOV masks, for inference only
    (e.g. qualitative figures). The local dataset has no test vessel masks.
    Returns (image, fov_mask, image_id).
    """

    def __init__(self, root: Union[str, Path], image_size: int = 512, resize_mode: str = "pad"):
        self.root = Path(root)
        self.image_size = image_size
        self.resize_mode = resize_mode

        test_dir = self.root / "test"
        image_files = _index_by_image_number(test_dir / "images", "DRIVE test images")
        fov_files = _index_by_image_number(test_dir / "mask", "DRIVE test FOV masks")

        if set(image_files) != set(fov_files):
            raise RuntimeError(
                "DRIVE test image/FOV IDs do not match. "
                f"Images={sorted(image_files)}, FOV={sorted(fov_files)}"
            )

        ids = sorted(image_files, key=int)
        self.records = [(image_id, image_files[image_id], fov_files[image_id]) for image_id in ids]

    def __len__(self):
        return len(self.records)

    def __getitem__(self, index):
        image_id, image_path, fov_path = self.records[index]

        image = fit_to_square(Image.open(image_path).convert("RGB"), self.image_size, self.resize_mode)
        fov = fit_to_square(Image.open(fov_path).convert("L"), self.image_size, self.resize_mode, is_mask=True)

        return image_to_tensor(image), mask_to_tensor(fov), image_id


# --------------------------------------------------------------------------
# Splits and loaders
# --------------------------------------------------------------------------

def split_indices(dataset: Dataset, lengths: Sequence[int], seed: int = 42):
    """Return deterministic Subsets using a fixed split seed."""
    if sum(lengths) != len(dataset):
        raise ValueError(
            f"Split lengths {list(lengths)} sum to {sum(lengths)}, "
            f"but dataset contains {len(dataset)} samples."
        )
    generator = torch.Generator().manual_seed(seed)
    return random_split(dataset, list(lengths), generator=generator)


def make_data_loaders(
    train_dataset: Dataset,
    val_dataset: Dataset,
    test_dataset: Dataset,
    batch_size: int,
    num_workers: int = 0,
    train_seed: int = 42,
):
    """Create train/validation/test DataLoaders with reproducible shuffling and augmentation."""
    pin_memory = torch.cuda.is_available()

    def loader(dataset, shuffle):
        return DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            pin_memory=pin_memory,
            worker_init_fn=seed_worker,
            generator=torch.Generator().manual_seed(train_seed),
        )

    return loader(train_dataset, True), loader(val_dataset, False), loader(test_dataset, False)


def create_ph2_loaders(
    root: PathLike,
    batch_size: int = 8,
    image_size: int = 256,
    resize_mode: str = "pad",
    max_rotation: float = 15.0,
    split_seed: int = 42,
    train_seed: int = 42,
    num_workers: int = 0,
):
    """Create PH2 loaders using the agreed 140/30/30 split."""
    check_unet_size(image_size)
    full_dataset = PH2Dataset(root, image_size=image_size, resize_mode=resize_mode)
    if len(full_dataset) != 200:
        raise ValueError(
            f"Expected 200 PH2 cases for a 140/30/30 split; found {len(full_dataset)}."
        )

    subsets = split_indices(full_dataset, [140, 30, 30], seed=split_seed)
    train_idx, val_idx, test_idx = [subset.indices for subset in subsets]

    common = dict(image_size=image_size, resize_mode=resize_mode)
    train_dataset = PH2Dataset(root, augment=True, max_rotation=max_rotation, indices=train_idx, **common)
    val_dataset = PH2Dataset(root, augment=False, indices=val_idx, **common)
    test_dataset = PH2Dataset(root, augment=False, indices=test_idx, **common)

    loaders = make_data_loaders(
        train_dataset, val_dataset, test_dataset, batch_size, num_workers, train_seed
    )
    return loaders, (train_dataset, val_dataset, test_dataset)


def create_drive_loaders(
    root: PathLike,
    batch_size: int = 4,
    image_size: int = 512,
    resize_mode: str = "pad",
    max_rotation: float = 15.0,
    split_seed: int = 42,
    train_seed: int = 42,
    num_workers: int = 0,
):
    """
    Create DRIVE loaders from its 20 labelled training cases (12/4/4).
    """
    check_unet_size(image_size)
    full_dataset = DRIVEDataset(root, image_size=image_size, resize_mode=resize_mode)
    if len(full_dataset) != 20:
        raise ValueError(
            f"Expected 20 labelled DRIVE training cases for a 12/4/4 split; "
            f"found {len(full_dataset)}."
        )

    subsets = split_indices(full_dataset, [12, 4, 4], seed=split_seed)
    train_idx, val_idx, test_idx = [subset.indices for subset in subsets]

    def records(indices):
        return [full_dataset.records[i] for i in indices]

    common = dict(image_size=image_size, resize_mode=resize_mode)
    train_dataset = DRIVEDataset(root, augment=True, max_rotation=max_rotation, records=records(train_idx), **common)
    val_dataset = DRIVEDataset(root, augment=False, records=records(val_idx), **common)
    test_dataset = DRIVEDataset(root, augment=False, records=records(test_idx), **common)

    loaders = make_data_loaders(
        train_dataset, val_dataset, test_dataset, batch_size, num_workers, train_seed
    )
    return loaders, (train_dataset, val_dataset, test_dataset)


def create_drive_official_test_loader(
    root: PathLike,
    batch_size: int = 4,
    image_size: int = 512,
    resize_mode: str = "pad",
    num_workers: int = 0,
):
    """Create inference loader for official DRIVE test images (no vessel labels)."""
    dataset = DRIVEOfficialTestDataset(root, image_size=image_size, resize_mode=resize_mode)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )
    return loader, dataset