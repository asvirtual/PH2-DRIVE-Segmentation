"""DRIVE retinal blood-vessel dataset."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Dict, Optional, Sequence, Tuple

from PIL import Image
from torch.utils.data import Dataset

from dataloaders import (
    image_to_tensor,
    mask_to_tensor,
    paired_random_flips,
    resize_and_pad,
)


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
    training/1st_manual; training/mask is the field-of-view mask.
    """

    def __init__(
        self,
        root: str | Path,
        image_size: int = 512,
        augment: bool = False,
        records: Optional[Sequence[Tuple[str, Path, Path, Path]]] = None,
    ):
        self.root = Path(root)
        self.image_size = image_size
        self.augment = augment

        if records is None:
            training = self.root / "training"
            image_files = _index_by_image_number(
                training / "images", "DRIVE training images"
            )
            vessel_files = _index_by_image_number(
                training / "1st_manual", "DRIVE vessel annotations"
            )
            fov_files = _index_by_image_number(
                training / "mask", "DRIVE training FOV masks"
            )

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
        vessel = Image.open(vessel_path)
        fov = Image.open(fov_path)

        # Preserve the DRIVE aspect ratio; do not stretch to a square.
        image = resize_and_pad(image, self.image_size, is_mask=False)
        vessel = resize_and_pad(vessel, self.image_size, is_mask=True)
        fov = resize_and_pad(fov, self.image_size, is_mask=True)

        image = image_to_tensor(image)
        vessel = mask_to_tensor(vessel)
        fov = mask_to_tensor(fov)

        if self.augment:
            image, vessel, fov = paired_random_flips(image, vessel, fov)

        return image, vessel, fov


class DRIVEOfficialTestDataset(Dataset):
    """
    Official DRIVE test images with FOV masks, for inference only.

    The inspected local dataset has no test vessel ground-truth masks.
    Returns (image, fov_mask, image_id).
    """

    def __init__(self, root: str | Path, image_size: int = 512):
        self.root = Path(root)
        self.image_size = image_size

        test_dir = self.root / "test"
        image_files = _index_by_image_number(
            test_dir / "images", "DRIVE test images"
        )
        fov_files = _index_by_image_number(
            test_dir / "mask", "DRIVE test FOV masks"
        )

        if set(image_files) != set(fov_files):
            raise RuntimeError(
                "DRIVE test image/FOV IDs do not match. "
                f"Images={sorted(image_files)}, FOV={sorted(fov_files)}"
            )

        ids = sorted(image_files, key=int)
        self.records = [
            (image_id, image_files[image_id], fov_files[image_id])
            for image_id in ids
        ]

    def __len__(self):
        return len(self.records)

    def __getitem__(self, index):
        image_id, image_path, fov_path = self.records[index]

        image = Image.open(image_path).convert("RGB")
        fov = Image.open(fov_path)

        image = resize_and_pad(image, self.image_size, is_mask=False)
        fov = resize_and_pad(fov, self.image_size, is_mask=True)

        return image_to_tensor(image), mask_to_tensor(fov), image_id
