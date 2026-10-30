"""PH2 skin-lesion dataset."""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Sequence

from PIL import Image
from torch.utils.data import Dataset

from dataloaders import (
    image_to_tensor,
    mask_to_tensor,
    paired_random_flips,
    resize_square,
)


class PH2Dataset(Dataset):
    """
    PH2 dermoscopic images and binary lesion masks.

    Expected sample layout:
      IMDxxx/
        IMDxxx_Dermoscopic_Image/IMDxxx.bmp
        IMDxxx_lesion/IMDxxx_lesion.bmp

    ROI files are deliberately not used as segmentation targets.
    """

    def __init__(
        self,
        root: str | Path,
        image_size: int = 256,
        augment: bool = False,
        indices: Optional[Sequence[int]] = None,
    ):
        self.root = Path(root)
        self.image_size = image_size
        self.augment = augment

        if not self.root.is_dir():
            raise FileNotFoundError(f"PH2 root directory not found: {self.root}")

        samples = []
        for sample_dir in sorted(self.root.glob("IMD*")):
            if not sample_dir.is_dir():
                continue

            sample_id = sample_dir.name
            image_path = (
                sample_dir / f"{sample_id}_Dermoscopic_Image" / f"{sample_id}.bmp"
            )
            mask_path = (
                sample_dir / f"{sample_id}_lesion" / f"{sample_id}_lesion.bmp"
            )

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
        mask = Image.open(mask_path)

        image = resize_square(image, self.image_size, is_mask=False)
        mask = resize_square(mask, self.image_size, is_mask=True)

        image = image_to_tensor(image)
        mask = mask_to_tensor(mask)

        if self.augment:
            image, mask, _ = paired_random_flips(image, mask)

        return image, mask
