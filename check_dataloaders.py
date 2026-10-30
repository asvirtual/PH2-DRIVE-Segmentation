"""
Sanity checks for the PH2 and DRIVE segmentation data loaders.

1. Edit PH2_ROOT and DRIVE_ROOT below.
2. Run from the directory containing PH2.py, DRIVE.py, and dataloaders.py:
       python check_dataloaders.py

The script checks split sizes, sample IDs, leakage, tensor shapes, image/mask
values, finite values, and (optionally) every sample in every split.
"""

from pathlib import Path
import torch

from dataloaders import (
    create_ph2_loaders,
    create_drive_loaders,
    create_drive_official_test_loader,
)

# EDIT THESE PATHS IF NEEDED
PH2_ROOT = Path("/dtu/datasets1/02516/PH2_Dataset_images")
DRIVE_ROOT = Path("/dtu/datasets1/02516/DRIVE")

# Set True to load and inspect every sample in every split (recommended once).
CHECK_EVERY_SAMPLE = True

EXPECTED_PH2_SPLITS = (140, 30, 30)
EXPECTED_DRIVE_SPLITS = (12, 4, 4)


def check_tensor(name, tensor, expected_channels=None, binary=False):
    assert isinstance(tensor, torch.Tensor), f"{name} is not a torch.Tensor"
    assert torch.isfinite(tensor).all(), f"{name} contains NaN or infinity"

    if expected_channels is not None:
        # Individual dataset samples are [C, H, W]; DataLoader batches are
        # [B, C, H, W]. Accept both layouts and check the channel axis.
        assert tensor.ndim in (3, 4), (
            f"{name} should have [C,H,W] or [B,C,H,W] dimensions; "
            f"got {tuple(tensor.shape)}"
        )
        channel_axis = 0 if tensor.ndim == 3 else 1
        assert tensor.shape[channel_axis] == expected_channels, (
            f"{name} expected {expected_channels} channels; "
            f"got {tensor.shape[channel_axis]}"
        )

    if binary:
        values = torch.unique(tensor)
        assert torch.all((values == 0) | (values == 1)), (
            f"{name} should be binary, but has values {values.tolist()}"
        )
    else:
        assert tensor.min() >= 0 and tensor.max() <= 1, (
            f"{name} should be in [0,1]; got range "
            f"[{tensor.min().item()}, {tensor.max().item()}]"
        )


def sample_ids(dataset, dataset_name):
    """Extract stable IDs from the split datasets for leakage checks."""
    if dataset_name == "PH2":
        return [sample[0] for sample in dataset.samples]
    if dataset_name == "DRIVE":
        return [record[0] for record in dataset.records]
    raise ValueError(dataset_name)


def check_split_ids(datasets, name, expected_sizes):
    ids = [sample_ids(ds, name) for ds in datasets]

    for split_name, split_ids, expected in zip(
        ("train", "validation", "test"), ids, expected_sizes
    ):
        assert len(split_ids) == expected, (
            f"{name} {split_name}: expected {expected}, got {len(split_ids)}"
        )
        assert len(set(split_ids)) == len(split_ids), (
            f"{name} {split_name} contains duplicate sample IDs"
        )

    train_ids, val_ids, test_ids = map(set, ids)
    assert train_ids.isdisjoint(val_ids), f"{name}: train/validation leakage"
    assert train_ids.isdisjoint(test_ids), f"{name}: train/test leakage"
    assert val_ids.isdisjoint(test_ids), f"{name}: validation/test leakage"

    all_ids = train_ids | val_ids | test_ids
    print(f"{name} split sizes: {[len(x) for x in ids]}")
    print(f"{name} unique samples across splits: {len(all_ids)}")
    print(f"{name} split IDs do not overlap: OK")


def inspect_segmentation_dataset(dataset, name, is_drive=False):
    """Read every item and verify dimensions, ranges, and binary labels."""
    print(f"\nInspecting every {name} sample...")
    for index in range(len(dataset)):
        item = dataset[index]

        if is_drive:
            image, vessel, fov = item
            check_tensor(f"{name}[{index}] image", image, 3)
            check_tensor(f"{name}[{index}] vessel mask", vessel, 1, binary=True)
            check_tensor(f"{name}[{index}] FOV mask", fov, 1, binary=True)

            assert image.shape[-2:] == vessel.shape[-2:] == fov.shape[-2:], (
                f"{name}[{index}]: image, vessel and FOV spatial sizes differ"
            )
            assert fov.sum() > 0, f"{name}[{index}]: FOV mask is empty"
        else:
            image, mask = item
            check_tensor(f"{name}[{index}] image", image, 3)
            check_tensor(f"{name}[{index}] lesion mask", mask, 1, binary=True)

            assert image.shape[-2:] == mask.shape[-2:], (
                f"{name}[{index}]: image and lesion-mask sizes differ"
            )

        assert image.shape[-2:] == (256, 256) if name.startswith("PH2") else True

    print(f"{name}: all {len(dataset)} samples passed")


def inspect_loader(loader, name, is_drive=False):
    """Inspect one actual batch produced by a DataLoader."""
    batch = next(iter(loader))

    if is_drive:
        images, vessel_masks, fov_masks = batch
        check_tensor(f"{name} images", images, 3)
        check_tensor(f"{name} vessel masks", vessel_masks, 1, binary=True)
        check_tensor(f"{name} FOV masks", fov_masks, 1, binary=True)

        assert images.shape[0] == vessel_masks.shape[0] == fov_masks.shape[0]
        assert images.shape[-2:] == vessel_masks.shape[-2:] == fov_masks.shape[-2:]
        print(
            f"{name}: images={tuple(images.shape)}, "
            f"vessel={tuple(vessel_masks.shape)}, FOV={tuple(fov_masks.shape)}"
        )
    else:
        images, masks = batch
        check_tensor(f"{name} images", images, 3)
        check_tensor(f"{name} masks", masks, 1, binary=True)

        assert images.shape[0] == masks.shape[0]
        assert images.shape[-2:] == masks.shape[-2:]
        print(
            f"{name}: images={tuple(images.shape)}, masks={tuple(masks.shape)}"
        )

    print(f"{name}: batch tensor checks passed")


def main():
    print("=" * 65)
    print("PH2 DATASET")
    print("=" * 65)
    print("Root:", PH2_ROOT)
    assert PH2_ROOT.is_dir(), f"PH2 directory not found: {PH2_ROOT}"

    ph2_loaders, ph2_datasets = create_ph2_loaders(
        PH2_ROOT, batch_size=8, image_size=256, split_seed=42
    )

    check_split_ids(ph2_datasets, "PH2", EXPECTED_PH2_SPLITS)
    for label, loader in zip(("train", "validation", "test"), ph2_loaders):
        inspect_loader(loader, f"PH2 {label}")

    if CHECK_EVERY_SAMPLE:
        for label, dataset in zip(("train", "validation", "test"), ph2_datasets):
            inspect_segmentation_dataset(dataset, f"PH2 {label}")

    print("\n" + "=" * 65)
    print("DRIVE DATASET")
    print("=" * 65)
    print("Root:", DRIVE_ROOT)
    assert DRIVE_ROOT.is_dir(), f"DRIVE directory not found: {DRIVE_ROOT}"

    drive_loaders, drive_datasets = create_drive_loaders(
        DRIVE_ROOT, batch_size=4, image_size=512, split_seed=42
    )

    check_split_ids(drive_datasets, "DRIVE", EXPECTED_DRIVE_SPLITS)
    for label, loader in zip(("train", "validation", "test"), drive_loaders):
        inspect_loader(loader, f"DRIVE {label}", is_drive=True)

    if CHECK_EVERY_SAMPLE:
        for label, dataset in zip(("train", "validation", "test"), drive_datasets):
            inspect_segmentation_dataset(
                dataset, f"DRIVE {label}", is_drive=True
            )

    print("\n" + "=" * 65)
    print("DRIVE OFFICIAL TEST SET (INFERENCE ONLY)")
    print("=" * 65)
    official_loader, official_dataset = create_drive_official_test_loader(
        DRIVE_ROOT, batch_size=4, image_size=512
    )
    print("Official test image count:", len(official_dataset))

    official_batch = next(iter(official_loader))
    images, fov_masks, image_ids = official_batch
    check_tensor("DRIVE official test images", images, 3)
    check_tensor("DRIVE official test FOV masks", fov_masks, 1, binary=True)
    assert images.shape[0] == fov_masks.shape[0]
    assert images.shape[-2:] == fov_masks.shape[-2:]
    print("Official test batch:", tuple(images.shape), tuple(fov_masks.shape))
    print("Example official test IDs:", list(image_ids))

    print("\n" + "=" * 65)
    print("ALL CHECKS PASSED")
    print("=" * 65)


if __name__ == "__main__":
    main()
