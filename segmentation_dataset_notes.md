# Image Segmentation Dataset Notes

## PH2 — skin lesions

- Location: `/dtu/datasets1/02516/PH2_Dataset_images`
- **200 samples**; each has one dermoscopic RGB image and one lesion annotation.
- File counts: 200 dermoscopic images, 200 lesion annotations, 50 additional ROI files (**450 files total**).
- Example (`IMD002`): image and lesion mask are both `765 × 572`; image mode is `RGB`, mask mode is `1` (binary).
- Validation check passed: **0 errors** across all 200 image/mask pairs (files present, dimensions match, masks contain two values).
- Use `*_Dermoscopic_Image/*.bmp` as inputs and `*_lesion/*.bmp` as segmentation targets. Do not use ROI files for the initial segmentation task.
- Agreed split: **140 train / 30 validation / 30 test**.
- Agreed image size: **256 × 256**.
- Resize images with bilinear interpolation and masks with nearest-neighbour interpolation. Convert images to RGB float tensors in `[0, 1]`; masks to binary float tensors.

## DRIVE — retinal vessels

- Dataset root: `/dtu/datasets1/02516` (folder structure inspected locally).
- Folder structure contains `training/images`, `training/1st_manual`, `training/mask`, `test/images`, and `test/mask`.
- Training set: **20 images**, **20 vessel annotations** in `training/1st_manual`, and **20 field-of-view (FOV) masks** in `training/mask`.
- Test set: **20 images** and **20 FOV masks**. No test vessel annotations were present in the inspected folder structure.
- Use `training/1st_manual` as the vessel segmentation target. The `mask` folder contains FOV masks, not vessel annotations.
- The inspected DRIVE images are **565 × 584**. Avoid distorting their aspect ratio when resizing to `512 × 512`; consider aspect-ratio-preserving resizing with padding, or another agreed approach.
- **Pending teacher clarification:** whether test vessel annotations are available/allowed.
  - If there are only 20 annotated training images: proposed split is **12 train / 4 validation / 4 test**.
  - If official test vessel annotations are provided and may be used: use the official 20 test images for final testing, and split the 20 annotated training images into training and validation sets. Agree on the exact training/validation split as a group.
- Calculate DRIVE metrics only within the FOV mask, excluding pixels outside the field of view.

## Shared experimental protocol

- Use a fixed **split seed of 42** so everyone uses exactly the same train/validation/test samples. This is separate from the training random seed.
- Apply augmentation only to the training set. Random flips are the agreed starting point; apply each geometric transformation identically to the image and its mask. Small rotations can be considered later.
- Keep the data split, preprocessing, augmentation, epochs, and learning rate consistent when comparing encoder-decoder versus U-Net and the three loss functions.
- For final robustness experiments, repeat training with different **training seeds** (for example, `42`, `7`, and `123`) while keeping the data split fixed. Report mean ± standard deviation where feasible.
- For PH2 weakly supervised experiments, preserve the same train/validation/test split as the fully supervised experiments.

## Python environment

The accompanying `requirements.txt` lists the libraries used by the current data-loader code: PyTorch, NumPy, and Pillow. Install PyTorch using the command appropriate for the machine and its CPU/CUDA setup if the default package installation is unsuitable.
