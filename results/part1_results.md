# Project 3, Part 1: segmentation results (PH2 and DRIVE)

Regenerated 2026-11-08 on main at `1ba05b8`. All runs in Tables 1–4 use the unchanged `train.py` on Gustav's PC (NVIDIA RTX 5080, 16 GB). Machine-readable version: `part1_results.csv` (column `run_set` says which run set a row belongs to). Raw per-run output (results.json, best.pt, predictions.png, logs) is in `outputs/part1/` and `outputs/part1_diagnostic/` on Gustav's PC.

All numbers are **mean ± std over the images of the split**: each metric is computed per image, then averaged. A pixel is predicted foreground when the sigmoid output is > 0.5 (logit > 0). Only pixels inside the `valid` mask count: for PH2 the real image area (padding excluded), for DRIVE the field-of-view (FOV) mask.

> **Which numbers go in the report:** Tables 1–3 (`run_set = local_main`). **EncDec is the task-2 "simple encoder-decoder CNN"**, so there is no separate simple-CNN model to run. The older 50-epoch HPC runs committed under `outputs/` on main (fb879e7, 1ba05b8) should *not* be used: the DRIVE EncDec BCE and focal runs there never predicted a vessel (test Dice 0.000), see Table 5.

## Table 1: EncDec (simple CNN) vs U-Net, all three splits (BCE loss)

**PH2** (train/val/test = 140/30/30 images)

| Model | Split | Dice | IoU | Accuracy | Sensitivity | Specificity |
|---|---|---|---|---|---|---|
| EncDec (simple CNN) | train | 0.924 ± 0.045 | 0.863 ± 0.074 | 0.955 ± 0.046 | 0.939 ± 0.063 | 0.956 ± 0.061 |
| EncDec (simple CNN) | val | 0.936 ± 0.045 | 0.883 ± 0.073 | 0.953 ± 0.058 | 0.931 ± 0.073 | 0.973 ± 0.025 |
| EncDec (simple CNN) | test | 0.912 ± 0.048 | 0.842 ± 0.077 | 0.952 ± 0.062 | 0.915 ± 0.078 | 0.969 ± 0.040 |
| U-Net | train | 0.922 ± 0.060 | 0.860 ± 0.092 | 0.956 ± 0.041 | 0.958 ± 0.051 | 0.940 ± 0.092 |
| U-Net | val | 0.935 ± 0.044 | 0.880 ± 0.072 | 0.953 ± 0.047 | 0.944 ± 0.068 | 0.948 ± 0.097 |
| U-Net | test | 0.925 ± 0.061 | 0.866 ± 0.091 | 0.964 ± 0.027 | 0.946 ± 0.055 | 0.941 ± 0.120 |

**DRIVE** (train/val/test = 12/4/4 images)

| Model | Split | Dice | IoU | Accuracy | Sensitivity | Specificity |
|---|---|---|---|---|---|---|
| EncDec (simple CNN) | train | 0.626 ± 0.023 | 0.456 ± 0.024 | 0.916 ± 0.016 | 0.560 ± 0.042 | 0.968 ± 0.009 |
| EncDec (simple CNN) | val | 0.567 ± 0.092 | 0.401 ± 0.086 | 0.902 ± 0.018 | 0.496 ± 0.113 | 0.965 ± 0.004 |
| EncDec (simple CNN) | test | 0.599 ± 0.049 | 0.429 ± 0.048 | 0.915 ± 0.006 | 0.548 ± 0.092 | 0.964 ± 0.014 |
| U-Net | train | 0.828 ± 0.015 | 0.706 ± 0.021 | 0.958 ± 0.007 | 0.808 ± 0.032 | 0.980 ± 0.005 |
| U-Net | val | 0.777 ± 0.067 | 0.640 ± 0.084 | 0.942 ± 0.018 | 0.768 ± 0.093 | 0.969 ± 0.006 |
| U-Net | test | 0.784 ± 0.020 | 0.645 ± 0.027 | 0.949 ± 0.002 | 0.794 ± 0.082 | 0.970 ± 0.010 |

## Table 2: Loss ablation, test split

Best Dice per dataset and model in **bold**.

| Dataset | Model | Loss | Dice | IoU | Accuracy | Sensitivity | Specificity |
|---|---|---|---|---|---|---|---|
| PH2 | EncDec (simple CNN) | BCE | **0.912 ± 0.048** | 0.842 ± 0.077 | 0.952 ± 0.062 | 0.915 ± 0.078 | 0.969 ± 0.040 |
| PH2 | EncDec (simple CNN) | Weighted BCE | 0.911 ± 0.106 | 0.849 ± 0.126 | 0.949 ± 0.073 | 0.946 ± 0.046 | 0.940 ± 0.114 |
| PH2 | EncDec (simple CNN) | Focal (γ=2) | 0.910 ± 0.085 | 0.844 ± 0.114 | 0.957 ± 0.043 | 0.939 ± 0.060 | 0.942 ± 0.092 |
| PH2 | U-Net | BCE | **0.925 ± 0.061** | 0.866 ± 0.091 | 0.964 ± 0.027 | 0.946 ± 0.055 | 0.941 ± 0.120 |
| PH2 | U-Net | Weighted BCE | 0.923 ± 0.051 | 0.862 ± 0.084 | 0.963 ± 0.030 | 0.961 ± 0.036 | 0.938 ± 0.100 |
| PH2 | U-Net | Focal (γ=2) | 0.919 ± 0.042 | 0.853 ± 0.070 | 0.956 ± 0.054 | 0.918 ± 0.077 | 0.970 ± 0.037 |
| DRIVE | EncDec (simple CNN) | BCE | 0.599 ± 0.049 | 0.429 ± 0.048 | 0.915 ± 0.006 | 0.548 ± 0.092 | 0.964 ± 0.014 |
| DRIVE | EncDec (simple CNN) | Weighted BCE | 0.581 ± 0.037 | 0.410 ± 0.037 | 0.861 ± 0.029 | 0.802 ± 0.052 | 0.869 ± 0.038 |
| DRIVE | EncDec (simple CNN) | Focal (γ=2) | **0.609 ± 0.024** | 0.439 ± 0.025 | 0.908 ± 0.007 | 0.606 ± 0.058 | 0.949 ± 0.014 |
| DRIVE | U-Net | BCE | 0.784 ± 0.020 | 0.645 ± 0.027 | 0.949 ± 0.002 | 0.794 ± 0.082 | 0.970 ± 0.010 |
| DRIVE | U-Net | Weighted BCE | 0.750 ± 0.022 | 0.600 ± 0.028 | 0.928 ± 0.005 | 0.913 ± 0.023 | 0.930 ± 0.007 |
| DRIVE | U-Net | Focal (γ=2) | **0.788 ± 0.013** | 0.651 ± 0.017 | 0.947 ± 0.002 | 0.839 ± 0.039 | 0.961 ± 0.004 |

## Table 3: Loss ablation, validation split

Best Dice per dataset and model in **bold**.

| Dataset | Model | Loss | Dice | IoU | Accuracy | Sensitivity | Specificity |
|---|---|---|---|---|---|---|---|
| PH2 | EncDec (simple CNN) | BCE | **0.936 ± 0.045** | 0.883 ± 0.073 | 0.953 ± 0.058 | 0.931 ± 0.073 | 0.973 ± 0.025 |
| PH2 | EncDec (simple CNN) | Weighted BCE | 0.934 ± 0.043 | 0.878 ± 0.070 | 0.955 ± 0.036 | 0.945 ± 0.055 | 0.958 ± 0.046 |
| PH2 | EncDec (simple CNN) | Focal (γ=2) | 0.933 ± 0.044 | 0.878 ± 0.070 | 0.953 ± 0.050 | 0.935 ± 0.065 | 0.963 ± 0.039 |
| PH2 | U-Net | BCE | 0.935 ± 0.044 | 0.880 ± 0.072 | 0.953 ± 0.047 | 0.944 ± 0.068 | 0.948 ± 0.097 |
| PH2 | U-Net | Weighted BCE | **0.938 ± 0.053** | 0.887 ± 0.083 | 0.957 ± 0.039 | 0.952 ± 0.059 | 0.961 ± 0.057 |
| PH2 | U-Net | Focal (γ=2) | 0.937 ± 0.043 | 0.885 ± 0.072 | 0.955 ± 0.047 | 0.933 ± 0.062 | 0.961 ± 0.084 |
| DRIVE | EncDec (simple CNN) | BCE | 0.567 ± 0.092 | 0.401 ± 0.086 | 0.902 ± 0.018 | 0.496 ± 0.113 | 0.965 ± 0.004 |
| DRIVE | EncDec (simple CNN) | Weighted BCE | 0.562 ± 0.074 | 0.394 ± 0.068 | 0.850 ± 0.034 | 0.727 ± 0.105 | 0.869 ± 0.025 |
| DRIVE | EncDec (simple CNN) | Focal (γ=2) | **0.576 ± 0.084** | 0.409 ± 0.080 | 0.896 ± 0.020 | 0.547 ± 0.108 | 0.949 ± 0.004 |
| DRIVE | U-Net | BCE | **0.777 ± 0.067** | 0.640 ± 0.084 | 0.942 ± 0.018 | 0.768 ± 0.093 | 0.969 ± 0.006 |
| DRIVE | U-Net | Weighted BCE | 0.745 ± 0.068 | 0.598 ± 0.082 | 0.925 ± 0.019 | 0.837 ± 0.116 | 0.940 ± 0.005 |
| DRIVE | U-Net | Focal (γ=2) | 0.767 ± 0.076 | 0.627 ± 0.094 | 0.938 ± 0.022 | 0.773 ± 0.094 | 0.963 ± 0.011 |

## Table 4: DRIVE with 3× the training (1800 epochs, BCE), both models

The DRIVE EncDec had not converged at 600 epochs (best epochs 588–593), so both models were also trained for 1800 epochs with BCE, everything else identical. This is a robustness check of Table 1, not a replacement for it.

| Model | Epochs | Best epoch | Val Dice | Test Dice | Test IoU | Test Accuracy | Test Sensitivity | Test Specificity | Wall time |
|---|---|---|---|---|---|---|---|---|---|
| EncDec (simple CNN) | 600 | 593 | 0.567 ± 0.092 | 0.599 ± 0.049 | 0.429 ± 0.048 | 0.915 ± 0.006 | 0.548 ± 0.092 | 0.964 ± 0.014 | 2m 22s |
| EncDec (simple CNN) | 1800 | 1573 | 0.633 ± 0.106 | 0.673 ± 0.033 | 0.508 ± 0.037 | 0.924 ± 0.006 | 0.669 ± 0.076 | 0.958 ± 0.012 | 6m 59s |
| U-Net | 600 | 309 | 0.777 ± 0.067 | 0.784 ± 0.020 | 0.645 ± 0.027 | 0.949 ± 0.002 | 0.794 ± 0.082 | 0.970 ± 0.010 | 3m 51s |
| U-Net | 1800 | 419 | 0.767 ± 0.073 | 0.785 ± 0.018 | 0.647 ± 0.024 | 0.949 ± 0.001 | 0.801 ± 0.066 | 0.968 ± 0.008 | 11m 39s |

## Table 5: the 50-epoch HPC runs on main vs these runs (test split)

Jimena's runs (`outputs/<run>/results.json` on main, DTU HPC) used train.py defaults: 50 epochs for both datasets and `--base-channels 32` for EncDec too (65,921 parameters instead of the 260,865 of the EncDec author's default 64). With 3 steps per epoch, 50 DRIVE epochs are only 150 optimizer steps.

| Dataset | Model | Loss | HPC: epochs / best | HPC test Dice | HPC test Sens. | Local: epochs / best | Local test Dice | Local test Sens. |
|---|---|---|---|---|---|---|---|---|
| PH2 | EncDec (simple CNN) | BCE | 50 / 49 | 0.903 ± 0.086 | 0.905 ± 0.073 | 100 / 82 | 0.912 ± 0.048 | 0.915 ± 0.078 |
| PH2 | EncDec (simple CNN) | Weighted BCE | 50 / 49 | 0.890 ± 0.111 | 0.898 ± 0.089 | 100 / 99 | 0.911 ± 0.106 | 0.946 ± 0.046 |
| PH2 | EncDec (simple CNN) | Focal (γ=2) | 50 / 36 | 0.886 ± 0.099 | 0.909 ± 0.094 | 100 / 94 | 0.910 ± 0.085 | 0.939 ± 0.060 |
| PH2 | U-Net | BCE | 50 / 44 | 0.913 ± 0.075 | 0.925 ± 0.082 | 100 / 95 | 0.925 ± 0.061 | 0.946 ± 0.055 |
| PH2 | U-Net | Weighted BCE | 50 / 48 | 0.911 ± 0.085 | 0.906 ± 0.114 | 100 / 97 | 0.923 ± 0.051 | 0.961 ± 0.036 |
| PH2 | U-Net | Focal (γ=2) | 50 / 43 | 0.904 ± 0.094 | 0.916 ± 0.123 | 100 / 91 | 0.919 ± 0.042 | 0.918 ± 0.077 |
| DRIVE | EncDec (simple CNN) | BCE | 50 / 1 | 0.000 ± 0.000 ⚠ | 0.000 ± 0.000 | 600 / 593 | 0.599 ± 0.049 | 0.548 ± 0.092 |
| DRIVE | EncDec (simple CNN) | Weighted BCE | 50 / 12 | 0.105 ± 0.085 ⚠ | 0.274 ± 0.291 | 600 / 589 | 0.581 ± 0.037 | 0.802 ± 0.052 |
| DRIVE | EncDec (simple CNN) | Focal (γ=2) | 50 / 1 | 0.000 ± 0.000 ⚠ | 0.000 ± 0.000 | 600 / 588 | 0.609 ± 0.024 | 0.606 ± 0.058 |
| DRIVE | U-Net | BCE | 50 / 47 | 0.753 ± 0.018 | 0.757 ± 0.079 | 600 / 309 | 0.784 ± 0.020 | 0.794 ± 0.082 |
| DRIVE | U-Net | Weighted BCE | 50 / 28 | 0.725 ± 0.015 | 0.848 ± 0.052 | 600 / 392 | 0.750 ± 0.022 | 0.913 ± 0.023 |
| DRIVE | U-Net | Focal (γ=2) | 50 / 47 | 0.726 ± 0.018 | 0.741 ± 0.081 | 600 / 230 | 0.788 ± 0.013 | 0.839 ± 0.039 |

⚠ = degenerate. The DRIVE EncDec BCE and focal HPC runs predicted background everywhere (validation Dice 0.000 for all 50 epochs, so `best.pt` is the epoch-1 model), and weighted BCE stalled at a validation Dice of 0.25. Their accuracy of 0.88 and specificity of 1.00 look fine only because 88% of FOV pixels are background, a ready-made example of why accuracy is misleading on DRIVE. On PH2 and for the DRIVE U-Net the HPC runs are consistent with ours but 0.01–0.06 Dice lower (half the steps on PH2, 1/12 on DRIVE; EncDec also half as wide).

## Training setup (identical for every run of a dataset)

| | PH2 | DRIVE |
|---|---|---|
| Split (split seed 42) | 140 / 30 / 30 | 12 / 4 / 4 (of the 20 labelled DRIVE training images) |
| Input size | 256×256 (765×572 resized to 256×191, aspect ratio kept, zero padding) | 512×512 (565×584 resized to 495×512, aspect ratio kept, zero padding) |
| Augmentation (train only) | random h/v flips + rotation ±15° | random h/v flips + rotation ±15° |
| Batch size | 8 | 4 |
| Epochs | 100 (18 steps/epoch, 1800 steps) | 600 (3 steps/epoch, 1800 steps) |
| Optimizer | Adam, lr 1e-3, no schedule | Adam, lr 1e-3, no schedule |
| Model selection | weights with the best validation Dice | weights with the best validation Dice |
| pos_weight for weighted BCE (#background / #foreground, train set) | 2.10 | 6.96 (inside FOV) |
| Foreground fraction (train, valid pixels) | 32.2% lesion | 12.6% vessel |
| Training seed | 42 | 42 |

| Model | Architecture | Parameters |
|---|---|---|
| EncDec (simple CNN) | 4× (3×3 conv, ReLU, 2×2 max-pool), bottleneck 3×3 conv, 4× (bilinear ×2 upsample, 3×3 conv); 64 channels throughout (`--base-channels 64`); no skip connections, no BatchNorm | 260,865 |
| U-Net | 4 levels, (3×3 conv, BN, ReLU)×2 per block, channels 32-64-128-256, 512 bottleneck, 2×2 transposed-conv upsampling, skip connections by concatenation | 7,763,041 |

## Feature-map sizes (for the U-Net sketch, task 3)

Measured with forward hooks (`python3 unet.py`), C × H × W per block. Each encoder block except `enc1` starts with a 2×2 max-pool, so its input is shown after pooling. `dec` blocks take the upsampled map concatenated with the skip connection.

| Block | PH2 input (256) → output | DRIVE input (512) → output |
|---|---|---|
| enc1 | 3 × 256 × 256 → 32 × 256 × 256 | 3 × 512 × 512 → 32 × 512 × 512 |
| enc2 | 32 × 128 × 128 → 64 × 128 × 128 | 32 × 256 × 256 → 64 × 256 × 256 |
| enc3 | 64 × 64 × 64 → 128 × 64 × 64 | 64 × 128 × 128 → 128 × 128 × 128 |
| enc4 | 128 × 32 × 32 → 256 × 32 × 32 | 128 × 64 × 64 → 256 × 64 × 64 |
| bottleneck | 256 × 16 × 16 → 512 × 16 × 16 | 256 × 32 × 32 → 512 × 32 × 32 |
| up4 | 512 × 16 × 16 → 256 × 32 × 32 | 512 × 32 × 32 → 256 × 64 × 64 |
| dec4 | 512 × 32 × 32 → 256 × 32 × 32 | 512 × 64 × 64 → 256 × 64 × 64 |
| up3 | 256 × 32 × 32 → 128 × 64 × 64 | 256 × 64 × 64 → 128 × 128 × 128 |
| dec3 | 256 × 64 × 64 → 128 × 64 × 64 | 256 × 128 × 128 → 128 × 128 × 128 |
| up2 | 128 × 64 × 64 → 64 × 128 × 128 | 128 × 128 × 128 → 64 × 256 × 256 |
| dec2 | 128 × 128 × 128 → 64 × 128 × 128 | 128 × 256 × 256 → 64 × 256 × 256 |
| up1 | 64 × 128 × 128 → 32 × 256 × 256 | 64 × 256 × 256 → 32 × 512 × 512 |
| dec1 | 64 × 256 × 256 → 32 × 256 × 256 | 64 × 512 × 512 → 32 × 512 × 512 |
| head | 32 × 256 × 256 → 1 × 256 × 256 | 32 × 512 × 512 → 1 × 512 × 512 |

EncDec (64 channels): 3×S×S → 64×S/2 → 64×S/4 → 64×S/8 → 64×S/16 (bottleneck, 16×16 for PH2, 32×32 for DRIVE) → 64×S/8 → 64×S/4 → 64×S/2 → 1×S×S.

## Run times and best epochs

| Dataset | Model | Loss | Epochs | Best epoch (val Dice) | Wall time | Peak GPU memory |
|---|---|---|---|---|---|---|
| PH2 | EncDec (simple CNN) | BCE | 100 | 82 | 1m 16s | 0.69 GB |
| PH2 | EncDec (simple CNN) | Weighted BCE | 100 | 99 | 1m 12s | 0.69 GB |
| PH2 | EncDec (simple CNN) | Focal (γ=2) | 100 | 94 | 1m 13s | 0.69 GB |
| PH2 | U-Net | BCE | 100 | 95 | 1m 57s | 1.65 GB |
| PH2 | U-Net | Weighted BCE | 100 | 97 | 1m 54s | 1.65 GB |
| PH2 | U-Net | Focal (γ=2) | 100 | 91 | 1m 53s | 1.65 GB |
| DRIVE | EncDec (simple CNN) | BCE | 600 | 593 | 2m 22s | 1.37 GB |
| DRIVE | EncDec (simple CNN) | Weighted BCE | 600 | 589 | 2m 21s | 1.37 GB |
| DRIVE | EncDec (simple CNN) | Focal (γ=2) | 600 | 588 | 2m 19s | 1.37 GB |
| DRIVE | U-Net | BCE | 600 | 309 | 3m 51s | 3.13 GB |
| DRIVE | U-Net | Weighted BCE | 600 | 392 | 3m 50s | 3.13 GB |
| DRIVE | U-Net | Focal (γ=2) | 600 | 230 | 3m 58s | 3.13 GB |
| **Total, Tables 1–3** | | | | | **28m 06s** | |
| DRIVE | EncDec (simple CNN) | BCE (Table 4) | 1800 | 1573 | 6m 59s | 1.37 GB |
| DRIVE | U-Net | BCE (Table 4) | 1800 | 419 | 11m 39s | 3.13 GB |

## Observations for the report (from the numbers above)

- **PH2: EncDec ≈ U-Net.** Test Dice is 0.912 ± 0.048 for EncDec and 0.925 ± 0.061 for U-Net, a gap well inside the image-to-image spread. Lesions are large, compact blobs (about 32% of pixels), so the missing skip connections cost little.
- **DRIVE: U-Net ≫ EncDec, also at equal and longer training.** At 600 epochs the test Dice is 0.784 vs 0.599 and the sensitivity 0.79 vs 0.55. With 3× the training (Table 4) the U-Net is unchanged (0.785, best epoch 419, so it had converged), while the EncDec improves to 0.673. Its best epoch is 1573 and its validation Dice is still creeping up: 0.53, 0.57, 0.60, 0.61 and 0.62 at epochs 600, 900, 1200, 1500 and 1800 (60-epoch averages). The gap is therefore about 0.11 Dice at convergence, not 0.19. In `predictions.png` the EncDec finds only the thick vessels, drawn too wide, and misses the 1–2 px capillaries. Its 32×32 bottleneck is upsampled bilinearly without skip connections, so that detail cannot be recovered; the U-Net's skip connections bring it back. Without BatchNorm, the EncDec also needed about 300 epochs before it predicted any vessel with BCE or focal loss.
- **Loss ablation on PH2: no meaningful effect.** All three losses are within 0.006 test Dice of each other for both models.
- **Loss ablation on DRIVE: weighted BCE trades specificity for sensitivity.** U-Net sensitivity rises from 0.79 to 0.91 and EncDec sensitivity from 0.55 to 0.80. Specificity falls (U-Net 0.97 to 0.93, EncDec 0.96 to 0.87) and so does Dice, because pos_weight = 6.96 makes the network over-predict vessels at threshold 0.5. Focal loss ≈ BCE: U-Net test Dice 0.788 vs 0.784, EncDec 0.609 vs 0.599.
- **The ranking of the losses is not stable.** For the U-Net, BCE is best on DRIVE validation but focal on DRIVE test, and weighted BCE is best on PH2 validation but BCE on PH2 test. With one seed and only 4 DRIVE val/test images, differences of about 0.01 Dice are noise.
- **Metric weaknesses, visible in these numbers.** About 87% of DRIVE FOV pixels are background. So the weak EncDec still scores accuracy 0.915 and specificity 0.964 at Dice 0.599, and the degenerate all-background HPC run in Table 5 scores accuracy 0.882 and specificity 1.000 at Dice 0.000. Accuracy and specificity are therefore uninformative on DRIVE, while Dice, IoU and sensitivity separate the models. Per image, IoU = Dice / (2 − Dice), so the two carry the same information. Their means over images can still rank close runs differently: for PH2 EncDec on test, BCE has the higher Dice (0.912 vs 0.911) and weighted BCE the higher IoU (0.849 vs 0.842).
- **Caveats for the report.** All results use a single training seed (42). DRIVE has 12/4/4 images, so its val/test std comes from 4 images. Checkpoints are selected on validation Dice only; the test split is never used for selection.

