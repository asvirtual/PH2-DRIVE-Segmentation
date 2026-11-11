# Project 3, Part 2: point supervision on PH2

Trained 2026-11-11 on Gustav's PC (NVIDIA RTX 5080, 16 GB), on branch `part2-point-supervision`.
The training code is Daniel Morales Ortiz's commit `2024860` (point-level loss, click dataloader and `train.py` options, 15:07 Copenhagen time) plus Susee's `weak_annotations.py` (`5d5e8db`, 16:15). A small uncommitted adapter in `point_supervision.py` registers Susee's `center` and `extreme` strategies so they can be passed to `--click-strategy`; her file was not edited. Machine-readable version: `part2_results.csv`. Raw per-run output (`results.json`, `predictions.png`, `best.pt`) is in `outputs/part2/`. Example click overlays (random, 5/5) are in `outputs/part2/clicks_examples.png`.

All numbers are **mean ± std over the images of the split**, computed against the **full masks** on every split (the clicks are used only as the training loss). A pixel is predicted foreground when the sigmoid output is > 0.5. Only pixels inside the `valid` mask (real image area, padding excluded) count. The fully supervised baseline is the Part 1 PH2 U-Net with BCE (`outputs/part1/ph2_unet_bce_seed42`, test Dice 0.925 ± 0.061).

> **Which numbers go in the report:** Tables 1–4 (`run_set = local_part2`).

## Table 1: test Dice by number of clicks and sampling strategy

Equal positive and negative clicks. Fully supervised baseline: **0.925 ± 0.061**. Best click run in **bold**.

| Clicks (pos / neg) | random | center | extreme |
|---|---|---|---|
| 1 / 1 | 0.877 ± 0.094 | 0.825 ± 0.134 | 0.646 ± 0.188 |
| 2 / 2 | 0.895 ± 0.068 | 0.876 ± 0.096 | 0.696 ± 0.175 |
| 5 / 5 | 0.907 ± 0.080 | 0.895 ± 0.098 | 0.869 ± 0.063 |
| 10 / 10 | 0.880 ± 0.095 | 0.903 ± 0.072 | 0.895 ± 0.069 |
| 25 / 25 | 0.902 ± 0.081 | 0.916 ± 0.044 | 0.911 ± 0.076 |
| 50 / 50 | **0.921 ± 0.060** | 0.917 ± 0.068 | 0.919 ± 0.053 |
| Full masks (Part 1, BCE) | 0.925 ± 0.061 | — | — |

## Table 2: validation Dice (same runs as Table 1)

Fully supervised baseline: **0.935 ± 0.044**.

| Clicks (pos / neg) | random | center | extreme |
|---|---|---|---|
| 1 / 1 | 0.858 ± 0.110 | 0.810 ± 0.123 | 0.728 ± 0.156 |
| 2 / 2 | 0.900 ± 0.082 | 0.890 ± 0.078 | 0.763 ± 0.155 |
| 5 / 5 | 0.924 ± 0.054 | 0.908 ± 0.069 | 0.891 ± 0.068 |
| 10 / 10 | 0.922 ± 0.054 | 0.928 ± 0.059 | 0.915 ± 0.056 |
| 25 / 25 | 0.930 ± 0.057 | 0.936 ± 0.044 | 0.934 ± 0.050 |
| 50 / 50 | 0.934 ± 0.042 | 0.934 ± 0.046 | 0.935 ± 0.048 |
| Full masks (Part 1, BCE) | 0.935 ± 0.044 | — | — |

## Table 3: test metrics for every equal-click run

| Strategy | Clicks | Val Dice | Test Dice | Test IoU | Test Accuracy | Test Sensitivity | Test Specificity | Best epoch |
|---|---|---|---|---|---|---|---|---|
| Full masks (BCE) | — | 0.935 ± 0.044 | 0.925 ± 0.061 | 0.866 ± 0.091 | 0.964 ± 0.027 | 0.946 ± 0.055 | 0.941 ± 0.120 | 95 |
| random | 1 / 1 | 0.858 ± 0.110 | 0.877 ± 0.094 | 0.792 ± 0.129 | 0.926 ± 0.092 | 0.882 ± 0.114 | 0.973 ± 0.045 | 35 |
| random | 2 / 2 | 0.900 ± 0.082 | 0.895 ± 0.068 | 0.816 ± 0.104 | 0.944 ± 0.059 | 0.907 ± 0.095 | 0.967 ± 0.045 | 90 |
| random | 5 / 5 | 0.924 ± 0.054 | 0.907 ± 0.080 | 0.839 ± 0.116 | 0.949 ± 0.065 | 0.933 ± 0.075 | 0.968 ± 0.042 | 97 |
| random | 10 / 10 | 0.922 ± 0.054 | 0.880 ± 0.095 | 0.797 ± 0.136 | 0.935 ± 0.105 | 0.921 ± 0.120 | 0.967 ± 0.032 | 85 |
| random | 25 / 25 | 0.930 ± 0.057 | 0.902 ± 0.081 | 0.831 ± 0.119 | 0.954 ± 0.039 | 0.968 ± 0.042 | 0.952 ± 0.045 | 86 |
| random | 50 / 50 | 0.934 ± 0.042 | **0.921 ± 0.060** | 0.860 ± 0.093 | 0.962 ± 0.029 | 0.958 ± 0.039 | 0.962 ± 0.044 | 92 |
| center | 1 / 1 | 0.810 ± 0.123 | 0.825 ± 0.134 | 0.721 ± 0.169 | 0.913 ± 0.094 | 0.852 ± 0.119 | 0.909 ± 0.158 | 10 |
| center | 2 / 2 | 0.890 ± 0.078 | 0.876 ± 0.096 | 0.791 ± 0.129 | 0.932 ± 0.074 | 0.881 ± 0.099 | 0.966 ± 0.062 | 90 |
| center | 5 / 5 | 0.908 ± 0.069 | 0.895 ± 0.098 | 0.821 ± 0.130 | 0.946 ± 0.060 | 0.879 ± 0.116 | 0.979 ± 0.037 | 59 |
| center | 10 / 10 | 0.928 ± 0.059 | 0.903 ± 0.072 | 0.831 ± 0.110 | 0.953 ± 0.037 | 0.960 ± 0.046 | 0.951 ± 0.052 | 64 |
| center | 25 / 25 | 0.936 ± 0.044 | 0.916 ± 0.044 | 0.848 ± 0.072 | 0.955 ± 0.048 | 0.940 ± 0.070 | 0.967 ± 0.030 | 91 |
| center | 50 / 50 | 0.934 ± 0.046 | 0.917 ± 0.068 | 0.853 ± 0.104 | 0.959 ± 0.040 | 0.948 ± 0.056 | 0.959 ± 0.063 | 82 |
| extreme | 1 / 1 | 0.728 ± 0.156 | 0.646 ± 0.188 | 0.505 ± 0.204 | 0.759 ± 0.177 | 0.923 ± 0.131 | 0.705 ± 0.220 | 14 |
| extreme | 2 / 2 | 0.763 ± 0.155 | 0.696 ± 0.175 | 0.562 ± 0.209 | 0.814 ± 0.144 | 0.917 ± 0.127 | 0.790 ± 0.179 | 5 |
| extreme | 5 / 5 | 0.891 ± 0.068 | 0.869 ± 0.063 | 0.773 ± 0.095 | 0.933 ± 0.052 | 0.944 ± 0.063 | 0.937 ± 0.041 | 86 |
| extreme | 10 / 10 | 0.915 ± 0.056 | 0.895 ± 0.069 | 0.816 ± 0.105 | 0.948 ± 0.044 | 0.967 ± 0.051 | 0.948 ± 0.033 | 96 |
| extreme | 25 / 25 | 0.934 ± 0.050 | 0.911 ± 0.076 | 0.845 ± 0.113 | 0.951 ± 0.063 | 0.956 ± 0.067 | 0.956 ± 0.066 | 79 |
| extreme | 50 / 50 | 0.935 ± 0.048 | 0.919 ± 0.053 | 0.855 ± 0.086 | 0.956 ± 0.058 | 0.946 ± 0.066 | 0.973 ± 0.032 | 88 |

## Table 4: positive vs negative balance at 10 clicks in total (random sampling)

| Clicks | Val Dice | Test Dice | Test IoU | Test Accuracy | Test Sensitivity | Test Specificity | Best epoch |
|---|---|---|---|---|---|---|---|
| 10 pos / 0 neg | 0.641 ± 0.219 | 0.519 ± 0.263 | 0.398 ± 0.273 | 0.539 ± 0.279 | 0.996 ± 0.011 | 0.355 ± 0.298 | 51 |
| 8 pos / 2 neg | 0.917 ± 0.060 | 0.850 ± 0.125 | 0.758 ± 0.173 | 0.923 ± 0.075 | 0.960 ± 0.070 | 0.915 ± 0.101 | 98 |
| 5 pos / 5 neg | 0.924 ± 0.054 | **0.907 ± 0.080** | 0.839 ± 0.116 | 0.949 ± 0.065 | 0.933 ± 0.075 | 0.968 ± 0.042 | 97 |
| 2 pos / 8 neg | 0.915 ± 0.062 | 0.904 ± 0.065 | 0.831 ± 0.101 | 0.945 ± 0.079 | 0.935 ± 0.089 | 0.957 ± 0.059 | 78 |

## Training setup (identical for every click run)

| | |
|---|---|
| Dataset / model | PH2, U-Net (`--base-channels 32`, 7,763,041 parameters) |
| Split (split seed 42) | 140 / 30 / 30 |
| Input size | 256×256 (aspect ratio kept, zero padding) |
| Augmentation (train only) | random h/v flips + rotation ±15° |
| Batch size / epochs / optimiser | 8 / 100 / Adam, lr 1e-3, no schedule |
| Loss | `point_bce` (partial BCE on clicked pixels, positive and negative clicks weighted equally) |
| Click sampling | `random` (Daniel, uniform inside the lesion and on real background); `center` and `extreme` (Susee: first positive near the lesion centroid, or at the top/bottom/left/right extremes, with remaining clicks random). Sampled once per training image after the 256 resize, drawn as 3×3 squares (`--click-radius 1`), `--click-seed 0` |
| Model selection | weights with the best validation Dice against the **full** validation masks |
| Training seed | 42 |
| Wall time | 1 m 50 s – 1 m 57 s per run, 38 m 57 s in total for 21 runs |
| Peak GPU memory | 1.65 GB |

## Observations for the report (from the numbers above)

- **A few random clicks already go a long way.** With 5 positive and 5 negative *random* clicks the test Dice is 0.907 ± 0.080, 0.018 below the fully supervised 0.925 ± 0.061, inside the 0.02 gap suggested in `part2_status.md`. Validation Dice at 5/5 random is 0.924 ± 0.054, close to the baseline 0.935 ± 0.044.
- **At 50/50 all three strategies are essentially tied with full supervision.** Test Dice is 0.921 (random), 0.917 (center) and 0.919 (extreme), gaps of 0.004–0.008. Validation Dice is 0.934–0.935 for all three.
- **The sampling strategy matters most when clicks are scarce.** At 1/1, random (0.877) beats center (0.825) and especially extreme (0.646). Extreme clicks sit on the lesion border, and with only one or two of them the model over-predicts lesion (1/1 extreme: sensitivity 0.923, specificity 0.705). Center needs about 25/25 to enter the 0.02 band (0.916); extreme likewise (0.911). Random gets there already at 5/5.
- **Test Dice is not strictly monotonic for random.** 10/10 random test Dice is 0.880, below both 5/5 and 2/2, while its validation Dice (0.922) sits between them. 25/25 random is 0.902, just outside the 0.02 band. With one training seed and 30 test images, gaps of about 0.02 are noisy (image-to-image std / √30 is about 0.011–0.017). Validation Dice is smoother for all three strategies.
- **Negative clicks matter.** On the random strategy, ten positive clicks and no negatives collapse: test Dice 0.518 ± 0.263, sensitivity 0.996, specificity 0.355 — the model over-predicts lesion. At 10 clicks in total, 5/5 (0.907) and 2/8 (0.904) are close; 8/2 drops to 0.850 with lower specificity (0.915).
- **Caveats.** All results use a single training seed (42). Checkpoints are selected on validation Dice against the full masks, the same protocol as Part 1; the report should say so. Susee's `random` implementation was not used for the random grid, so that those nine runs stay comparable with Daniel's sampler; `center` and `extreme` go through a thin adapter that crops out padding before calling her function.

## Answer to the PDF question

How many clicks are needed for comparable performance with the fully supervised model? Taking “comparable” as a test Dice within about 0.02 of 0.925:

- **random:** 5 positive and 5 negative clicks already qualify (0.907). 50/50 closes the remaining gap (0.921). Because 10/10 dips below the threshold on this seed, the 5-click result should be reported with the non-monotonic test curve and the smoother validation curve, not as a hard minimum.
- **center** and **extreme:** 25/25 qualifies (0.916 and 0.911); 50/50 is 0.917 and 0.919. At 1–5 clicks these strategies lag random, so the answer depends on how the simulated annotator clicks.
