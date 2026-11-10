# Project 3, Part 2 (point supervision on PH2): status on 8 Nov 2026

**No Part 2 experiments have been run, because the Part 2 code does not exist yet.**
On 8 Nov (main at `1ba05b8`) there is no click simulation, no point-level loss and no
weak-label dataloader in this repository. I checked every remote branch (`main`, `unet`),
pull request #1 (dataloaders-v2, merged), the repo's forks (none) and the teammates' public
GitHub repos.

## What is missing, by owner

| # | Piece | Owner | What it has to do | Status |
|---|---|---|---|---|
| 1 | Click simulation (PDF task 1) | Susee | From each **training** mask, sample `n_pos` clicks inside the lesion and `n_neg` outside it, with at least two strategies (the PDF asks for random inside/outside, plus one more, e.g. biased towards the lesion centre or the boundary). Clicks are sampled **once per image with a fixed seed**, not again every epoch, because an annotator clicks once. They come only from valid (non-padded) pixels. | missing |
| 2 | Weak-label dataloader (PDF task 2) | Susee (with Daniel) | A PH2 **train** dataset that returns `(image, click_target, click_mask)`: `click_target` = 1 on positive clicks and 0 elsewhere, `click_mask` = 1 on every clicked pixel (positive or negative). Val and test keep the **full masks** from `create_ph2_loaders`, with the same 140/30/30 split (split seed 42). | missing |
| 3 | Point-level loss (PDF task 2) | Daniel | A loss over the clicked pixels only, called as `loss_fn(logits, target, valid)` like the Part 1 losses so that `engine.fit` works unchanged. | missing |
| 4 | Wiring in `train.py` | Daniel/Susee (needs Gustav's OK, it is a code change) | E.g. `--clicks-pos N --clicks-neg N --click-strategy S` switching the PH2 train loader to the click dataset, and a run-folder name that includes them. | missing |
| 5 | Click ablation runs + tables (PDF tasks 3–4) | Gustav | Run the grid on the RTX 5080 and compare with the fully supervised baseline below. | blocked on 1–4 |

## Things the code already gives you, and pitfalls to avoid

- **Loss:** `losses.BCELoss` already averages only over the pixels where `valid == 1`. Called
  with `valid = click_mask`, it is exactly a partial (point-wise) BCE. Daniel can use it
  as is or extend it, e.g. weighting positive and negative clicks equally when
  `n_pos != n_neg`.
- **Resolution:** sample the clicks **after** `fit_to_square()` (on the 256×256 mask), not on
  the original 765×572 mask. Masks are resized with nearest-neighbour, so a 1-pixel click
  drawn at full resolution survives the resize only about 1 time in 9. Alternatively,
  draw each click as a small disk (radius 2–3 px).
- **Augmentation:** pass `click_target` and `click_mask` through `paired_augment(...)` as
  masks (`is_mask=True`), so they get the same flips and rotation as the image. Single-pixel
  clicks can be dropped or duplicated by the nearest-neighbour rotation; small disks avoid
  this.
- **Train-split metrics:** `train.py` reports metrics on the train split using
  `eval_loader(train_set)`. If the train dataset returns click maps, those "train metrics"
  would be computed against the clicks. Evaluate the train split against the full masks,
  or drop the train row from the Part 2 tables.
- **Model selection:** `engine.fit` keeps the epoch with the best validation Dice against
  the **full** validation masks. This is the same protocol as Part 1, but it does use
  dense labels for 30 images, and the report should say so.
- **`--seed`:** `train.py` does not pass `train_seed=args.seed` to the loaders (reported on
  4 Nov, fix still awaiting approval). This only matters for multi-seed runs.

## Fully supervised baseline to compare against (Part 1, PH2, BCE, 100 epochs, seed 42)

| Model | Val Dice | Test Dice | Test IoU | Test Accuracy | Test Sensitivity | Test Specificity |
|---|---|---|---|---|---|---|
| U-Net | 0.935 ± 0.044 | 0.925 ± 0.061 | 0.866 ± 0.091 | 0.964 ± 0.027 | 0.946 ± 0.055 | 0.941 ± 0.120 |
| EncDec (simple CNN) | 0.936 ± 0.045 | 0.912 ± 0.048 | 0.842 ± 0.077 | 0.952 ± 0.062 | 0.915 ± 0.078 | 0.969 ± 0.040 |

Checkpoint: `outputs/part1/ph2_unet_bce_seed42/best.pt`, on Gustav's PC only. The hyperparameters
Part 2 should copy are PH2, U-Net `--base-channels 32`, 256×256 with padding, batch 8,
Adam lr 1e-3, 100 epochs, flips + ±15° rotation, seed 42, and the 140/30/30 split.

## Proposed ablation grid, once 1–4 exist (about 2 min per run on the RTX 5080)

- **Number of clicks**, positive = negative ∈ {1, 2, 5, 10, 25, 50}, for each of 2 sampling
  strategies: 12 runs, about 25 min.
- **Positive vs negative balance** at 10 clicks in total: (10 pos, 0 neg), (5, 5), (2, 8),
  (8, 2): 3 extra runs, about 6 min.
- **Answer to the PDF question:** "how many clicks for comparable performance" = the smallest
  click count whose test Dice is within about 0.02 of the 0.925 above. That gap is the
  image-to-image std / √30 ≈ 0.011, doubled.
