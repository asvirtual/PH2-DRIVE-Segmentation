"""Train and evaluate a segmentation model on PH2 or DRIVE.

Run from the repository root, for example:
    python train.py --dataset ph2 --loss bce --data-root ../data
    python train.py --dataset drive --loss focal --epochs 200 --data-root ../data
On the DTU HPC the default --data-root (/dtu/datasets1/02516) can be used.

Part 2 (point supervision on PH2): add --clicks-pos / --clicks-neg, e.g.
    python train.py --dataset ph2 --loss point_bce --clicks-pos 5 --clicks-neg 5
The training set then has only the simulated clicks; val/test keep full masks.

Each run writes to <out-dir>/<dataset>_<model>_<loss>_seed<seed>/
(Part 2: <dataset>_<model>_<loss>_clicks<pos>p<neg>n_<strategy>_seed<seed>/):
    best.pt          weights with the best validation Dice
    results.json     arguments, training history, and the five metrics
                     (mean and std over images) on train, val and test
    predictions.png  image / ground truth / prediction for a few test images
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import numpy as np
import torch

from dataloaders import create_drive_loaders, create_ph2_loaders
from engine import (
    compute_pos_weight,
    eval_loader,
    evaluate,
    fit,
    save_prediction_figure,
)
from losses import LOSS_NAMES, build_loss
from point_supervision import CLICK_STRATEGIES, create_ph2_click_loaders
from metrics import METRIC_NAMES
from unet import UNet, count_parameters
from EncDec import EncDec, count_parameters

DATASETS = {
    # name: (folder inside --data-root, loader factory from dataloaders.py)
    "ph2": ("PH2_Dataset_images", create_ph2_loaders),
    "drive": ("DRIVE", create_drive_loaders),
}


def parse_args():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--dataset", choices=DATASETS, required=True)
    parser.add_argument("--model", choices=["unet","encdec"], default="unet")
    parser.add_argument("--loss", choices=LOSS_NAMES, default="bce")
    parser.add_argument(
        "--data-root",
        type=Path,
        default=Path("/dtu/datasets1/02516"),
        help="folder that contains PH2_Dataset_images/ and DRIVE/",
    )
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument(
        "--batch-size", type=int, default=None, help="default: 8 for PH2, 4 for DRIVE"
    )
    parser.add_argument(
        "--image-size", type=int, default=None, help="default: 256 for PH2, 512 for DRIVE"
    )
    parser.add_argument("--base-channels", type=int, default=32)
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="training seed; the data split always uses split seed 42",
    )
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--out-dir", type=Path, default=Path("outputs"))
    parser.add_argument(
        "--max-batches",
        type=int,
        default=None,
        help="only use this many batches per pass (for quick smoke tests)",
    )
    # Part 2: point supervision (PH2 only). Leave --clicks-pos unset for full masks.
    parser.add_argument("--clicks-pos", type=int, default=None, help="positive clicks per training image")
    parser.add_argument("--clicks-neg", type=int, default=None, help="negative clicks per training image")
    parser.add_argument("--click-strategy", choices=list(CLICK_STRATEGIES), default="random")
    parser.add_argument("--click-radius", type=int, default=1, help="click = (2r+1)^2 square")
    parser.add_argument("--click-seed", type=int, default=0, help="seed for sampling the clicks")
    args = parser.parse_args()

    args.point_supervision = args.clicks_pos is not None or args.clicks_neg is not None
    if args.point_supervision:
        if args.dataset != "ph2":
            parser.error("point supervision (--clicks-pos/--clicks-neg) is only for --dataset ph2")
        args.clicks_pos = args.clicks_pos or 0
        args.clicks_neg = args.clicks_neg or 0
        if args.clicks_pos + args.clicks_neg == 0:
            parser.error("give at least one click")
        if args.loss == "weighted_bce":
            parser.error("weighted_bce needs full masks; use point_bce or bce with clicks")
    return args


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)  # also seeds CUDA


def build_model(args) -> torch.nn.Module:
    # Other models (e.g. the encoder-decoder baseline) can be added here.
    if args.model == "unet":
        return UNet(base_channels=args.base_channels)
    if args.model == "encdec":
        return EncDec(base_channels=args.base_channels)
    raise ValueError(f"Unknown model {args.model!r}")


def print_metrics(metrics):
    print(f"\n{'split':<7}" + "".join(f"{name:>18}" for name in METRIC_NAMES))
    for split, summary in metrics.items():
        cells = "".join(
            f"{summary[name]['mean']:>10.4f} ± {summary[name]['std']:.3f}"
            for name in METRIC_NAMES
        )
        print(f"{split:<7}{cells}")


def main():
    args = parse_args()
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    folder, create_loaders = DATASETS[args.dataset]
    sizes = {"batch_size": args.batch_size, "image_size": args.image_size}
    sizes = {key: value for key, value in sizes.items() if value is not None}
    if args.point_supervision:
        # Training set with clicks; train_set has the full masks, used only to
        # evaluate the training split (never for training).
        (train_loader, val_loader, test_loader), (click_set, train_set, _, _) = (
            create_ph2_click_loaders(
                args.data_root / folder,
                n_pos=args.clicks_pos,
                n_neg=args.clicks_neg,
                strategy=args.click_strategy,
                click_radius=args.click_radius,
                click_seed=args.click_seed,
                train_seed=args.seed,
                num_workers=args.num_workers,
                **sizes,
            )
        )
        placed = np.array(click_set.n_clicks)
        print(
            f"Point supervision: {args.clicks_pos} pos / {args.clicks_neg} neg clicks per image "
            f"({args.click_strategy}); placed on average {placed[:, 0].mean():.1f} pos / "
            f"{placed[:, 1].mean():.1f} neg"
        )
    else:
        (train_loader, val_loader, test_loader), (train_set, _, _) = create_loaders(
            args.data_root / folder, num_workers=args.num_workers, **sizes
        )
    # The training split is evaluated without augmentation.
    train_eval_loader = eval_loader(
        train_set, train_loader.batch_size, args.num_workers
    )

    pos_weight = (
        compute_pos_weight(train_eval_loader) if args.loss == "weighted_bce" else None
    )
    loss_fn = build_loss(args.loss, pos_weight).to(device)
    model = build_model(args).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    run_name = f"{args.dataset}_{args.model}_{args.loss}"
    if args.point_supervision:
        run_name += f"_clicks{args.clicks_pos}p{args.clicks_neg}n_{args.click_strategy}"
    run_dir = args.out_dir / f"{run_name}_seed{args.seed}"
    run_dir.mkdir(parents=True, exist_ok=True)
    print(
        f"{args.model} ({count_parameters(model) / 1e6:.2f}M parameters) on "
        f"{args.dataset}, loss={args.loss}"
        + (f" (pos_weight={pos_weight:.2f})" if pos_weight is not None else "")
        + f", device={device}, output={run_dir}"
    )

    best_epoch, history = fit(
        model,
        train_loader,
        val_loader,
        loss_fn,
        optimizer,
        device,
        args.epochs,
        run_dir / "best.pt",
        args.max_batches,
    )
    print(f"Best validation Dice at epoch {best_epoch}")

    metrics = {}
    for split, loader in (
        ("train", train_eval_loader),
        ("val", val_loader),
        ("test", test_loader),
    ):
        _, metrics[split] = evaluate(model, loader, device, max_batches=args.max_batches)
    print_metrics(metrics)
    save_prediction_figure(model, test_loader, device, run_dir / "predictions.png")

    results = {
        "args": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
        "pos_weight": pos_weight,
        "best_epoch": best_epoch,
        "history": history,
        "metrics": metrics,
    }
    if device.type == "cuda":
        results["peak_gpu_memory_gb"] = torch.cuda.max_memory_allocated() / 1e9
        print(f"Peak GPU memory: {results['peak_gpu_memory_gb']:.2f} GB")
    (run_dir / "results.json").write_text(json.dumps(results, indent=2))
    print(f"Saved results to {run_dir}")


if __name__ == "__main__":
    main()