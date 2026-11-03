"""Segmentation metrics: Dice, IoU, accuracy, sensitivity and specificity.

Each metric is computed per image from the confusion counts (TP, FP, TN, FN),
then averaged over the images of a split (reported as mean and std).

`valid` restricts the counts to part of the image. For DRIVE it is the
field-of-view (FOV) mask, so the black border and the padding are not counted.
"""

from __future__ import annotations

import numpy as np
import torch

METRIC_NAMES = ("dice", "iou", "accuracy", "sensitivity", "specificity")

# Added to numerator and denominator, so an empty prediction of an empty target
# scores 1 instead of 0/0.
EPS = 1e-7


def confusion_counts(pred, target, valid=None):
    """Per-image TP, FP, TN, FN for binary tensors of shape [B, 1, H, W]."""
    pred = pred.bool()
    target = target.bool()
    valid = torch.ones_like(target) if valid is None else valid.bool()

    dims = tuple(range(1, pred.ndim))
    tp = (pred & target & valid).sum(dims)
    fp = (pred & ~target & valid).sum(dims)
    tn = (~pred & ~target & valid).sum(dims)
    fn = (~pred & target & valid).sum(dims)
    return tp.double(), fp.double(), tn.double(), fn.double()


def segmentation_metrics(pred, target, valid=None):
    """Return a dict of per-image metrics, each a tensor of shape [B]."""
    tp, fp, tn, fn = confusion_counts(pred, target, valid)
    return {
        "dice": (2 * tp + EPS) / (2 * tp + fp + fn + EPS),
        "iou": (tp + EPS) / (tp + fp + fn + EPS),
        "accuracy": (tp + tn + EPS) / (tp + tn + fp + fn + EPS),
        "sensitivity": (tp + EPS) / (tp + fn + EPS),  # recall of the foreground
        "specificity": (tn + EPS) / (tn + fp + EPS),  # recall of the background
    }


class MetricTracker:
    """Collect per-image metrics over a split and summarise them."""

    def __init__(self):
        self.values = {name: [] for name in METRIC_NAMES}

    def update(self, logits, target, valid=None):
        pred = logits > 0  # same as sigmoid(logits) > 0.5
        for name, per_image in segmentation_metrics(pred, target, valid).items():
            self.values[name].extend(per_image.tolist())

    def summary(self):
        """{metric: {"mean": ..., "std": ...}} over all images seen."""
        return {
            name: {"mean": float(np.mean(values)), "std": float(np.std(values))}
            for name, values in self.values.items()
        }
