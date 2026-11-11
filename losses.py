"""Loss functions: BCE, weighted BCE and focal loss (Part 1 ablation) and
point-level BCE for click supervision (Part 2).

All losses take raw logits and a binary target of shape [B, 1, H, W], plus an
optional `valid` mask of the same shape. Only pixels where valid == 1 count:
  * DRIVE: the field-of-view mask, so the black border is ignored;
  * part 2 (weak supervision): the clicked pixels.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

LOSS_NAMES = ("bce", "weighted_bce", "focal", "point_bce")


def masked_mean(loss_map, valid=None):
    """Average a per-pixel loss over the valid pixels (all pixels if valid is None)."""
    if valid is None:
        return loss_map.mean()
    return (loss_map * valid).sum() / valid.sum().clamp_min(1)


class BCELoss(nn.Module):
    """
    Binary cross-entropy. With pos_weight, foreground pixels are weighted by
    pos_weight (e.g. #background / #foreground) to counter class imbalance.
    """

    def __init__(self, pos_weight: float | None = None):
        super().__init__()
        weight = None if pos_weight is None else torch.tensor([pos_weight])
        self.register_buffer("pos_weight", weight)

    def forward(self, logits, target, valid=None):
        loss_map = F.binary_cross_entropy_with_logits(
            logits, target, pos_weight=self.pos_weight, reduction="none"
        )
        return masked_mean(loss_map, valid)


class FocalLoss(nn.Module):
    """
    Focal loss (Lin et al., 2017): BCE scaled by (1 - p_t)^gamma, where p_t is
    the predicted probability of the true class. Easy, confident pixels get a
    small weight, so training focuses on the hard pixels.
    """

    def __init__(self, gamma: float = 2.0):
        super().__init__()
        self.gamma = gamma

    def forward(self, logits, target, valid=None):
        bce = F.binary_cross_entropy_with_logits(logits, target, reduction="none")
        p_t = torch.exp(-bce)  # bce = -log(p_t)
        loss_map = (1 - p_t) ** self.gamma * bce
        return masked_mean(loss_map, valid)


class PointBCELoss(nn.Module):
    """
    Partial (point-level) cross-entropy for click supervision (Part 2).

    Only the clicked pixels count: `valid` is the click mask (1 on every
    positive or negative click) and `target` is 1 on positive clicks.

    With balanced=True the positive and the negative clicks each contribute
    half of the loss,
        L = 1/2 * mean_{i in P} -log p_i  +  1/2 * mean_{i in N} -log(1 - p_i),
    so the result does not depend on how many clicks of each class there are
    (e.g. 2 positive and 8 negative clicks). With balanced=False it is the plain
    BCE averaged over all clicked pixels, i.e. BCELoss with valid = click mask.

    During validation `valid` is the full image area and `target` the full mask,
    so the same loss gives a class-balanced BCE over the whole image.
    """

    def __init__(self, balanced: bool = True):
        super().__init__()
        self.balanced = balanced

    def forward(self, logits, target, valid=None):
        bce = F.binary_cross_entropy_with_logits(logits, target, reduction="none")
        if valid is None:
            valid = torch.ones_like(target)
        if not self.balanced:
            return masked_mean(bce, valid)

        positive = target * valid
        negative = (1 - target) * valid
        terms = [masked_mean(bce, m) for m in (positive, negative) if m.sum() > 0]
        if not terms:  # no clicks in this batch: zero loss that keeps the graph
            return (logits * 0).sum()
        return torch.stack(terms).mean()


def build_loss(name: str, pos_weight: float | None = None) -> nn.Module:
    if name == "point_bce":
        return PointBCELoss(balanced=True)
    if name == "bce":
        return BCELoss()
    if name == "weighted_bce":
        if pos_weight is None:
            raise ValueError("weighted_bce needs a pos_weight.")
        return BCELoss(pos_weight)
    if name == "focal":
        return FocalLoss(gamma=2.0)
    raise ValueError(f"Unknown loss {name!r}; choose from {LOSS_NAMES}.")